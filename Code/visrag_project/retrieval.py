from __future__ import annotations

from pathlib import Path

import numpy as np

from .bm25 import search_bm25
from .config import DUAL_INDEX_BASELINES, IMAGE_INDEX_BASELINES, TEXT_INDEX_BASELINES
from .data import load_queries
from .embedding import VisRAGEncoder, batched
from .indexing import load_index, load_text_index
from .io_utils import ensure_dir, write_jsonl

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def retrieval_dir(data_root: str | Path, baseline: str, dataset: str) -> Path:
    return Path(data_root) / "runs" / baseline / dataset


def _topk_search(query_vectors: np.ndarray, doc_vectors: np.ndarray, ids: list[str], topk: int) -> list[list[dict]]:
    if len(ids) == 0:
        return [[] for _ in range(query_vectors.shape[0])]
    k = min(topk, len(ids))
    scores = query_vectors @ doc_vectors.T
    top_indices = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    results = []
    for row_idx, candidate_idx in enumerate(top_indices):
        ordered = candidate_idx[np.argsort(-scores[row_idx, candidate_idx])]
        results.append(
            [
                {
                    "doc_id": ids[int(idx)],
                    "score": float(scores[row_idx, int(idx)]),
                    "rank": rank + 1,
                }
                for rank, idx in enumerate(ordered)
            ]
        )
    return results


def run_retrieval(
    dataset: str,
    baseline: str,
    data_root: str | Path,
    encoder: VisRAGEncoder | None,
    *,
    topk: int = 5,
    batch_size: int = 16,
    overwrite: bool = False,
) -> dict:
    out_dir = retrieval_dir(data_root, baseline, dataset)
    jsonl_path = out_dir / "retrieval.jsonl"
    if jsonl_path.exists() and not overwrite:
        return {"dataset": dataset, "baseline": baseline, "output_path": str(jsonl_path), "skipped": True}

    queries = load_queries(data_root, dataset)
    rows = []

    image_index = None
    text_index = None
    needs_image = baseline in IMAGE_INDEX_BASELINES or baseline in DUAL_INDEX_BASELINES
    needs_text = baseline in TEXT_INDEX_BASELINES or baseline in DUAL_INDEX_BASELINES
    if needs_image:
        if encoder is None:
            raise ValueError(f"Baseline {baseline} needs the VisRAG image encoder.")
        image_index = load_index(data_root, "image", dataset)
    if needs_text:
        text_index = load_text_index(data_root, dataset)

    query_batches = list(batched(queries, batch_size))
    iterator = tqdm(query_batches, desc=f"{dataset} {baseline} retrieval", total=len(query_batches)) if tqdm else query_batches
    for batch in iterator:
        image_hits = [[] for _ in batch]
        text_hits = [[] for _ in batch]

        if image_index is not None:
            query_vectors = encoder.encode_queries([row["query"] for row in batch])
            image_hits = _topk_search(query_vectors, image_index[0], image_index[1], topk)
        if text_index is not None:
            text_hits = [search_bm25(text_index[0], row["query"], topk) for row in batch]

        for query_row, img_hits, txt_hits in zip(batch, image_hits, text_hits):
            if baseline in IMAGE_INDEX_BASELINES:
                primary_hits = img_hits
            elif baseline in TEXT_INDEX_BASELINES:
                primary_hits = txt_hits
            else:
                primary_hits = _merge_dual_hits(img_hits, txt_hits)
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "query_id": query_row["query_id"],
                    "query": query_row["query"],
                    "answer": query_row.get("answer"),
                    "query_record": query_row,
                    "image_hits": img_hits,
                    "text_hits": txt_hits,
                    "hits": primary_hits[:topk],
                }
            )

    ensure_dir(out_dir)
    write_jsonl(jsonl_path, rows)
    write_trec(out_dir / "run.trec", rows, hit_key="hits")
    if baseline in DUAL_INDEX_BASELINES:
        write_trec(out_dir / "image_run.trec", rows, hit_key="image_hits")
        write_trec(out_dir / "text_run.trec", rows, hit_key="text_hits")
    return {"dataset": dataset, "baseline": baseline, "rows": len(rows), "output_path": str(jsonl_path)}


def _merge_dual_hits(image_hits: list[dict], text_hits: list[dict]) -> list[dict]:
    merged = {}
    for modality, hits in (("image", image_hits), ("text", text_hits)):
        for hit in hits:
            doc_id = hit["doc_id"]
            item = merged.setdefault(doc_id, {"doc_id": doc_id, "score": 0.0, "modalities": []})
            item["score"] += 1.0 / max(int(hit["rank"]), 1)
            item["modalities"].append(modality)
    ordered = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(ordered, start=1):
        item["rank"] = rank
    return ordered


def write_trec(path: str | Path, rows: list[dict], *, hit_key: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            for hit in row.get(hit_key, []):
                f.write(
                    f"{row['query_id']}\tQ0\t{hit['doc_id']}\t{hit['rank']}\t{hit['score']}\t{row['baseline']}\n"
                )
