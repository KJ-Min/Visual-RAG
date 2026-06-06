from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .config import DATASETS, HF_DATASET_TEMPLATE
from .io_utils import ensure_dir, read_jsonl, safe_filename, write_jsonl


def validate_dataset_name(dataset: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of: {', '.join(DATASETS)}")


def hf_repo_name(dataset: str) -> str:
    validate_dataset_name(dataset)
    return HF_DATASET_TEMPLATE.format(dataset=dataset)


def raw_dataset_dir(data_root: str | Path, dataset: str) -> Path:
    validate_dataset_name(dataset)
    return Path(data_root) / "raw" / dataset


def manifest_path(data_root: str | Path, dataset: str) -> Path:
    return raw_dataset_dir(data_root, dataset) / "manifest.jsonl"


def queries_path(data_root: str | Path, dataset: str) -> Path:
    return raw_dataset_dir(data_root, dataset) / "queries.jsonl"


def qrels_path(data_root: str | Path, dataset: str) -> Path:
    return raw_dataset_dir(data_root, dataset) / "qrels.jsonl"


def image_dir(data_root: str | Path, dataset: str) -> Path:
    return raw_dataset_dir(data_root, dataset) / "images"


def load_hf_split(dataset: str, name: str, streaming: bool = False):
    from datasets import load_dataset

    return load_dataset(hf_repo_name(dataset), name=name, split="train", streaming=streaming)


def _save_image(image: Image.Image, path: Path) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f".{path.name}.tmp")
    image.convert("RGB").save(tmp_path, format="PNG")
    with Image.open(tmp_path) as saved_image:
        saved_image.load()
    tmp_path.replace(path)


def _with_progress(iterable, *, total=None, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def materialize_dataset(
    dataset: str,
    data_root: str | Path,
    *,
    max_docs: int | None = None,
    max_queries: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Download/load a VisRAG HF dataset and save a stable local manifest."""
    validate_dataset_name(dataset)
    out_dir = raw_dataset_dir(data_root, dataset)
    imgs_dir = image_dir(data_root, dataset)
    ensure_dir(imgs_dir)

    manifest_file = manifest_path(data_root, dataset)
    queries_file = queries_path(data_root, dataset)
    qrels_file = qrels_path(data_root, dataset)

    if manifest_file.exists() and queries_file.exists() and qrels_file.exists() and not overwrite:
        return summarize_raw_dataset(data_root, dataset)

    corpus_rows = []
    corpus = load_hf_split(dataset, "corpus")
    corpus_total = min(len(corpus), max_docs) if max_docs is not None else len(corpus)
    for idx, item in enumerate(_with_progress(corpus, total=corpus_total, desc=f"{dataset} corpus images")):
        if max_docs is not None and idx >= max_docs:
            break
        doc_id = str(item["corpus-id"])
        img_path = imgs_dir / safe_filename(doc_id, ".png")
        if overwrite or not img_path.exists():
            _save_image(item["image"], img_path)
        corpus_rows.append(
            {
                "dataset": dataset,
                "doc_id": doc_id,
                "image_path": str(img_path),
                "source_index": idx,
            }
        )
    write_jsonl(manifest_file, corpus_rows)

    query_rows = []
    queries = load_hf_split(dataset, "queries")
    query_total = min(len(queries), max_queries) if max_queries is not None else len(queries)
    for idx, item in enumerate(_with_progress(queries, total=query_total, desc=f"{dataset} queries")):
        if max_queries is not None and idx >= max_queries:
            break
        row = dict(item)
        row["dataset"] = dataset
        row["query_id"] = str(row.pop("query-id", row.get("query_id", row.get("id", idx))))
        query_rows.append(row)
    write_jsonl(queries_file, query_rows)

    qrel_rows = []
    qrels = load_hf_split(dataset, "qrels")
    allowed_queries = {row["query_id"] for row in query_rows}
    allowed_docs = {row["doc_id"] for row in corpus_rows}
    for item in _with_progress(qrels, total=len(qrels), desc=f"{dataset} qrels"):
        qid = str(item["query-id"])
        doc_id = str(item["corpus-id"])
        if allowed_queries and qid not in allowed_queries:
            continue
        if allowed_docs and doc_id not in allowed_docs:
            continue
        qrel_rows.append(
            {
                "dataset": dataset,
                "query_id": qid,
                "doc_id": doc_id,
                "score": int(item.get("score", 1)),
            }
        )
    write_jsonl(qrels_file, qrel_rows)

    return summarize_raw_dataset(data_root, dataset)


def load_manifest(data_root: str | Path, dataset: str) -> list[dict]:
    path = manifest_path(data_root, dataset)
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}. Run prepare_data.py first.")
    return list(read_jsonl(path))


def load_queries(data_root: str | Path, dataset: str) -> list[dict]:
    path = queries_path(data_root, dataset)
    if not path.exists():
        raise FileNotFoundError(f"Missing queries: {path}. Run prepare_data.py first.")
    return list(read_jsonl(path))


def load_qrels(data_root: str | Path, dataset: str) -> dict[str, dict[str, int]]:
    path = qrels_path(data_root, dataset)
    if not path.exists():
        raise FileNotFoundError(f"Missing qrels: {path}. Run prepare_data.py first.")
    qrels: dict[str, dict[str, int]] = {}
    for row in read_jsonl(path):
        qrels.setdefault(str(row["query_id"]), {})[str(row["doc_id"])] = int(row.get("score", 1))
    return qrels


def summarize_raw_dataset(data_root: str | Path, dataset: str) -> dict:
    manifest = list(read_jsonl(manifest_path(data_root, dataset)))
    queries = list(read_jsonl(queries_path(data_root, dataset)))
    qrels = list(read_jsonl(qrels_path(data_root, dataset)))
    return {
        "dataset": dataset,
        "manifest_rows": len(manifest),
        "query_rows": len(queries),
        "qrel_rows": len(qrels),
        "raw_dir": str(raw_dataset_dir(data_root, dataset)),
    }


def iter_datasets(selected: Iterable[str] | None) -> tuple[str, ...]:
    if selected is None:
        return DATASETS
    datasets = tuple(selected)
    for dataset in datasets:
        validate_dataset_name(dataset)
    return datasets
