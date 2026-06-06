from __future__ import annotations

from pathlib import Path

from PIL import Image

from .data import load_manifest
from .io_utils import index_by_id, read_jsonl
from .ocr.pipeline import load_ocr_texts


def load_evidence_sources(data_root: str | Path, dataset: str, ocr_engine: str) -> tuple[dict, dict]:
    manifest_by_id = index_by_id(load_manifest(data_root, dataset), "doc_id")
    text_by_id = load_ocr_texts(data_root, ocr_engine, dataset)
    return manifest_by_id, text_by_id


def build_evidence_bundle(
    retrieval_row: dict,
    *,
    data_root: str | Path,
    ocr_engine: str = "pytesseract",
    max_items: int | None = None,
) -> dict:
    manifest_by_id, text_by_id = load_evidence_sources(data_root, retrieval_row["dataset"], ocr_engine)
    baseline = retrieval_row["baseline"]

    if baseline == "lvlm_only":
        images = []
        texts = []
        pairs = []
    elif baseline == "visrag_image":
        images = [_image_item(hit, manifest_by_id) for hit in retrieval_row["image_hits"]]
        texts = []
        pairs = []
    elif baseline == "naive_text":
        images = []
        texts = [_text_item(hit, text_by_id) for hit in retrieval_row["text_hits"]]
        pairs = []
    elif baseline == "image_and_its_text":
        pairs = [_pair_item(hit, manifest_by_id, text_by_id, source="image") for hit in retrieval_row["image_hits"]]
        images = [{"doc_id": item["doc_id"], "image_path": item["image_path"], "rank": item["rank"]} for item in pairs]
        texts = [{"doc_id": item["doc_id"], "text": item["text"], "rank": item["rank"]} for item in pairs]
    elif baseline == "text_and_its_image":
        pairs = [_pair_item(hit, manifest_by_id, text_by_id, source="text") for hit in retrieval_row["text_hits"]]
        images = [{"doc_id": item["doc_id"], "image_path": item["image_path"], "rank": item["rank"]} for item in pairs]
        texts = [{"doc_id": item["doc_id"], "text": item["text"], "rank": item["rank"]} for item in pairs]
    elif baseline == "text_and_image":
        images = [_image_item(hit, manifest_by_id) for hit in retrieval_row["image_hits"]]
        texts = [_text_item(hit, text_by_id) for hit in retrieval_row["text_hits"]]
        pairs = []
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    if max_items is not None:
        images = images[:max_items]
        texts = texts[:max_items]
        pairs = pairs[:max_items]

    return {
        "dataset": retrieval_row["dataset"],
        "baseline": baseline,
        "query_id": retrieval_row["query_id"],
        "query": retrieval_row["query"],
        "answer": retrieval_row.get("answer"),
        "query_record": retrieval_row.get("query_record", {}),
        "images": images,
        "texts": texts,
        "pairs": pairs,
    }


def load_retrieval_rows(path: str | Path) -> list[dict]:
    return list(read_jsonl(path))


def open_bundle_images(bundle: dict) -> list[Image.Image]:
    return [Image.open(item["image_path"]).convert("RGB") for item in bundle["images"]]


def evidence_text(bundle: dict) -> str:
    blocks = []
    for item in bundle.get("texts", []):
        text = item.get("text") or ""
        blocks.append(f"[{item['doc_id']}]\n{text}".strip())
    return "\n\n".join(blocks)


def _image_item(hit: dict, manifest_by_id: dict) -> dict:
    doc_id = hit["doc_id"]
    item = manifest_by_id[doc_id]
    return {
        "doc_id": doc_id,
        "image_path": item["image_path"],
        "rank": hit["rank"],
        "score": hit["score"],
    }


def _text_item(hit: dict, text_by_id: dict) -> dict:
    doc_id = hit["doc_id"]
    item = text_by_id.get(doc_id, {})
    return {
        "doc_id": doc_id,
        "text": item.get("text", ""),
        "rank": hit["rank"],
        "score": hit["score"],
        "ocr_status": item.get("status"),
    }


def _pair_item(hit: dict, manifest_by_id: dict, text_by_id: dict, *, source: str) -> dict:
    doc_id = hit["doc_id"]
    image = manifest_by_id[doc_id]
    text = text_by_id.get(doc_id, {})
    return {
        "doc_id": doc_id,
        "image_path": image["image_path"],
        "text": text.get("text", ""),
        "rank": hit["rank"],
        "score": hit["score"],
        "source": source,
        "ocr_status": text.get("status"),
    }
