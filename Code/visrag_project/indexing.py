from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .bm25 import build_bm25_index, load_bm25_index, save_bm25_index
from .data import load_manifest
from .embedding import VisRAGEncoder, batched
from .io_utils import ensure_dir, read_json, write_json
from .ocr.pipeline import load_ocr_texts

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def index_dir(data_root: str | Path, index_type: str, dataset: str) -> Path:
    if index_type not in {"image", "text"}:
        raise ValueError("index_type must be image or text")
    return Path(data_root) / "indexes" / index_type / dataset




def validate_image_files(
    dataset: str,
    data_root: str | Path,
    *,
    limit: int | None = None,
) -> None:
    manifest = load_manifest(data_root, dataset)
    if limit is not None:
        manifest = manifest[:limit]

    bad_images = []
    iterator = tqdm(manifest, desc=f"{dataset} image validation", total=len(manifest)) if tqdm else manifest
    for idx, row in enumerate(iterator):
        image_path = row["image_path"]
        try:
            with Image.open(image_path) as image:
                image.load()
        except Exception as exc:
            bad_images.append(
                {
                    "index": idx,
                    "doc_id": row.get("doc_id"),
                    "image_path": image_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if not bad_images:
        return

    examples = "\n".join(
        f"  - index={row['index']} doc_id={row['doc_id']} path={row['image_path']} error={row['error']}"
        for row in bad_images[:10]
    )
    raise RuntimeError(
        f"Found {len(bad_images)} unreadable image(s) in {dataset} before indexing.\n"
        f"Repair the raw data with prepare_data.py --datasets {dataset} --overwrite, then rerun indexing.\n"
        f"Examples:\n{examples}"
    )

def build_image_index(
    dataset: str,
    data_root: str | Path,
    encoder: VisRAGEncoder,
    *,
    batch_size: int = 8,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict:
    manifest = load_manifest(data_root, dataset)
    if limit is not None:
        manifest = manifest[:limit]
    return _build_index(
        dataset=dataset,
        data_root=data_root,
        index_type="image",
        ids=[row["doc_id"] for row in manifest],
        payloads=[row["image_path"] for row in manifest],
        encode_batch=encoder.encode_image_paths,
        batch_size=batch_size,
        overwrite=overwrite,
        source={"kind": "raw_images"},
    )


def build_text_index(
    dataset: str,
    data_root: str | Path,
    *,
    ocr_engine: str = "pytesseract",
    limit: int | None = None,
    overwrite: bool = False,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict:
    manifest = load_manifest(data_root, dataset)
    texts_by_id = load_ocr_texts(data_root, ocr_engine, dataset)
    ids = []
    texts = []
    for row in manifest:
        doc_id = str(row["doc_id"])
        if doc_id not in texts_by_id:
            continue
        ids.append(doc_id)
        texts.append(texts_by_id[doc_id].get("text") or "empty document")
        if limit is not None and len(ids) >= limit:
            break

    out_dir = index_dir(data_root, "text", dataset)
    bm25_path = out_dir / "bm25_index.json"
    meta_path = out_dir / "meta.json"
    if bm25_path.exists() and meta_path.exists() and not overwrite:
        return load_index_meta(data_root, "text", dataset)

    ensure_dir(out_dir)
    index = build_bm25_index(doc_ids=ids, texts=texts, k1=k1, b=b)
    save_bm25_index(bm25_path, index)
    write_json(out_dir / "ids.json", ids)
    meta = {
        "dataset": dataset,
        "index_type": "text",
        "retriever": "bm25",
        "num_items": len(ids),
        "source": {"kind": "ocr_text", "ocr_engine": ocr_engine},
        "bm25_path": str(bm25_path),
        "ids_path": str(out_dir / "ids.json"),
        "k1": k1,
        "b": b,
    }
    write_json(meta_path, meta)
    return meta


def _build_index(
    *,
    dataset: str,
    data_root: str | Path,
    index_type: str,
    ids: list[str],
    payloads: list,
    encode_batch,
    batch_size: int,
    overwrite: bool,
    source: dict,
) -> dict:
    out_dir = index_dir(data_root, index_type, dataset)
    embeddings_path = out_dir / "embeddings.npy"
    ids_path = out_dir / "ids.json"
    meta_path = out_dir / "meta.json"
    if embeddings_path.exists() and ids_path.exists() and meta_path.exists() and not overwrite:
        return load_index_meta(data_root, index_type, dataset)

    ensure_dir(out_dir)
    vectors = []
    batches = list(batched(payloads, batch_size))
    iterator = tqdm(batches, desc=f"{dataset} {index_type} indexing", total=len(batches)) if tqdm else batches
    for batch in iterator:
        vectors.append(encode_batch(batch))
    embeddings = np.concatenate(vectors, axis=0).astype("float32") if vectors else np.zeros((0, 0), dtype="float32")
    np.save(embeddings_path, embeddings)
    write_json(ids_path, ids)
    meta = {
        "dataset": dataset,
        "index_type": index_type,
        "num_items": len(ids),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
        "source": source,
        "embeddings_path": str(embeddings_path),
        "ids_path": str(ids_path),
    }
    write_json(meta_path, meta)
    return meta


def load_index(data_root: str | Path, index_type: str, dataset: str) -> tuple[np.ndarray, list[str], dict]:
    if index_type == "text":
        raise ValueError("Text indexes are BM25 indexes. Use load_text_index instead.")
    out_dir = index_dir(data_root, index_type, dataset)
    embeddings_path = out_dir / "embeddings.npy"
    ids_path = out_dir / "ids.json"
    meta_path = out_dir / "meta.json"
    if not embeddings_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"Missing {index_type} index for {dataset}: {out_dir}")
    return np.load(embeddings_path), list(read_json(ids_path)), read_json(meta_path)


def load_index_meta(data_root: str | Path, index_type: str, dataset: str) -> dict:
    return read_json(index_dir(data_root, index_type, dataset) / "meta.json")


def load_text_index(data_root: str | Path, dataset: str) -> tuple[dict, dict]:
    out_dir = index_dir(data_root, "text", dataset)
    bm25_path = out_dir / "bm25_index.json"
    meta_path = out_dir / "meta.json"
    if not bm25_path.exists():
        raise FileNotFoundError(f"Missing BM25 text index for {dataset}: {bm25_path}")
    return load_bm25_index(bm25_path), read_json(meta_path)
