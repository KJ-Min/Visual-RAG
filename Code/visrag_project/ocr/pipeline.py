from __future__ import annotations

import json
import time
from pathlib import Path

from ..data import load_manifest
from ..io_utils import ensure_dir, read_jsonl
from .engines import get_ocr_engine

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def ocr_output_path(data_root: str | Path, engine: str, dataset: str) -> Path:
    return Path(data_root) / "ocr" / engine / f"{dataset}.jsonl"


def load_ocr_texts(data_root: str | Path, engine: str, dataset: str) -> dict[str, dict]:
    path = ocr_output_path(data_root, engine, dataset)
    if not path.exists():
        raise FileNotFoundError(f"Missing OCR output: {path}. Run run_ocr.py first.")
    return {str(row["doc_id"]): row for row in read_jsonl(path)}


def run_ocr_for_dataset(
    dataset: str,
    data_root: str | Path,
    *,
    engine: str = "pytesseract",
    overwrite: bool = False,
    limit: int | None = None,
    engine_kwargs: dict | None = None,
) -> dict:
    engine_kwargs = engine_kwargs or {}
    extractor = get_ocr_engine(engine, **engine_kwargs)
    output = ocr_output_path(data_root, engine, dataset)
    ensure_dir(output.parent)

    existing = {}
    if output.exists() and not overwrite:
        existing = {str(row["doc_id"]): row for row in read_jsonl(output)}
    elif output.exists() and overwrite:
        output.unlink()

    manifest = load_manifest(data_root, dataset)
    if limit is not None:
        manifest = manifest[:limit]

    ok_count = sum(1 for row in existing.values() if row.get("status") == "ok")
    error_count = len(existing) - ok_count
    iterator = tqdm(manifest, desc=f"{dataset} OCR/{engine}", total=len(manifest)) if tqdm else manifest

    with output.open("a", encoding="utf-8") as f:
        for item in iterator:
            doc_id = str(item["doc_id"])
            if doc_id in existing:
                continue

            started = time.time()
            try:
                result = extractor.extract(item["image_path"])
                row = {
                    "dataset": dataset,
                    "doc_id": doc_id,
                    "image_path": item["image_path"],
                    "text": result.text,
                    "status": "ok",
                    "elapsed_sec": round(time.time() - started, 4),
                    "metadata": result.metadata,
                }
                ok_count += 1
            except Exception as exc:
                row = {
                    "dataset": dataset,
                    "doc_id": doc_id,
                    "image_path": item["image_path"],
                    "text": "",
                    "status": "error",
                    "elapsed_sec": round(time.time() - started, 4),
                    "error": repr(exc),
                    "metadata": {"engine": engine},
                }
                error_count += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    return {
        "dataset": dataset,
        "engine": engine,
        "rows": ok_count + error_count,
        "ok_rows": ok_count,
        "error_rows": error_count,
        "output_path": str(output),
    }
