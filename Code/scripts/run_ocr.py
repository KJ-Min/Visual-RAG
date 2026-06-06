#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrag_project.cli import add_data_root_arg, add_dataset_arg, resolve_datasets
from visrag_project.ocr.pipeline import run_ocr_for_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR over materialized VisRAG corpus images.")
    add_data_root_arg(parser)
    add_dataset_arg(parser)
    parser.add_argument("--engine", default="pytesseract", choices=["pytesseract", "layout_preserving", "adjacent_merging"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lang", default=None)
    parser.add_argument("--tesseract-config", default=None)
    parser.add_argument("--det-model-dir", default=None)
    parser.add_argument("--rec-model-dir", default=None)
    parser.add_argument("--cls-model-dir", default=None)
    parser.add_argument("--rec-label-file", default=None)
    parser.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--min-score", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine_kwargs = {
        "lang": args.lang,
        "tesseract_config": args.tesseract_config,
        "det_model_dir": args.det_model_dir,
        "rec_model_dir": args.rec_model_dir,
        "cls_model_dir": args.cls_model_dir,
        "rec_label_file": args.rec_label_file,
        "backend": args.backend,
        "device_id": args.device_id,
        "min_score": args.min_score,
    }
    summaries = []
    for dataset in resolve_datasets(args.datasets):
        summaries.append(
            run_ocr_for_dataset(
                dataset,
                args.data_root,
                engine=args.engine,
                overwrite=args.overwrite,
                limit=args.limit,
                engine_kwargs=engine_kwargs,
            )
        )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
