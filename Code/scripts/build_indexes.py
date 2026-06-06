#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrag_project.cli import add_data_root_arg, add_dataset_arg, resolve_datasets
from visrag_project.config import DEFAULT_RETRIEVER_MODEL
from visrag_project.embedding import VisRAGEncoder
from visrag_project.indexing import build_image_index, build_text_index, validate_image_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VisRAG image embeddings and BM25 OCR-text indexes.")
    add_data_root_arg(parser)
    add_dataset_arg(parser)
    parser.add_argument("--index-type", default="both", choices=["image", "text", "both"])
    parser.add_argument("--ocr-engine", default="pytesseract")
    parser.add_argument("--model-name-or-path", default=DEFAULT_RETRIEVER_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = resolve_datasets(args.datasets)
    summaries = []

    if args.index_type in {"text", "both"}:
        for dataset in datasets:
            summaries.append(
                build_text_index(
                    dataset,
                    args.data_root,
                    ocr_engine=args.ocr_engine,
                    limit=args.limit,
                    overwrite=args.overwrite,
                    k1=args.bm25_k1,
                    b=args.bm25_b,
                )
            )

    if args.index_type in {"image", "both"}:
        for dataset in datasets:
            validate_image_files(dataset, args.data_root, limit=args.limit)

        encoder = VisRAGEncoder(args.model_name_or_path, device=args.device, dtype=args.dtype)
        for dataset in datasets:
            summaries.append(
                build_image_index(
                    dataset,
                    args.data_root,
                    encoder,
                    batch_size=args.batch_size,
                    limit=args.limit,
                    overwrite=args.overwrite,
                )
            )

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
