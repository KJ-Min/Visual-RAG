#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrag_project.cli import add_baseline_arg, add_data_root_arg, add_dataset_arg, resolve_baselines, resolve_datasets
from visrag_project.config import DEFAULT_RETRIEVER_MODEL, DUAL_INDEX_BASELINES, IMAGE_INDEX_BASELINES
from visrag_project.embedding import VisRAGEncoder
from visrag_project.retrieval import run_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VisRAG image retrieval and BM25 OCR-text retrieval.")
    add_data_root_arg(parser)
    add_dataset_arg(parser)
    add_baseline_arg(parser)
    parser.add_argument("--model-name-or-path", default=DEFAULT_RETRIEVER_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baselines = resolve_baselines(args.baselines)
    needs_image_encoder = any(
        baseline in IMAGE_INDEX_BASELINES or baseline in DUAL_INDEX_BASELINES
        for baseline in baselines
    )
    encoder = VisRAGEncoder(args.model_name_or_path, device=args.device, dtype=args.dtype) if needs_image_encoder else None

    summaries = []
    for dataset in resolve_datasets(args.datasets):
        for baseline in baselines:
            summaries.append(
                run_retrieval(
                    dataset,
                    baseline,
                    args.data_root,
                    encoder,
                    topk=args.topk,
                    batch_size=args.batch_size,
                    overwrite=args.overwrite,
                )
            )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
