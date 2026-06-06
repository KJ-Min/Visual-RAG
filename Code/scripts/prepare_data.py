#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrag_project.cli import add_data_root_arg, add_dataset_arg, resolve_datasets
from visrag_project.data import materialize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize VisRAG HF datasets locally.")
    add_data_root_arg(parser)
    add_dataset_arg(parser)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for dataset in resolve_datasets(args.datasets):
        summaries.append(
            materialize_dataset(
                dataset,
                args.data_root,
                max_docs=args.max_docs,
                max_queries=args.max_queries,
                overwrite=args.overwrite,
            )
        )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
