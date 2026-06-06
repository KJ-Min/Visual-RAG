#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrag_project.cli import add_baseline_arg, add_data_root_arg, add_dataset_arg, resolve_baselines, resolve_datasets
from visrag_project.metrics import evaluate_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and generation outputs.")
    add_data_root_arg(parser)
    add_dataset_arg(parser)
    add_baseline_arg(parser)
    parser.add_argument("--topk", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for dataset in resolve_datasets(args.datasets):
        for baseline in resolve_baselines(args.baselines):
            summaries.append(evaluate_and_save(dataset=dataset, baseline=baseline, data_root=args.data_root, topk=args.topk))
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
