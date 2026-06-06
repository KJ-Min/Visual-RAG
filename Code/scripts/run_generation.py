#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrag_project.cli import add_baseline_arg, add_data_root_arg, add_dataset_arg, resolve_baselines, resolve_datasets
from visrag_project.config import DEFAULT_GENERATOR_MODEL
from visrag_project.generation import run_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate answers with MiniCPM-V-2.6 from retrieval outputs.")
    add_data_root_arg(parser)
    add_dataset_arg(parser)
    add_baseline_arg(parser)
    parser.add_argument("--ocr-engine", default="pytesseract")
    parser.add_argument("--model-name-or-path", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write prompts/evidence without loading the generator.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for dataset in resolve_datasets(args.datasets):
        for baseline in resolve_baselines(args.baselines):
            summaries.append(
                run_generation(
                    dataset=dataset,
                    baseline=baseline,
                    data_root=args.data_root,
                    ocr_engine=args.ocr_engine,
                    model_name_or_path=args.model_name_or_path,
                    max_new_tokens=args.max_new_tokens,
                    max_items=args.max_items,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                )
            )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
