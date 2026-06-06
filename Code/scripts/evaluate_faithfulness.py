#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrag_project.cli import add_baseline_arg, add_data_root_arg, add_dataset_arg, resolve_baselines, resolve_datasets
from visrag_project.faithfulness import DEFAULT_API_BASE, DEFAULT_JUDGE_MODEL, evaluate_faithfulness_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multimodal context faithfulness with a judge model.")
    add_data_root_arg(parser)
    add_dataset_arg(parser)
    add_baseline_arg(parser)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=None, help="Defaults to OPENAI_API_KEY or EMPTY.")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--max-image-edge", type=int, default=1024, help="Resize attached images to this longest edge. Use 0 to disable.")
    parser.add_argument("--max-text-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many predictions before applying --limit.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-slices", type=int, default=1, help="Split the selected predictions into this many contiguous slices.")
    parser.add_argument("--slice-index", type=int, default=0, help="Evaluate this 0-based slice after applying --offset/--limit.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for dataset in resolve_datasets(args.datasets):
        for baseline in resolve_baselines(args.baselines):
            summaries.append(
                evaluate_faithfulness_and_save(
                    dataset=dataset,
                    baseline=baseline,
                    data_root=args.data_root,
                    judge_model=args.judge_model,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    max_images=args.max_images,
                    max_image_edge=args.max_image_edge,
                    max_text_chars=args.max_text_chars,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    offset=args.offset,
                    limit=args.limit,
                    slice_index=args.slice_index,
                    num_slices=args.num_slices,
                    overwrite=args.overwrite,
                    show_progress=not args.no_progress,
                )
            )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
