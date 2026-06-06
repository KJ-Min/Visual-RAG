from __future__ import annotations

import argparse
from pathlib import Path

from .config import BASELINES, DATASETS, DEFAULT_DATA_ROOT


def add_data_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)


def add_dataset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Dataset names or 'all'.")


def add_baseline_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baselines", nargs="+", default=["all"], help="Baseline names or 'all'.")


def resolve_datasets(values: list[str]) -> tuple[str, ...]:
    if values == ["all"] or "all" in values:
        return DATASETS
    unknown = sorted(set(values) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")
    return tuple(values)


def resolve_baselines(values: list[str]) -> tuple[str, ...]:
    if values == ["all"] or "all" in values:
        return BASELINES
    unknown = sorted(set(values) - set(BASELINES))
    if unknown:
        raise ValueError(f"Unknown baselines: {', '.join(unknown)}")
    return tuple(values)
