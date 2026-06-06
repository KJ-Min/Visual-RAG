from __future__ import annotations

from pathlib import Path

import torch

from .config import DEFAULT_GENERATOR_MODEL, NO_RETRIEVAL_BASELINES
from .data import load_queries
from .embedding import patch_transformers_cache_compat
from .evidence import build_evidence_bundle, evidence_text, load_retrieval_rows, open_bundle_images
from .io_utils import ensure_dir, write_jsonl

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def default_max_new_tokens(dataset: str) -> int:
    return 2 if dataset == "ArxivQA" else 20


class MiniCPMV26Generator:
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_GENERATOR_MODEL,
        *,
        device: str | None = None,
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
    ):
        patch_transformers_cache_compat()
        from transformers import AutoModel, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype)
        print(f"Loading MiniCPM-V generator {model_name_or_path} on {self.device}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        self.model.eval().to(self.device)
        print("MiniCPM-V generator loaded.", flush=True)

    def generate(self, bundle: dict, *, max_new_tokens: int | None = None) -> str:
        prompt = build_prompt(bundle)
        images = open_bundle_images(bundle)
        max_new_tokens = max_new_tokens or default_max_new_tokens(bundle["dataset"])

        if images:
            msgs = [{"role": "user", "content": images + [prompt]}]
        else:
            msgs = [{"role": "user", "content": prompt}]

        return self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=False,
            max_new_tokens=max_new_tokens,
        )


def build_prompt(bundle: dict) -> str:
    query_record = bundle.get("query_record") or {}
    query = bundle["query"]
    if bundle["dataset"] == "ArxivQA" and query_record.get("options"):
        options = _format_options(query_record["options"])
        answer_instruction = "Answer directly with the letter of the correct option as the first character."
        question_block = f"Question: {query}\nOptions:\n{options}\n{answer_instruction}"
    else:
        question_block = f"Answer the question using a single word or phrase.\nQuestion: {query}\nAnswer:"

    text = evidence_text(bundle)
    if text:
        return f"OCR text evidence:\n{text}\n\n{question_block}"
    return question_block


def run_generation(
    *,
    dataset: str,
    baseline: str,
    data_root: str | Path,
    retrieval_path: str | Path | None = None,
    ocr_engine: str = "pytesseract",
    model_name_or_path: str = DEFAULT_GENERATOR_MODEL,
    max_new_tokens: int | None = None,
    max_items: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    run_dir = Path(data_root) / "runs" / baseline / dataset
    retrieval_path = Path(retrieval_path) if retrieval_path else run_dir / "retrieval.jsonl"
    output_path = run_dir / "predictions.jsonl"
    if output_path.exists() and not overwrite:
        return {"dataset": dataset, "baseline": baseline, "output_path": str(output_path), "skipped": True}

    if retrieval_path.exists():
        rows = load_retrieval_rows(retrieval_path)
    elif baseline in NO_RETRIEVAL_BASELINES:
        rows = _query_only_rows(dataset, baseline, data_root)
    else:
        rows = load_retrieval_rows(retrieval_path)
    generator = None if dry_run else MiniCPMV26Generator(model_name_or_path)
    outputs = []
    iterator = tqdm(rows, desc=f"{dataset} {baseline} generation", total=len(rows)) if tqdm else rows
    for row in iterator:
        bundle = build_evidence_bundle(row, data_root=data_root, ocr_engine=ocr_engine, max_items=max_items)
        prompt = build_prompt(bundle)
        prediction = "" if dry_run else generator.generate(bundle, max_new_tokens=max_new_tokens)
        outputs.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "query_id": row["query_id"],
                "query": row["query"],
                "answer": row.get("answer"),
                "prediction": prediction,
                "prompt": prompt,
                "evidence": {
                    "images": bundle["images"],
                    "texts": bundle["texts"],
                    "pairs": bundle["pairs"],
                },
                "dry_run": dry_run,
            }
        )

    ensure_dir(output_path.parent)
    write_jsonl(output_path, outputs)
    return {"dataset": dataset, "baseline": baseline, "rows": len(outputs), "output_path": str(output_path)}


def _query_only_rows(dataset: str, baseline: str, data_root: str | Path) -> list[dict]:
    rows = []
    for query_row in load_queries(data_root, dataset):
        rows.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "query_id": query_row["query_id"],
                "query": query_row["query"],
                "answer": query_row.get("answer"),
                "query_record": query_row,
                "image_hits": [],
                "text_hits": [],
                "hits": [],
            }
        )
    return rows


def _format_options(options) -> str:
    lines = []
    for idx, option in enumerate(options):
        option = str(option).strip()
        letter = chr(65 + idx)
        if not option.startswith(letter):
            option = f"{letter}. {option}"
        lines.append(option)
    return "\n".join(lines)
