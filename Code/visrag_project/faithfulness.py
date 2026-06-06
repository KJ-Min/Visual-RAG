from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from .config import NO_RETRIEVAL_BASELINES
from .io_utils import ensure_dir, read_json, read_jsonl, write_json

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_JUDGE_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"


def evaluate_faithfulness_and_save(
    *,
    dataset: str,
    baseline: str,
    data_root: str | Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    api_base: str | None = None,
    api_key: str | None = None,
    max_images: int = 5,
    max_image_edge: int = 1024,
    max_text_chars: int = 12000,
    max_tokens: int = 512,
    temperature: float = 0.0,
    offset: int = 0,
    limit: int | None = None,
    slice_index: int = 0,
    num_slices: int = 1,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict:
    data_root = Path(data_root)
    predictions_path = data_root / "runs" / baseline / dataset / "predictions.jsonl"
    output_path = data_root / "faithfulness" / baseline / f"{dataset}.jsonl"
    summary_path = data_root / "faithfulness" / baseline / f"{dataset}.summary.json"

    predictions = list(read_jsonl(predictions_path))
    if offset < 0:
        raise ValueError("--offset must be non-negative.")
    if offset:
        predictions = predictions[offset:]
    if limit is not None:
        predictions = predictions[:limit]
    predictions = _slice_predictions(predictions, slice_index=slice_index, num_slices=num_slices)

    ensure_dir(output_path.parent)
    existing_query_ids = set()
    summary_state = _new_summary_state()
    if overwrite:
        output_path.write_text("", encoding="utf-8")
    elif output_path.exists():
        for row in read_jsonl(output_path):
            if row.get("judge_model") != judge_model:
                continue
            existing_query_ids.add(row["query_id"])
            _add_summary_row(summary_state, row)

    iterator = predictions
    if show_progress and tqdm is not None:
        iterator = tqdm(predictions, desc=f"{dataset} {baseline} faithfulness", total=len(predictions))
    with output_path.open("a", encoding="utf-8") as output_file:
        for prediction in iterator:
            query_id = prediction["query_id"]
            if query_id in existing_query_ids:
                continue

            row = evaluate_prediction_faithfulness(
                prediction,
                judge_model=judge_model,
                api_base=api_base,
                api_key=api_key,
                max_images=max_images,
                max_image_edge=max_image_edge,
                max_text_chars=max_text_chars,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
            existing_query_ids.add(query_id)
            _add_summary_row(summary_state, row)

    summary = _summary_from_state(dataset=dataset, baseline=baseline, state=summary_state, judge_model=judge_model)
    write_json(summary_path, summary)
    _update_metric_file(data_root, dataset, baseline, summary)
    return summary | {"output_path": str(output_path), "summary_path": str(summary_path)}


def _slice_predictions(predictions: list[dict], *, slice_index: int, num_slices: int) -> list[dict]:
    if num_slices <= 0:
        raise ValueError("--num-slices must be positive.")
    if slice_index < 0 or slice_index >= num_slices:
        raise ValueError("--slice-index must satisfy 0 <= slice-index < num-slices.")
    if num_slices == 1:
        return predictions
    start = len(predictions) * slice_index // num_slices
    end = len(predictions) * (slice_index + 1) // num_slices
    return predictions[start:end]


def evaluate_prediction_faithfulness(
    prediction: dict,
    *,
    judge_model: str,
    api_base: str | None,
    api_key: str | None,
    max_images: int,
    max_image_edge: int,
    max_text_chars: int,
    max_tokens: int,
    temperature: float,
) -> dict:
    baseline = prediction["baseline"]
    if baseline in NO_RETRIEVAL_BASELINES:
        return _empty_context_row(prediction, judge_model)

    evidence = prediction.get("evidence") or {}
    has_images = bool(evidence.get("images"))
    has_texts = bool(evidence.get("texts"))
    if not has_images and not has_texts:
        return _empty_context_row(prediction, judge_model)

    content = _build_message_content(
        prediction,
        max_images=max_images,
        max_image_edge=max_image_edge,
        max_text_chars=max_text_chars,
    )
    raw_response = _chat_completion(
        content,
        judge_model=judge_model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    parsed = _parse_judge_json(raw_response)

    return {
        "dataset": prediction["dataset"],
        "baseline": baseline,
        "query_id": prediction["query_id"],
        "judge_model": judge_model,
        "faithfulness_score": _coerce_score(parsed.get("score")),
        "supported": parsed.get("supported"),
        "evidence_used": parsed.get("evidence_used"),
        "reason": parsed.get("reason", ""),
        "raw_response": raw_response,
        "parse_error": parsed.get("_parse_error"),
    }


def summarize_faithfulness(*, dataset: str, baseline: str, rows: list[dict], judge_model: str) -> dict:
    scores = [row["faithfulness_score"] for row in rows if row.get("faithfulness_score") is not None]
    supported = [row["supported"] for row in rows if isinstance(row.get("supported"), bool)]
    return {
        "dataset": dataset,
        "baseline": baseline,
        "judge_model": judge_model,
        "num_predictions": len(rows),
        "num_faithfulness_judged": len(scores),
        "judge_faithfulness_macro": _mean(scores),
        "judge_supported_rate": _mean([float(value) for value in supported]),
    }


def _new_summary_state() -> dict:
    return {
        "num_predictions": 0,
        "num_faithfulness_judged": 0,
        "score_sum": 0.0,
        "supported_count": 0,
        "supported_sum": 0.0,
    }


def _add_summary_row(state: dict, row: dict) -> None:
    state["num_predictions"] += 1
    score = row.get("faithfulness_score")
    if score is not None:
        state["num_faithfulness_judged"] += 1
        state["score_sum"] += float(score)
    supported = row.get("supported")
    if isinstance(supported, bool):
        state["supported_count"] += 1
        state["supported_sum"] += float(supported)


def _summary_from_state(*, dataset: str, baseline: str, state: dict, judge_model: str) -> dict:
    num_scores = state["num_faithfulness_judged"]
    num_supported = state["supported_count"]
    return {
        "dataset": dataset,
        "baseline": baseline,
        "judge_model": judge_model,
        "num_predictions": state["num_predictions"],
        "num_faithfulness_judged": num_scores,
        "judge_faithfulness_macro": state["score_sum"] / num_scores if num_scores else None,
        "judge_supported_rate": state["supported_sum"] / num_supported if num_supported else None,
    }


def _build_message_content(prediction: dict, *, max_images: int, max_image_edge: int, max_text_chars: int) -> list[dict]:
    content = [{"type": "text", "text": _build_judge_prompt(prediction, max_text_chars=max_text_chars)}]
    for item in (prediction.get("evidence") or {}).get("images", [])[:max_images]:
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(item["image_path"], max_edge=max_image_edge)}})
    return content


def _build_judge_prompt(prediction: dict, *, max_text_chars: int) -> str:
    evidence = prediction.get("evidence") or {}
    text_context = _format_text_context(evidence.get("texts", []), max_chars=max_text_chars)
    image_count = len(evidence.get("images", []))
    return f"""You are evaluating answer faithfulness for a multimodal RAG system.

Assume the retrieved context is the only source of truth.
Do not use outside knowledge.
Do not judge whether the retrieved context is the correct gold context.
Judge only whether the model answer is supported by the provided context.

If the answer is a multiple-choice letter, map it to the option text in the original prompt before judging support.
If the provided context is insufficient to support the answer, mark it unsupported.
Return only valid JSON with this schema:
{{
  "supported": true or false,
  "score": 1.0, 0.5, or 0.0,
  "evidence_used": "image", "text", "both", or "none",
  "reason": "brief explanation"
}}
Do not use LaTeX or backslash characters in the JSON string values.

Dataset: {prediction.get("dataset")}
Baseline: {prediction.get("baseline")}
Question and answer options shown to the answer model:
{_question_block(prediction)}

Model answer:
{prediction.get("prediction", "")}

Retrieved image evidence: {image_count} image(s) attached after this text.
Retrieved OCR text evidence:
{text_context if text_context else "[none]"}
"""


def _format_text_context(items: list[dict], *, max_chars: int) -> str:
    blocks = []
    total = 0
    for item in items:
        text = item.get("text") or ""
        block = f"[{item.get('doc_id')}]\n{text}".strip()
        if not block:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        blocks.append(block[:remaining])
        total += len(blocks[-1])
    return "\n\n".join(blocks)


def _question_block(prediction: dict) -> str:
    prompt = prediction.get("prompt") or ""
    marker = "\n\nQuestion:"
    if marker in prompt:
        return "Question:" + prompt.rsplit(marker, 1)[1]
    if "Question:" in prompt:
        return "Question:" + prompt.rsplit("Question:", 1)[1]
    return str(prediction.get("query") or "")


def _chat_completion(
    content: list[dict],
    *,
    judge_model: str,
    api_base: str | None,
    api_key: str | None,
    max_tokens: int,
    temperature: float,
) -> str:
    api_base = (api_base or os.getenv("OPENAI_BASE_URL") or DEFAULT_API_BASE).rstrip("/")
    api_key = api_key or os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY
    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Faithfulness judge request failed: HTTP {exc.code} {exc.reason}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Faithfulness judge request failed: {exc}") from exc
    return data["choices"][0]["message"]["content"]


def _parse_judge_json(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    decoder = json.JSONDecoder()
    for candidate in (cleaned, _escape_invalid_json_backslashes(cleaned)):
        for idx, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[idx:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return {"_parse_error": cleaned}


def _escape_invalid_json_backslashes(value: str) -> str:
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", value)


def _image_data_url(path: str | Path, *, max_edge: int) -> str:
    path = Path(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    image_bytes = _resized_image_bytes(path, max_edge=max_edge)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _resized_image_bytes(path: Path, *, max_edge: int) -> bytes:
    if max_edge <= 0:
        return path.read_bytes()

    from PIL import Image

    with Image.open(path) as image:
        image.load()
        if max(image.size) <= max_edge:
            return path.read_bytes()
        image = image.convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()


def _empty_context_row(prediction: dict, judge_model: str) -> dict:
    return {
        "dataset": prediction["dataset"],
        "baseline": prediction["baseline"],
        "query_id": prediction["query_id"],
        "judge_model": judge_model,
        "faithfulness_score": None,
        "supported": None,
        "evidence_used": "none",
        "reason": "No retrieved context is available for faithfulness evaluation.",
        "raw_response": "",
        "parse_error": None,
    }


def _coerce_score(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))


def _update_metric_file(data_root: Path, dataset: str, baseline: str, summary: dict) -> None:
    path = data_root / "metrics" / baseline / f"{dataset}.json"
    metrics = read_json(path) if path.exists() else {"dataset": dataset, "baseline": baseline}
    metrics.update(
        {
            "faithfulness_judge_model": summary["judge_model"],
            "num_faithfulness_judged": summary["num_faithfulness_judged"],
            "judge_faithfulness_macro": summary["judge_faithfulness_macro"],
            "judge_supported_rate": summary["judge_supported_rate"],
        }
    )
    ensure_dir(path.parent)
    write_json(path, metrics)


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None
