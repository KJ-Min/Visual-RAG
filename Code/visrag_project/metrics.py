from __future__ import annotations

import re
from pathlib import Path

from .config import NO_RETRIEVAL_BASELINES
from .data import load_qrels
from .data import load_queries
from .io_utils import as_list, read_jsonl, write_json


def normalize_answer(value) -> str:
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-z.% ]+", "", text)
    return text.strip()


def token_f1(prediction, answers) -> dict:
    pred = normalize_answer(prediction)
    golds = [normalize_answer(answer) for answer in as_list(answers)]
    if not golds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact": 0.0}

    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact": 0.0}
    pred_tokens = pred.split()
    for gold in golds:
        gold_tokens = gold.split()
        exact = float(pred == gold and gold != "")
        if not pred_tokens or not gold_tokens:
            scores = {"precision": exact, "recall": exact, "f1": exact, "exact": exact}
        else:
            common = _overlap_count(pred_tokens, gold_tokens)
            precision = common / len(pred_tokens)
            recall = common / len(gold_tokens)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            scores = {"precision": precision, "recall": recall, "f1": f1, "exact": exact}
        if scores["f1"] > best["f1"] or scores["exact"] > best["exact"]:
            best = scores
    return best


def context_support(prediction, evidence_text: str) -> float:
    pred = normalize_answer(prediction)
    evidence = normalize_answer(evidence_text)
    if not pred:
        return 0.0
    if pred in evidence:
        return 1.0
    pred_tokens = [token for token in pred.split() if token]
    if not pred_tokens:
        return 0.0
    evidence_tokens = set(evidence.split())
    return sum(1 for token in pred_tokens if token in evidence_tokens) / len(pred_tokens)


def evaluate_retrieval_file(
    *,
    dataset: str,
    baseline: str,
    data_root: str | Path,
    retrieval_path: str | Path | None = None,
    topk: int | None = None,
) -> dict:
    if baseline in NO_RETRIEVAL_BASELINES:
        return {
            "dataset": dataset,
            "baseline": baseline,
            "num_queries": len(load_queries(data_root, dataset)),
            "retrieval_recall_macro": None,
        }

    retrieval_path = Path(retrieval_path) if retrieval_path else Path(data_root) / "runs" / baseline / dataset / "retrieval.jsonl"
    qrels = load_qrels(data_root, dataset)
    rows = list(read_jsonl(retrieval_path))
    recalls = []
    for row in rows:
        relevant = {doc_id for doc_id, score in qrels.get(row["query_id"], {}).items() if score > 0}
        ranked = row.get("hits", [])
        if topk is not None:
            ranked = ranked[:topk]
        retrieved = {hit["doc_id"] for hit in ranked}
        recalls.append(len(relevant & retrieved) / len(relevant) if relevant else 0.0)
    return {
        "dataset": dataset,
        "baseline": baseline,
        "num_queries": len(rows),
        "retrieval_recall_macro": _mean(recalls),
    }


def evaluate_predictions_file(
    *,
    dataset: str,
    baseline: str,
    data_root: str | Path,
    predictions_path: str | Path | None = None,
) -> dict:
    predictions_path = Path(predictions_path) if predictions_path else Path(data_root) / "runs" / baseline / dataset / "predictions.jsonl"
    rows = list(read_jsonl(predictions_path))
    precision = []
    recall = []
    f1 = []
    exact = []
    support = []
    for row in rows:
        scores = token_f1(row.get("prediction"), row.get("answer"))
        precision.append(scores["precision"])
        recall.append(scores["recall"])
        f1.append(scores["f1"])
        exact.append(scores["exact"])
        evidence_text = "\n".join(item.get("text", "") for item in row.get("evidence", {}).get("texts", []))
        support.append(context_support(row.get("prediction"), evidence_text))
    return {
        "dataset": dataset,
        "baseline": baseline,
        "num_predictions": len(rows),
        "answer_precision_macro": _mean(precision),
        "answer_recall_macro": _mean(recall),
        "answer_f1_macro": _mean(f1),
        "answer_exact_macro": _mean(exact),
        "context_faithfulness_macro": _mean(support),
    }


def evaluate_and_save(
    *,
    dataset: str,
    baseline: str,
    data_root: str | Path,
    topk: int | None = None,
) -> dict:
    metrics = evaluate_retrieval_file(dataset=dataset, baseline=baseline, data_root=data_root, topk=topk)
    pred_path = Path(data_root) / "runs" / baseline / dataset / "predictions.jsonl"
    if pred_path.exists():
        metrics.update(evaluate_predictions_file(dataset=dataset, baseline=baseline, data_root=data_root))
    output = Path(data_root) / "metrics" / baseline / f"{dataset}.json"
    write_json(output, metrics)
    return metrics


def _overlap_count(left: list[str], right: list[str]) -> int:
    remaining = {}
    for token in right:
        remaining[token] = remaining.get(token, 0) + 1
    count = 0
    for token in left:
        if remaining.get(token, 0) > 0:
            count += 1
            remaining[token] -= 1
    return count


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
