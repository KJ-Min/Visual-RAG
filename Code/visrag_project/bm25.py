from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from .io_utils import read_json, write_json


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def build_bm25_index(
    *,
    doc_ids: list[str],
    texts: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict:
    tokenized = [tokenize(text) for text in texts]
    doc_freq: Counter[str] = Counter()
    term_freqs = []
    doc_lens = []

    for tokens in tokenized:
        counts = Counter(tokens)
        term_freqs.append(dict(counts))
        doc_lens.append(len(tokens))
        doc_freq.update(counts.keys())

    num_docs = len(doc_ids)
    avgdl = sum(doc_lens) / num_docs if num_docs else 0.0
    idf = {
        term: math.log(1 + (num_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freq.items()
    }

    return {
        "doc_ids": doc_ids,
        "term_freqs": term_freqs,
        "doc_lens": doc_lens,
        "avgdl": avgdl,
        "idf": idf,
        "k1": k1,
        "b": b,
        "tokenizer": "regex_alnum_lower",
    }


def save_bm25_index(path: str | Path, index: dict) -> None:
    write_json(path, index)


def load_bm25_index(path: str | Path) -> dict:
    return read_json(path)


def search_bm25(index: dict, query: str, topk: int) -> list[dict]:
    query_terms = tokenize(query)
    if not query_terms or not index["doc_ids"]:
        return []

    k1 = float(index["k1"])
    b = float(index["b"])
    avgdl = float(index["avgdl"]) or 1.0
    idf = index["idf"]
    scores = []

    for doc_idx, doc_id in enumerate(index["doc_ids"]):
        doc_len = int(index["doc_lens"][doc_idx])
        term_freq = index["term_freqs"][doc_idx]
        score = 0.0
        for term in query_terms:
            tf = int(term_freq.get(term, 0))
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * doc_len / avgdl)
            score += float(idf.get(term, 0.0)) * (tf * (k1 + 1)) / denom
        if score > 0:
            scores.append({"doc_id": doc_id, "score": score})

    scores.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(scores[:topk], start=1):
        item["rank"] = rank
    return scores[:topk]
