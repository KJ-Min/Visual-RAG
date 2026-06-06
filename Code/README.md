# VisRAG Assignment 2 Pipeline

This folder contains project-owned wrappers around the official `VisRAG/` clone.
The official repo is treated as read-only.

## Smoke Test Order

Run from `/home/kjmin/Project/AI-Algorithm_26-1/Assignment_2`.

```bash
python Code/scripts/prepare_data.py --datasets MP-DocVQA --max-docs 20 --max-queries 20
python Code/scripts/run_ocr.py --datasets MP-DocVQA --engine pytesseract
python Code/scripts/build_indexes.py --datasets MP-DocVQA --index-type both
python Code/scripts/run_retrieval.py --datasets MP-DocVQA --baselines all --topk 5
python Code/scripts/run_generation.py --datasets MP-DocVQA --baselines all --dry-run
python Code/scripts/evaluate.py --datasets MP-DocVQA --baselines all --topk 5
python Code/scripts/evaluate_faithfulness.py --datasets MP-DocVQA --baselines all --limit 1
```

Remove the smoke-test limits and `--dry-run` for the full experiment.

## Retrieval Setup

- `lvlm_only` skips retrieval and answers from the question/options only.
- Image baselines use `openbmb/VisRAG-Ret` image embeddings.
- Text baselines use BM25 over OCR text, matching the paper's OCR lexical retrieval setting more closely than dense text embeddings.
- OCR rows, BM25 ids, image manifest rows, retrieval hits, and evidence bundles all use the same `doc_id` for image-text mapping.

## Outputs

- `Data/raw/{dataset}/`: manifest, images, queries, qrels.
- `Data/ocr/{engine}/{dataset}.jsonl`: OCR text keyed by `doc_id`.
- `Data/indexes/image/{dataset}/`: VisRAG-Ret image embeddings and ids.
- `Data/indexes/text/{dataset}/`: BM25 index over OCR text keyed by `doc_id`.
- `Data/runs/{baseline}/{dataset}/`: retrieval and prediction JSONL/TREC files.
- `Data/metrics/{baseline}/{dataset}.json`: retrieval and answer metrics.
- `Data/faithfulness/{baseline}/{dataset}.jsonl`: judge-based context faithfulness results.
