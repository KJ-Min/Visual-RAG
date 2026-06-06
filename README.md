# Visual RAG Faithfulness Sample

This repository includes the code and a small `visrag_image` sample for running the multimodal faithfulness judge on one ArxivQA query.

The committed sample contains:

- `Data/runs/visrag_image/ArxivQA/predictions.jsonl`
- the top-5 evidence images for the first `visrag_image` prediction

It is intended for a quick sanity check of `Code/scripts/evaluate_faithfulness.py`, not for reproducing the full experiment.

## 1. Clone

```bash
git clone https://github.com/KJ-Min/Visual-RAG.git
cd Visual-RAG
```

## 2. Start a VLM Judge Server

In one terminal, start an OpenAI-compatible vLLM server:

```bash
vllm serve Qwen/Qwen3.6-27B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --reasoning-parser qwen3
```

You can verify the served model name with:

```bash
curl http://localhost:8000/v1/models
```

The `--judge-model` argument below must match the model id returned by this endpoint.

## 3. Run Faithfulness on the Sample

In another terminal:

```bash
python Code/scripts/evaluate_faithfulness.py \
  --datasets ArxivQA \
  --baselines visrag_image \
  --judge-model Qwen/Qwen3.6-27B \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --limit 1 \
  --max-images 5 \
  --max-image-edge 0 \
  --max-text-chars 12000
```

The result is appended to:

```text
Data/faithfulness/visrag_image/ArxivQA.jsonl
```

and the summary is written to:

```text
Data/faithfulness/visrag_image/ArxivQA.summary.json
```

## Notes

- Use `--limit 1` with the committed sample. Only the first query's top-5 evidence images are included in Git.
- `--max-image-edge 0` keeps the original image resolution.
- If the run is interrupted, rerun the same command without `--overwrite`; already evaluated query ids are skipped.
- To evaluate the full dataset, regenerate or provide the full `Data/` directory.
