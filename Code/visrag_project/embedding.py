from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import DEFAULT_QUERY_INSTRUCTION, DEFAULT_RETRIEVER_MODEL

def patch_transformers_cache_compat() -> None:
    """Restore the cache API expected by the VisRAG-Ret remote model code."""
    try:
        from transformers import cache_utils
    except Exception:
        return

    patched = []

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        previous_seq_length = self.get_seq_length(layer_idx)
        max_length = self.get_max_cache_shape(layer_idx)
        if max_length is not None and max_length > 0 and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    for class_name in ("Cache", "DynamicCache", "StaticCache"):
        cache_class = getattr(cache_utils, class_name, None)
        if cache_class is None:
            continue
        if hasattr(cache_class, "get_seq_length") and not hasattr(cache_class, "get_usable_length"):
            setattr(cache_class, "get_usable_length", get_usable_length)
            patched.append(class_name)

    if patched:
        print(
            "Applied Transformers cache compatibility shim for VisRAG-Ret "
            f"({', '.join(patched)}).",
            flush=True,
        )


def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    summed = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    denom = attention_mask_.sum(dim=1, keepdim=True).float()
    return summed / denom


class VisRAGEncoder:
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_RETRIEVER_MODEL,
        *,
        device: str | None = None,
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = True,
    ):
        patch_transformers_cache_compat()
        from transformers import AutoModel, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype)
        print(f"Loading VisRAG encoder {model_name_or_path} on {self.device}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        self.model.to(self.device)
        print("VisRAG encoder loaded.", flush=True)

    @torch.no_grad()
    def encode(self, text_or_images: Sequence[str | Image.Image]) -> np.ndarray:
        if not text_or_images:
            return np.zeros((0, 0), dtype=np.float32)

        if isinstance(text_or_images[0], str):
            inputs = {
                "text": list(text_or_images),
                "image": [None] * len(text_or_images),
                "tokenizer": self.tokenizer,
            }
        else:
            inputs = {
                "text": [""] * len(text_or_images),
                "image": list(text_or_images),
                "tokenizer": self.tokenizer,
            }

        outputs = self.model(**inputs)
        reps = weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
        embeddings = F.normalize(reps, p=2, dim=1)
        return embeddings.detach().cpu().float().numpy()

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode([text if text.strip() else "empty document" for text in texts])

    def encode_queries(self, queries: Sequence[str], instruction: str = DEFAULT_QUERY_INSTRUCTION) -> np.ndarray:
        return self.encode_texts([instruction + query for query in queries])

    def encode_image_paths(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        images = []
        for path in image_paths:
            try:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            except Exception as exc:
                raise OSError(f"Failed to load image for embedding: {path}") from exc
        return self.encode(images)


def batched(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
