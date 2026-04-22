from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Pooling strategy registry
# ---------------------------------------------------------------------------

POOLING_FNS: Dict[str, Callable] = {}


def register_pooling(name: str):
    """Decorator to register a pooling strategy by name."""
    def decorator(fn):
        POOLING_FNS[name] = fn
        return fn
    return decorator


@register_pooling("mean")
def _mean_pooling(hidden_states, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    return torch.sum(hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


@register_pooling("cls")
def _cls_pooling(hidden_states, attention_mask):
    return hidden_states[:, 0, :]


@register_pooling("last")
def _last_token_pooling(hidden_states, attention_mask):
    import torch
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, seq_lengths]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    """Configuration for the HuggingFace embedding model.

    ``layer`` selects which transformer layer to extract hidden states from.
    ``None`` (default) uses ``last_hidden_state``.  An integer indexes into
    ``outputs.hidden_states`` (0 = embedding layer, 1..N = transformer layers).
    """
    model_path: str
    pooling: str = "mean"           # any registered pooling name
    normalize: bool = True
    layer: Optional[int] = None     # None = last hidden state
    batch_size: int = 32
    max_length: Optional[int] = None
    instruction: Optional[str] = None
    concurrency: int = 1
    n_devices_per_instance: int = 1
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Instruction formatting
# ---------------------------------------------------------------------------

def _format_instruction(text: str, instruction: Optional[str]) -> str:
    if instruction is None:
        return text
    return f"Instruct: {instruction}\nQuery:{text}"


# ---------------------------------------------------------------------------
# Worker (runs in spawned process)
# ---------------------------------------------------------------------------

def _hf_embed_worker(
    gpu_ids: List[int],
    model_path: str,
    items: List[Dict[str, Any]],
    batch_size: int,
    model_kwargs: Dict[str, Any],
    pooling: str,
    normalize: bool,
    max_length: Optional[int],
    layer: Optional[int],
    return_dict,
    progress_counter,
):
    import torch
    from transformers import AutoTokenizer, AutoModel
    from tqdm import trange

    assert len(gpu_ids) == 1, "Only one GPU per embedding worker is supported"
    gpu_id = gpu_ids[0]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=model_kwargs.get("trust_remote_code", False),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(model_path, **model_kwargs)
    model.to(f"cuda:{gpu_id}")
    model.eval()

    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings", 8192)

    pool_fn = POOLING_FNS.get(pooling)
    if pool_fn is None:
        raise ValueError(f"Unknown pooling: {pooling}. Available: {list(POOLING_FNS.keys())}")

    for i in trange(0, len(items), batch_size, desc=f"GPU {gpu_id}"):
        batch = items[i:i + batch_size]
        ids = [x["id"] for x in batch]
        texts = [x["text"] for x in batch]

        encoded = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.inference_mode():
            need_hidden = layer is not None
            outputs = model(**encoded, output_hidden_states=need_hidden)

            if layer is not None:
                hs = outputs.hidden_states[layer]
            else:
                hs = outputs.last_hidden_state

            embeddings = pool_fn(hs, encoded["attention_mask"])

            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            embeddings = embeddings.cpu().numpy()

        for b, emb in enumerate(embeddings):
            return_dict[ids[b]] = emb

        progress_counter.value += len(batch)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """HuggingFace embedding model with configurable pooling and layer extraction."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @staticmethod
    def from_config(config: EmbeddingConfig) -> EmbeddingModel:
        return EmbeddingModel(config)

    def embed(
        self,
        texts: Union[List[str], List[Dict[str, Any]]],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """Compute embeddings for a list of texts.

        Each text can be a plain string or a dict with ``text`` and
        optional per-item ``instruction``.
        """
        if not texts:
            return np.array([])

        effective_inst = instruction or self.config.instruction
        items: List[Dict[str, Any]] = []
        for i, t in enumerate(texts):
            if isinstance(t, str):
                text, inst = t, effective_inst
            elif isinstance(t, dict):
                text = t["text"]
                inst = t.get("instruction", effective_inst)
            else:
                raise ValueError(f"Invalid input type: {type(t)}")
            items.append({"id": i, "text": _format_instruction(text, inst)})

        return self._run_workers(items)

    def _run_workers(self, items: List[Dict[str, Any]]) -> np.ndarray:
        import torch
        from multiprocessing import get_context, Manager
        from tqdm import tqdm

        cfg = self.config
        total_gpus = torch.cuda.device_count()
        needed = cfg.concurrency * cfg.n_devices_per_instance
        if total_gpus < needed:
            raise RuntimeError(f"Need {needed} GPUs, only {total_gpus} available")

        gpu_groups = [
            list(range(i * cfg.n_devices_per_instance, (i + 1) * cfg.n_devices_per_instance))
            for i in range(cfg.concurrency)
        ]

        splits: list = [[] for _ in range(cfg.concurrency)]
        for j, item in enumerate(items):
            splits[j % cfg.concurrency].append(item)

        ctx = get_context("spawn")
        manager = Manager()
        shared_out = manager.dict()
        progress = manager.Value("i", 0)
        processes = []

        for gpu_group, subset in zip(gpu_groups, splits):
            if not subset:
                continue
            p = ctx.Process(
                target=_hf_embed_worker,
                kwargs=dict(
                    gpu_ids=gpu_group,
                    model_path=cfg.model_path,
                    items=subset,
                    batch_size=cfg.batch_size,
                    model_kwargs=cfg.model_kwargs,
                    pooling=cfg.pooling,
                    normalize=cfg.normalize,
                    max_length=cfg.max_length,
                    layer=cfg.layer,
                    return_dict=shared_out,
                    progress_counter=progress,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)

        with tqdm(total=len(items), desc="Embedding") as pbar:
            last = 0
            while any(p.is_alive() for p in processes):
                cur = progress.value
                if cur > last:
                    pbar.update(cur - last)
                    last = cur
                time.sleep(0.1)
            cur = progress.value
            if cur > last:
                pbar.update(cur - last)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker exited with code {p.exitcode}")

        return np.array([shared_out[i] for i in range(len(items))])
