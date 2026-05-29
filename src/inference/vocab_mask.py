"""Vocabulary masking for generation.

A vocab mask restricts which token IDs the model is allowed to sample.
Masked-out tokens receive ``-inf`` logits before sampling, so they have
exactly zero probability of being chosen — a HARD constraint, not a
soft bias.

File formats (auto-detected by extension):

    *.json         sparse spec; recommended human-authored format
    *.npy          dense numpy bool array of shape (vocab_size,)
    *.safetensors  dense bool tensor named "mask" of shape (vocab_size,)

Sparse JSON schema (all keys required)::

    {
      "vocab_size": 151936,
      "mode": "allow" | "deny",
      "tokens": [0, 1, 2, ...]
    }

``mode == "allow"`` means the listed tokens are the only ones permitted.
``mode == "deny"`` means the listed tokens are forbidden, everything else
is permitted. The ``vocab_size`` is sanity-checked against the runtime
tokenizer at materialize-time.

The resulting boolean tensor is plugged into each backend's sampling path:

    * vLLM / ray_vllm   per-request ``SamplingParams.allowed_token_ids``
                        (V1 engine; the legacy ``logits_processors`` field
                        was removed in vLLM V1)
    * transformers      ``model.generate(logits_processor=...)``

Both processors are top-level callables so they pickle cleanly across
vLLM worker processes and Ray actors. The mask tensor is shared by
reference.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional


_VALID_MODES = ("allow", "deny")


@dataclass
class VocabMaskSpec:
    """Logical vocab mask, decoded from a file or built programmatically.

    Exactly one of ``tokens`` (sparse ID list) or ``dense`` (full bool
    tensor) must be set. ``materialize(vocab_size)`` collapses either
    form into a single dense bool tensor of shape ``(vocab_size,)``
    where ``True == allowed``.
    """

    tokens: Optional[List[int]] = None
    mode: str = "allow"
    dense: Optional[Any] = None
    vocab_size: Optional[int] = None

    def __post_init__(self) -> None:
        if (self.tokens is None) == (self.dense is None):
            raise ValueError(
                "VocabMaskSpec requires exactly one of `tokens` or `dense` "
                "to be set"
            )
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"VocabMaskSpec mode must be one of {_VALID_MODES}, "
                f"got {self.mode!r}"
            )

    @classmethod
    def from_file(cls, path: str) -> "VocabMaskSpec":
        """Load a mask from a file. Extension picks the format."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"vocab mask file not found: {path!r}")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            return cls._from_json(path)
        if ext == ".npy":
            return cls._from_npy(path)
        if ext == ".safetensors":
            return cls._from_safetensors(path)
        raise ValueError(
            f"unknown vocab mask file extension {ext!r} for {path!r}. "
            "Supported: .json (sparse spec), .npy (dense bool array), "
            ".safetensors (dense bool tensor named 'mask')."
        )

    @classmethod
    def _from_json(cls, path: str) -> "VocabMaskSpec":
        with open(path) as f:
            data = json.load(f)
        required = {"vocab_size", "mode", "tokens"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"vocab mask JSON {path!r} is missing required keys "
                f"{sorted(missing)}. Required schema: "
                "{'vocab_size': int, 'mode': 'allow'|'deny', "
                "'tokens': [int, ...]}"
            )
        vocab_size = int(data["vocab_size"])
        mode = data["mode"]
        tokens_raw = data["tokens"]
        if not isinstance(tokens_raw, list):
            raise ValueError(
                f"vocab mask JSON {path!r}: 'tokens' must be a list, "
                f"got {type(tokens_raw).__name__}"
            )
        tokens: List[int] = []
        for t in tokens_raw:
            if not isinstance(t, int) or isinstance(t, bool):
                raise ValueError(
                    f"vocab mask JSON {path!r}: 'tokens' must contain "
                    f"integers, found {t!r}"
                )
            if not 0 <= t < vocab_size:
                raise ValueError(
                    f"vocab mask JSON {path!r}: token id {t} is outside "
                    f"[0, vocab_size={vocab_size})"
                )
            tokens.append(t)
        return cls(tokens=tokens, mode=mode, vocab_size=vocab_size)

    @classmethod
    def _from_npy(cls, path: str) -> "VocabMaskSpec":
        import numpy as np
        import torch
        arr = np.load(path)
        if arr.ndim != 1:
            raise ValueError(
                f"npy vocab mask {path!r} must be 1-D, got shape {arr.shape}"
            )
        if arr.dtype != np.bool_:
            arr = arr.astype(np.bool_)
        dense = torch.from_numpy(arr).clone()
        return cls(dense=dense, mode="allow", vocab_size=int(arr.shape[0]))

    @classmethod
    def _from_safetensors(cls, path: str) -> "VocabMaskSpec":
        import safetensors.torch
        import torch
        tensors = safetensors.torch.load_file(path)
        if "mask" not in tensors:
            raise ValueError(
                f"safetensors vocab mask {path!r} is missing the 'mask' "
                f"tensor (found keys: {sorted(tensors.keys())})"
            )
        dense = tensors["mask"]
        if dense.ndim != 1:
            raise ValueError(
                f"safetensors vocab mask {path!r} must be 1-D, "
                f"got shape {tuple(dense.shape)}"
            )
        dense = dense.to(torch.bool)
        return cls(dense=dense, mode="allow", vocab_size=int(dense.shape[0]))

    def materialize(self, vocab_size: int) -> Any:
        """Return a dense bool tensor of shape ``(vocab_size,)``.

        ``True`` entries are allowed; ``False`` entries are masked out.
        Sanity-checks the spec's declared ``vocab_size`` against the
        argument and raises if every token would be masked.
        """
        import torch
        if self.vocab_size is not None and self.vocab_size != vocab_size:
            raise ValueError(
                f"vocab mask size mismatch: spec declares vocab_size="
                f"{self.vocab_size}, tokenizer reports {vocab_size}. "
                "The mask was built for a different tokenizer."
            )
        if self.dense is not None:
            if self.dense.shape[0] != vocab_size:
                raise ValueError(
                    f"dense vocab mask length {self.dense.shape[0]} != "
                    f"tokenizer vocab_size {vocab_size}"
                )
            mask = self.dense.to(torch.bool).clone()
        else:
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            mask[torch.tensor(self.tokens, dtype=torch.long)] = True
        if self.mode == "deny":
            mask = ~mask
        n_allowed = int(mask.sum().item())
        if n_allowed == 0:
            raise ValueError(
                "vocab mask allows zero tokens — generation would be "
                "impossible. Check that `mode` and `tokens` line up "
                "(allow=[] is the all-deny mask; deny=[] is the all-allow mask)."
            )
        return mask

    def num_allowed(self, vocab_size: int) -> int:
        """Number of tokens the materialized mask would allow."""
        return int(self.materialize(vocab_size).sum().item())


def warn_if_eos_masked(mask: Any, eos_token_id: Optional[int]) -> None:
    """Emit a ``UserWarning`` if the mask excludes the EOS token.

    Without an allowed EOS, generation runs to ``max_tokens`` every
    time — a common foot-gun. The warning is informational; some
    callers intentionally drop EOS for fixed-length sampling.
    """
    if eos_token_id is None:
        return
    if 0 <= eos_token_id < mask.shape[0] and not bool(mask[eos_token_id].item()):
        warnings.warn(
            f"vocab mask excludes the EOS token (id={eos_token_id}); "
            "generation will run to max_tokens every time. If this is "
            "intentional (fixed-length sampling), ignore this warning.",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Backend-specific logits processors
# ---------------------------------------------------------------------------
#
# Both processors are top-level classes (not closures) so cloudpickle can
# ship them to vLLM worker processes / Ray actors. The mask tensor lives
# as an attribute; on first call we lazily move it to the same device as
# the incoming logits, then reuse.


class _VLLMVocabMaskLogitsProcessor:
    """vLLM-shaped logits processor.

    vLLM's contract for an entry in ``SamplingParams.logits_processors``::

        processor(prompt_token_ids, output_token_ids, logits) -> logits

    where ``logits`` is a 1-D tensor of shape ``(vocab_size,)``.
    Returns a new tensor — vLLM threads the return value back into its
    sampler. Top-level class so pickling for worker dispatch is clean.
    """

    def __init__(self, mask: Any) -> None:
        self.mask = mask

    def __call__(self, prompt_token_ids, output_token_ids, logits):
        if self.mask.device != logits.device:
            self.mask = self.mask.to(logits.device)
        return logits.masked_fill(~self.mask, float("-inf"))


def make_vllm_logits_processor(mask: Any) -> _VLLMVocabMaskLogitsProcessor:
    """Build a picklable vLLM-shaped logits processor for ``mask``.

    Retained for the legacy (V0) per-request ``logits_processors`` path
    and any external callers. vLLM's V1 engine dropped per-request
    ``SamplingParams.logits_processors``; the V1-compatible path uses
    :func:`allowed_token_ids_from_mask` + ``SamplingParams.allowed_token_ids``
    instead.
    """
    return _VLLMVocabMaskLogitsProcessor(mask)


def allowed_token_ids_from_mask(mask: Any) -> List[int]:
    """Return the sorted token ids a dense bool ``mask`` permits.

    The dense mask is ``True`` for allowed tokens (the convention
    :meth:`VocabMaskSpec.materialize` produces). vLLM's V1 engine accepts
    this list directly via ``SamplingParams.allowed_token_ids``: the
    sampler retains scores only for these ids and ``-inf``-masks the
    rest — the same hard constraint the logits processor applied, but
    expressed through V1's supported per-request field.
    """
    import torch
    return torch.nonzero(mask, as_tuple=True)[0].tolist()


def make_hf_logits_processor(mask: Any):
    """Build a ``transformers.LogitsProcessor`` subclass instance for ``mask``.

    Subclassing happens lazily inside this factory so the rest of
    ``inference.vocab_mask`` is importable without ``transformers``
    installed (the vLLM-only paths don't need it).
    """
    from transformers import LogitsProcessor

    class _HFVocabMaskLogitsProcessor(LogitsProcessor):
        """transformers logits processor: scores[:, ~mask] = -inf."""

        def __init__(self, mask):
            self.mask = mask

        def __call__(self, input_ids, scores):
            if self.mask.device != scores.device:
                self.mask = self.mask.to(scores.device)
            return scores.masked_fill(~self.mask.unsqueeze(0), float("-inf"))

    return _HFVocabMaskLogitsProcessor(mask)
