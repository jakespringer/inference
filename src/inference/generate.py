from __future__ import annotations

import json
import os
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .vocab_mask import (
    VocabMaskSpec,
    allowed_token_ids_from_mask,
    make_hf_logits_processor,
    make_vllm_logits_processor,
    warn_if_eos_masked,
)


# ---------------------------------------------------------------------------
# LoRA-adapter detection
# ---------------------------------------------------------------------------
#
# A PEFT-style adapter directory contains ``adapter_config.json`` +
# ``adapter_model.safetensors`` (and usually tokenizer files), but NO
# ``config.json`` for the base model. ``AutoModelForCausalLM.from_pretrained``
# pointed at such a directory will fail with "Can't find config.json".
# The helpers below let each backend opt into a load path that reads
# ``adapter_config.json`` to find the base model, loads that, and then
# applies the adapter on top (either via PEFT-merge for HF, or vLLM's
# runtime ``LoRARequest`` for the vLLM backend).

def _is_lora_adapter_dir(path: Optional[str]) -> bool:
    """True iff ``path`` is a local directory holding a PEFT adapter."""
    if not path or not os.path.isdir(path):
        return False
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def _read_adapter_config(adapter_dir: str) -> Tuple[str, int]:
    """Return ``(base_model_name_or_path, r)`` from ``adapter_config.json``.

    ``r`` defaults to 8 if missing, matching PEFT's own default; the
    value is used by the vLLM backend to size its LoRA caches
    (``max_lora_rank``) when the caller didn't specify one explicitly.
    """
    with open(os.path.join(adapter_dir, "adapter_config.json"), "r") as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            f"adapter_config.json at {adapter_dir} is missing "
            "``base_model_name_or_path`` — cannot resolve base model"
        )
    return base, int(cfg.get("r", 8))


# ---------------------------------------------------------------------------
# Partial-finetune detection
# ---------------------------------------------------------------------------
#
# A partial-finetune directory (produced by ``finetuning.finetune.checkpoint.
# save_partial_model``) carries ``partial_config.json`` +
# ``partial_model.safetensors`` and a tokenizer, but NO ``config.json``
# or ``model.safetensors`` for the base model. The safetensors file
# stores ONLY the parameters whose ``requires_grad=True`` at save time
# — every other weight must come from the base model named in
# ``partial_config.json``.
#
# Each backend has its own merge strategy:
#   * HF backend: load base → overlay partial state dict in memory.
#   * vLLM backend: vLLM has no runtime layer-overlay API, so we
#     materialize a fully-merged HF directory in a tempdir and point
#     vLLM at that.

def _is_partial_finetune_dir(path: Optional[str]) -> bool:
    """True iff ``path`` is a local directory holding a partial-finetune save."""
    if not path or not os.path.isdir(path):
        return False
    return os.path.exists(os.path.join(path, "partial_config.json"))


def _read_partial_config(partial_dir: str) -> Tuple[str, str]:
    """Return ``(base_model_name_or_path, partial_weights_path)``."""
    with open(os.path.join(partial_dir, "partial_config.json"), "r") as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            f"partial_config.json at {partial_dir} is missing "
            "``base_model_name_or_path`` — cannot resolve base model"
        )
    return base, os.path.join(partial_dir, "partial_model.safetensors")


def _materialize_partial_merge(partial_dir: str) -> str:
    """Load base + overlay partial state dict + save HF dir to a tempdir.

    Returns the tempdir path. Registers an ``atexit`` hook that deletes
    the directory so the ~30 GB merged checkpoint doesn't leak after
    the process exits. Honors ``$TMPDIR`` for placement — users can
    redirect to a large scratch dir (``/media/disk1/...``) when ``/tmp``
    is too small for a 14B-param merge.

    Used only by the vLLM backend; the HF backends overlay weights in
    memory without ever touching disk.
    """
    import atexit
    import shutil
    import tempfile

    import safetensors.torch
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model, partial_weights = _read_partial_config(partial_dir)
    tempdir = tempfile.mkdtemp(prefix="vllm_partial_merge_")
    atexit.register(lambda d=tempdir: shutil.rmtree(d, ignore_errors=True))

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, trust_remote_code=True,
    )
    partial_sd = safetensors.torch.load_file(partial_weights)
    _, unexpected = model.load_state_dict(partial_sd, strict=False)
    if unexpected:
        raise RuntimeError(
            f"partial weights at {partial_weights} contain keys not present "
            f"in base model {base_model!r}: {unexpected[:5]}..."
        )
    model.save_pretrained(tempdir, safe_serialization=True)

    # Tokenizer: prefer the partial dir (carries any trainer-side
    # ``chat_template`` overrides written by ``tokenizer.save_pretrained``);
    # fall back to the base model only when the partial dir didn't carry one.
    try:
        tok = AutoTokenizer.from_pretrained(partial_dir, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.save_pretrained(tempdir)
    return tempdir


# ---------------------------------------------------------------------------
# Port-picking helper (vLLM)
# ---------------------------------------------------------------------------

def _ensure_vllm_random_port(env_var: str = "VLLM_PORT") -> None:
    """Set ``VLLM_PORT`` to a random free port if not already set.

    Multiple vLLM engines on the same node would otherwise share the
    same default port and collide on ``bind``. We bind a kernel-chosen
    ephemeral port (``port=0``), read back the assigned number, and
    close the socket — vLLM will then bind that port itself moments
    later. The race window between close and re-bind is small enough
    in practice to be reliable, and far cheaper than letting vLLM
    crash on a collision.

    Honors any externally-set ``VLLM_PORT`` (callers can pin a port
    by setting the env var before invoking the inference script).
    """
    if os.environ.get(env_var):
        return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        os.environ[env_var] = str(s.getsockname()[1])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Unified generation config for all backends.

    ``sampling_params`` uses backend-agnostic names (``max_tokens``,
    ``temperature``, ``top_p``, ``top_k``).  For HF, ``max_tokens`` is
    automatically translated to ``max_new_tokens``.

    ``backend_kwargs`` are passed directly to the backend constructor
    (e.g. ``tensor_parallel_size`` for vLLM, ``concurrency`` for HF).

    ``enable_thinking`` controls reasoning-mode toggles for chat
    templates that support them (currently auto-detected for Qwen3+
    families). ``None`` leaves the model default in place. ``True`` /
    ``False`` flips the chat template's ``enable_thinking`` flag through
    ``chat_template_kwargs`` for vLLM/HF and ``apply_chat_template`` for
    Ray. Non-instruct mode ignores it.

    ``force_thinking`` appends ``<think>\\n`` to the rendered prompt
    (after the chat template) so the model is committed to opening a
    reasoning block rather than being free to skip thinking. Qwen-only
    (auto-detected). Incompatible with ``enable_thinking=False`` —
    that mode prefills an empty ``<think>\\n\\n</think>\\n\\n`` block
    as a hard switch, which would conflict with the forced opener;
    the two combined raise at construction. When on, ``raw_response``
    is prefixed with ``<think>\\n`` so the saved string still contains
    a complete reasoning block (the model never emits the opener
    itself, since we prefilled it).
    """
    model_path: str
    backend: str = "vllm"           # "vllm", "hf", "ray_vllm"
    mode: str = "instruct"          # "instruct" or "base"
    system_prompt: Optional[str] = None
    prefill_assistant: Optional[str] = None
    chat_template: Optional[str] = None
    # HF id or local path of a tokenizer whose ``chat_template`` should
    # be borrowed when ``model_path``'s own tokenizer lacks one — useful
    # for base models that ship without a chat template but should be
    # evaluated with a chat-formatted prompt (e.g. Llama-3.1-8B base
    # paired with -Instruct). Resolved into ``chat_template`` at the
    # call site if ``chat_template`` is still ``None``.
    chat_template_tokenizer: Optional[str] = None
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)
    enable_thinking: Optional[bool] = None
    # Cap the number of tokens the model is allowed to spend inside the
    # ``<think>...</think>`` block. Currently auto-detected for Qwen3
    # (and Qwen3.x) — the chat template recognizes ``thinking_budget``
    # inside ``chat_template_kwargs`` and emits a system-level cue that
    # tells the model to wrap up. ``None`` leaves the model default
    # ("think as long as you want"). Useful for judging-style tasks
    # where the answer is short and long deliberation just burns tokens.
    thinking_budget: Optional[int] = None
    force_thinking: bool = False
    # Restrict sampling to a subset of the vocabulary. When set, masked-out
    # tokens have ``-inf`` logits before sampling, so they have exactly zero
    # probability of being chosen — this is a HARD constraint, not a soft
    # bias. Accepts either a file path (``.json`` sparse spec / ``.npy``
    # dense bool array / ``.safetensors`` dense bool tensor — see
    # ``inference.vocab_mask`` for the schemas) or a programmatically
    # constructed ``VocabMaskSpec``. Honored by all three backends and
    # the streaming chat CLI.
    output_vocab_mask: Optional[Union[str, "VocabMaskSpec"]] = None

    def __post_init__(self):
        if (
            self.force_thinking
            and self.enable_thinking is False
            and detect_model_family(self.model_path) == "qwen"
        ):
            raise ValueError(
                "force_thinking=True is incompatible with "
                "enable_thinking=False: the latter prefills an empty "
                "<think>\\n\\n</think>\\n\\n block via the chat template "
                "as a 'no-thinking' hard switch, while force_thinking "
                "would append a conflicting <think>\\n opener after it. "
                "Either drop force_thinking or set enable_thinking to "
                "True / None."
            )
        # Resolve string-path masks into a ``VocabMaskSpec`` at config
        # construction time, so file errors (missing file, bad schema)
        # surface before the model is loaded. Programmatic specs pass
        # through unchanged. The dense tensor isn't built here — that
        # requires the tokenizer's vocab size, which only the backend has.
        if isinstance(self.output_vocab_mask, str):
            self.output_vocab_mask = VocabMaskSpec.from_file(self.output_vocab_mask)


# Model families whose chat templates accept ``enable_thinking``. Only
# Qwen is wired up today (Qwen3 / Qwen3.x — the templates ship with the
# ``{% if enable_thinking %}`` block); add more as they show up.
_THINKING_FAMILIES = ("qwen",)


def detect_model_family(model_path: str) -> Optional[str]:
    """Return a coarse model-family tag (``"qwen"``, ...) or ``None``.

    Detection is path-based — it works for both HF Hub IDs (e.g.
    ``Qwen/Qwen3-8B``) and local directories that include the family
    name in the path. Used to decide which chat-template kwargs are
    safe to pass.
    """
    if not model_path:
        return None
    needle = model_path.lower()
    for family in _THINKING_FAMILIES:
        if family in needle:
            return family
    return None


def resolve_chat_template(config: "GenerationConfig") -> Optional[str]:
    """Return the jinja2 chat template string for ``config``, if any.

    Precedence:
      1. ``config.chat_template`` (raw jinja2 string) wins outright.
      2. Otherwise, if ``config.chat_template_tokenizer`` is set, load
         that tokenizer and return its ``chat_template`` attribute.
      3. Otherwise, return ``None`` so the backend falls back to
         ``model_path``'s own tokenizer template.
    """
    if config.chat_template:
        return config.chat_template
    if config.chat_template_tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            config.chat_template_tokenizer, trust_remote_code=True,
        )
        template = getattr(tok, "chat_template", None)
        if not template:
            raise ValueError(
                f"--chat-template-tokenizer={config.chat_template_tokenizer!r} "
                "does not define a chat_template."
            )
        return template
    return None


def build_chat_template_kwargs(config: "GenerationConfig") -> Dict[str, Any]:
    """Build a ``chat_template_kwargs`` dict honoring ``enable_thinking``
    + ``thinking_budget``.

    Returns an empty dict when no overrides apply. Currently only the
    Qwen family is wired up — non-Qwen models silently ignore both
    flags so callers can pass them unconditionally without
    family-specific branching.
    """
    kwargs: Dict[str, Any] = {}
    family = detect_model_family(config.model_path)
    if family != "qwen":
        return kwargs
    if config.enable_thinking is not None:
        kwargs["enable_thinking"] = bool(config.enable_thinking)
    if config.thinking_budget is not None:
        # Qwen3 chat templates read ``thinking_budget`` from the
        # template kwargs; the value is the cap on tokens spent inside
        # ``<think>...</think>``. Only meaningful when thinking is on
        # — pass it through regardless and let the template gate it.
        kwargs["thinking_budget"] = int(config.thinking_budget)
    return kwargs


# Literal prefix appended after the chat template's generation prompt
# to commit the model to opening a ``<think>`` block. The Qwen3 chat
# template renders ``<|im_start|>assistant\n`` (and nothing else when
# ``enable_thinking`` is True/None); concatenating this string forces
# the model to continue from inside an open thinking block.
_FORCE_THINKING_PREFIX = "<think>\n"


def _force_thinking_active(config: "GenerationConfig") -> bool:
    """Whether ``force_thinking`` actually takes effect for this config.

    Auto-restricted to the Qwen family — non-Qwen models silently
    ignore the flag, matching ``enable_thinking`` / ``thinking_budget``.
    The contradictory combo with ``enable_thinking=False`` is rejected
    in ``GenerationConfig.__post_init__``, so by the time this runs
    we only need the family + flag check.
    """
    return (
        config.force_thinking
        and detect_model_family(config.model_path) == "qwen"
    )


def apply_force_thinking(rendered_prompt: str, config: "GenerationConfig") -> str:
    """Append ``<think>\\n`` to a rendered prompt when ``force_thinking``
    is active for this config; otherwise return it unchanged."""
    if _force_thinking_active(config):
        return rendered_prompt + _FORCE_THINKING_PREFIX
    return rendered_prompt


def format_raw_response(raw_response: str, config: "GenerationConfig") -> str:
    """Prepend ``<think>\\n`` to a raw response when ``force_thinking``
    is active, so the saved string still contains a complete reasoning
    block. The model never emits the opening ``<think>`` itself — we
    prefilled it — so without this reconstruction the saved raw output
    would start mid-reasoning."""
    if _force_thinking_active(config):
        return _FORCE_THINKING_PREFIX + raw_response
    return raw_response


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def build_chat_messages(
    text: Union[str, List[Dict[str, str]]],
    system_prompt: Optional[str] = None,
    prefill_assistant: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build a chat-style message list from a string or existing messages."""
    if isinstance(text, str):
        body: List[Dict[str, str]] = [{"role": "user", "content": text}]
    elif isinstance(text, list):
        body = list(text)
    else:
        raise ValueError(f"Expected str or list of messages, got {type(text)}")
    messages: List[Dict[str, str]] = []
    # Prepend a system message when one is supplied -- including the empty string,
    # which yields an explicit EMPTY system block (use ``is not None``, not a
    # truthiness test, so "" is honored). This suppresses chat templates that
    # inject a hardcoded default system prompt when none is present (e.g. Olmo-3).
    # Skip when the body already opens with a system turn so we never double up.
    if system_prompt is not None and not (body and body[0].get("role") == "system"):
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(body)
    if prefill_assistant:
        messages.append({"role": "assistant", "content": prefill_assistant})
    return messages


def _translate_sampling_for_hf(params: Dict[str, Any]) -> Dict[str, Any]:
    """Translate generic (vLLM-flavored) sampling params to HF generate kwargs.

    Translations:
      * ``max_tokens`` → ``max_new_tokens``.
      * ``top_k=-1`` (vLLM "no filter" sentinel) → ``top_k=0`` (HF "no filter").
      * ``do_sample`` is set from ``temperature`` when not supplied: greedy
        (``do_sample=False``) for ``temperature<=0``, sampling otherwise.
        HF defaults to greedy regardless of ``temperature``/``top_p`` if
        ``do_sample`` is not explicit, which silently neutralizes the
        caller's sampling intent.

    Dropped (vLLM-only keys that raise ``TypeError`` on HF ``generate``):
      ``frequency_penalty``, ``presence_penalty``, ``seed``, ``stop``,
      ``skip_special_tokens``. Callers needing seeded sampling should set
      ``torch.manual_seed`` upstream; callers needing stop strings should
      build a ``StoppingCriteriaList``.
    """
    _DROP = {
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "stop",
        "skip_special_tokens",
    }
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if k in _DROP:
            continue
        if k == "max_tokens":
            out["max_new_tokens"] = v
        else:
            out[k] = v
    if out.get("top_k") == -1:
        out["top_k"] = 0
    if "do_sample" not in out:
        temp = out.get("temperature", 1.0)
        do_sample = isinstance(temp, (int, float)) and temp > 0.0
        out["do_sample"] = do_sample
        if not do_sample:
            # Greedy: HF ignores ``temperature`` here, and passing
            # ``temperature=0`` triggers a confusing warning. Drop it.
            out.pop("temperature", None)
            # Same for the sampling-only filters.
            out.pop("top_p", None)
            out.pop("top_k", None)
            out.pop("min_p", None)
    return out


def _build_max_memory(gpu_ids: List[int], reserve_gib: float = 4.0, reserve_pct: float = 0.12):
    import torch
    gpu_set = set(gpu_ids)
    mm: Dict[Any, str] = {}
    for i in range(torch.cuda.device_count()):
        if i in gpu_set:
            free_b, total_b = torch.cuda.mem_get_info(i)
            free_g, total_g = free_b / (1024**3), total_b / (1024**3)
            reserve = max(reserve_gib, reserve_pct * total_g)
            usable = min(max(free_g - reserve, 0.0), 0.75 * free_g)
            mm[i] = f"{int(usable)}GiB"
        else:
            mm[i] = "0GiB"
    mm["cpu"] = "64GiB"
    return mm


# ---------------------------------------------------------------------------
# HF vocab-mask helpers (used by both single-GPU and TP workers)
# ---------------------------------------------------------------------------

def _hf_build_mask_processor(
    vocab_mask_spec: Optional["VocabMaskSpec"], tokenizer, device,
):
    """Materialize a HF logits processor for ``vocab_mask_spec`` or None.

    Run inside each HF worker after the tokenizer/model are loaded; the
    materialized bool tensor is moved to ``device`` so the masked-fill
    on each generate step has no host↔device transfer.
    """
    if vocab_mask_spec is None:
        return None
    mask = vocab_mask_spec.materialize(len(tokenizer))
    warn_if_eos_masked(mask, getattr(tokenizer, "eos_token_id", None))
    return make_hf_logits_processor(mask.to(device))


def _hf_inject_mask_into_gen_kwargs(gen_kwargs, mask_processor):
    """Prepend ``mask_processor`` to ``gen_kwargs['logits_processor']``.

    Any caller-supplied processor list is preserved (runs after ours so
    post-mask shaping like temperature warping is unchanged).
    """
    if mask_processor is None:
        return
    from transformers import LogitsProcessorList
    existing = gen_kwargs.pop("logits_processor", None)
    procs = [mask_processor]
    if existing is not None:
        procs.extend(list(existing))
    gen_kwargs["logits_processor"] = LogitsProcessorList(procs)


# ---------------------------------------------------------------------------
# HF worker (runs in spawned process)
# ---------------------------------------------------------------------------

def _hf_tp_generate_worker(
    rank: int,
    world_size: int,
    master_port: int,
    model_path: str,
    prompts: List[Dict[str, Any]],
    batch_size: int,
    model_kwargs: Dict[str, Any],
    tp_plan: Any,
    return_dict,
    progress_counter,
    return_raw: bool = False,
    vocab_mask_spec: Optional["VocabMaskSpec"] = None,
):
    """HuggingFace tensor-parallel worker.

    One process per TP rank; all ranks participate in every forward
    pass on the same prompts. Only rank 0 writes results to
    ``return_dict``. Each rank pins to its own GPU (``LOCAL_RANK ==
    rank``) and joins a single ``nccl`` process group; HF's
    ``tp_plan="auto"`` consumes the standard torchrun env vars
    (``RANK``/``WORLD_SIZE``/``LOCAL_RANK``/``MASTER_ADDR``/
    ``MASTER_PORT``) — see
    https://huggingface.co/docs/transformers/perf_infer_gpu_multi.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    is_rank_0 = rank == 0

    # Partial-finetune auto-detection — check this BEFORE LoRA detection
    # (the two formats are mutually exclusive at save time). When the
    # input path is a partial-finetune directory, swap to the base model
    # for the initial load and remember the partial dir so we can
    # overlay its weights below.
    partial_dir: Optional[str] = None
    if _is_partial_finetune_dir(model_path):
        partial_dir = model_path
        base_model, _partial_weights = _read_partial_config(partial_dir)
        model_path = base_model

    # LoRA-adapter auto-detection — same shape as the non-TP worker.
    adapter_dir: Optional[str] = None
    if _is_lora_adapter_dir(model_path):
        adapter_dir = model_path
        base_model, _ = _read_adapter_config(adapter_dir)
        model_path = base_model

    tokenizer_path = partial_dir or adapter_dir or model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=model_kwargs.get("trust_remote_code", False),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    mk = {k: v for k, v in model_kwargs.items() if k != "tp_plan"}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, tp_plan=tp_plan, **mk,
    )
    if partial_dir is not None:
        # Overlay the partial-finetune weights on top of the loaded base.
        # ``strict=False`` because the partial safetensors carries only a
        # subset of parameters — the rest are already in place from the
        # base load. ``unexpected`` keys, however, indicate a real mismatch
        # (e.g. the partial save was for a different architecture) and
        # we raise.
        import safetensors.torch
        _, partial_weights = _read_partial_config(partial_dir)
        partial_sd = safetensors.torch.load_file(
            partial_weights, device=f"cuda:{rank}",
        )
        _, unexpected = model.load_state_dict(partial_sd, strict=False)
        if unexpected:
            raise RuntimeError(
                f"partial_model.safetensors at {partial_weights} contains "
                f"unexpected keys not in base model: {unexpected[:5]}..."
            )
    if adapter_dir is not None:
        # ``low_cpu_mem_usage=True`` keeps the adapter shells on
        # ``meta`` until the safetensors land directly on this rank's
        # GPU — paired with ``torch_device`` so the saved adapter
        # weights are streamed straight onto ``cuda:{rank}`` instead
        # of staging through CPU.
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, adapter_dir,
            torch_device=f"cuda:{rank}",
            low_cpu_mem_usage=True,
        )
        model = model.merge_and_unload()
    model.eval()

    mask_processor = _hf_build_mask_processor(
        vocab_mask_spec, tokenizer, f"cuda:{rank}",
    )

    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            ids = [x["id"] for x in batch]
            texts = [x["text"] for x in batch]

            gen_kwargs = batch[0].get("generation_kwargs", {}).copy()
            gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
            gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)
            _hf_inject_mask_into_gen_kwargs(gen_kwargs, mask_processor)

            tok_max_length = gen_kwargs.pop("max_length", None)
            include_special = gen_kwargs.pop("include_special_tokens", False)
            include_prompt = gen_kwargs.pop("include_prompt_in_response", False)

            encoded = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=(tok_max_length is not None), max_length=tok_max_length,
            )
            encoded = {k: v.to(f"cuda:{rank}") for k, v in encoded.items()}
            context_len = encoded["input_ids"].shape[1]

            with torch.inference_mode():
                sequences = model.generate(**encoded, **gen_kwargs)

            if is_rank_0:
                for b, seq in enumerate(sequences):
                    to_decode = seq if include_prompt else seq[context_len:]
                    text = tokenizer.decode(to_decode, skip_special_tokens=not include_special)
                    if return_raw:
                        raw_response = tokenizer.decode(seq[context_len:], skip_special_tokens=False)
                        return_dict[ids[b]] = {"text": text, "raw_response": raw_response}
                    else:
                        return_dict[ids[b]] = text
                progress_counter.value += len(batch)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _hf_generate_worker(
    gpu_ids: List[int],
    model_path: str,
    prompts: List[Dict[str, Any]],
    batch_size: int,
    model_kwargs: Dict[str, Any],
    return_dict,
    progress_counter,
    return_raw: bool = False,
    vocab_mask_spec: Optional["VocabMaskSpec"] = None,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Partial-finetune auto-detection: check BEFORE LoRA. When the input
    # path is a partial-finetune dir, swap to the base model for the
    # initial load and remember the partial dir for the in-memory weight
    # overlay below.
    partial_dir: Optional[str] = None
    if _is_partial_finetune_dir(model_path):
        partial_dir = model_path
        base_model, _partial_weights = _read_partial_config(partial_dir)
        model_path = base_model

    # LoRA-adapter auto-detection: when ``model_path`` is a PEFT adapter
    # directory, point the base loader at the base referenced inside
    # ``adapter_config.json`` and remember the adapter dir so we can
    # merge it in below. The tokenizer prefers the adapter dir (PEFT's
    # ``save_pretrained`` wrote tokenizer files alongside the adapter)
    # but falls back to the base when the adapter dir didn't carry one.
    adapter_dir: Optional[str] = None
    if _is_lora_adapter_dir(model_path):
        adapter_dir = model_path
        base_model, _ = _read_adapter_config(adapter_dir)
        model_path = base_model

    tokenizer_path = partial_dir or adapter_dir or model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=model_kwargs.get("trust_remote_code", False),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if len(gpu_ids) == 1:
        # Single-GPU load — skip device_map="auto" entirely and pin the
        # whole model to the assigned device. ``device_map={"": dev}``
        # is the supported way to do this with HF Accelerate while
        # remaining compatible with quantized model loaders (FP8,
        # bitsandbytes, …) that look for a device_map.
        dev = f"cuda:{gpu_ids[0]}"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map={"": dev},
            low_cpu_mem_usage=False, **model_kwargs,
        )
    else:
        max_memory = _build_max_memory(gpu_ids, reserve_pct=0.25)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", max_memory=max_memory,
            low_cpu_mem_usage=False, **model_kwargs,
        )
        dev = None
    if partial_dir is not None:
        # Overlay partial-finetune weights on top of the loaded base.
        # ``strict=False`` because the partial safetensors carries only a
        # subset of parameters — the rest are already in place from the
        # base load. Unexpected keys raise (architecture mismatch).
        import safetensors.torch
        _, partial_weights = _read_partial_config(partial_dir)
        partial_sd = safetensors.torch.load_file(
            partial_weights, device=dev if dev is not None else "cpu",
        )
        _, unexpected = model.load_state_dict(partial_sd, strict=False)
        if unexpected:
            raise RuntimeError(
                f"partial_model.safetensors at {partial_weights} contains "
                f"unexpected keys not in base model: {unexpected[:5]}..."
            )
    if adapter_dir is not None:
        # Apply the LoRA adapter on top of the loaded base and fold
        # the deltas into the weights so the rest of the worker treats
        # the model like any other HF causal LM.
        #
        # ``low_cpu_mem_usage=True`` instructs PEFT to allocate adapter
        # shells on the ``meta`` device before loading the saved
        # weights, and ``torch_device=dev`` makes the safetensors load
        # land directly on the target GPU — no CPU-resident copy of
        # the adapter weights at any point. Falls back to PEFT's own
        # ``infer_device`` for the multi-GPU (sharded base) path,
        # where a single device string would be wrong.
        from peft import PeftModel
        peft_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
        if dev is not None:
            peft_kwargs["torch_device"] = dev
        model = PeftModel.from_pretrained(model, adapter_dir, **peft_kwargs)
        model = model.merge_and_unload()
    model.eval()

    mask_processor = _hf_build_mask_processor(
        vocab_mask_spec, tokenizer, model.device,
    )

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        ids = [x["id"] for x in batch]
        texts = [x["text"] for x in batch]

        gen_kwargs = batch[0].get("generation_kwargs", {}).copy()
        gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)
        _hf_inject_mask_into_gen_kwargs(gen_kwargs, mask_processor)

        tok_max_length = gen_kwargs.pop("max_length", None)
        include_special = gen_kwargs.pop("include_special_tokens", False)
        include_prompt = gen_kwargs.pop("include_prompt_in_response", False)

        encoded = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=(tok_max_length is not None), max_length=tok_max_length,
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        context_len = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            sequences = model.generate(**encoded, **gen_kwargs)

        for b, seq in enumerate(sequences):
            to_decode = seq if include_prompt else seq[context_len:]
            text = tokenizer.decode(to_decode, skip_special_tokens=not include_special)
            if return_raw:
                raw_response = tokenizer.decode(seq[context_len:], skip_special_tokens=False)
                return_dict[ids[b]] = {"text": text, "raw_response": raw_response}
            else:
                return_dict[ids[b]] = text

        progress_counter.value += len(batch)


# ---------------------------------------------------------------------------
# Base class + registry
# ---------------------------------------------------------------------------

class GenerationModel(ABC):
    _registry: Dict[str, type] = {}

    @abstractmethod
    def generate(
        self,
        prompts: Union[List[str], List[Dict[str, Any]]],
        *,
        return_raw: bool = False,
        **kwargs,
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """Generate text for a list of prompts.

        Each prompt can be a plain string or a dict with ``text`` and
        optional ``sampling_params`` overrides.

        When ``return_raw=True``, returns a list of dicts with keys
        ``text`` (the post-decode output, matching the default
        ``List[str]`` return), ``raw_prompt`` (the formatted input
        actually fed to the model — post chat-template for instruct
        mode, the input string for base mode), and ``raw_response``
        (the model's output decoded with special tokens preserved).
        Default ``False`` returns ``List[str]`` for backwards
        compatibility.
        """

    @staticmethod
    def from_config(config: GenerationConfig) -> GenerationModel:
        if config.backend not in GenerationModel._registry:
            raise ValueError(
                f"Unknown backend: {config.backend}. "
                f"Available: {list(GenerationModel._registry.keys())}"
            )
        return GenerationModel._registry[config.backend](config)

    @staticmethod
    def register(name: str):
        def decorator(cls):
            GenerationModel._registry[name] = cls
            return cls
        return decorator

    def _normalize_prompts(
        self,
        prompts: Union[List[str], List[Dict[str, Any]], List[List[Dict[str, str]]]],
        defaults: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for i, p in enumerate(prompts):
            if isinstance(p, str):
                normalized.append({"id": i, "text": p, "sampling_params": defaults.copy()})
            elif isinstance(p, list):
                # A list is a chat-message sequence — only meaningful in
                # instruct mode, where ``build_chat_messages`` will splat
                # it into the conversation.
                normalized.append({"id": i, "text": p, "sampling_params": defaults.copy()})
            elif isinstance(p, dict):
                sp = defaults.copy()
                sp.update(p.get("sampling_params", {}))
                normalized.append({"id": i, "text": p["text"], "sampling_params": sp})
            else:
                raise ValueError(f"Invalid prompt type: {type(p)}")
        return normalized


# ---------------------------------------------------------------------------
# VLLM backend
# ---------------------------------------------------------------------------

@GenerationModel.register("vllm")
class VLLMGenerationModel(GenerationModel):
    def __init__(self, config: GenerationConfig):
        # Pick a random free ``VLLM_PORT`` BEFORE importing vLLM, so
        # parallel engines on the same node don't collide on the
        # default ZMQ / IPC port. Honors a caller-supplied
        # ``VLLM_PORT`` env var.
        _ensure_vllm_random_port()

        from vllm import LLM

        self.config = config
        self.sampling_params = config.sampling_params

        model_path = config.model_path
        bk = config.backend_kwargs.copy()

        # Partial-finetune pre-merge. vLLM has no runtime layer-overlay
        # API (its LoRA path is a special-purpose low-rank fuser, not a
        # general weight-override hook), so we materialize a fully-merged
        # HF directory in a tempdir (honors $TMPDIR) and point vLLM at
        # that. The merge tempdir is registered for ``atexit`` cleanup
        # inside ``_materialize_partial_merge`` so the merged ~30GB
        # doesn't leak when this process exits.
        self._merge_tempdir: Optional[str] = None
        if _is_partial_finetune_dir(model_path):
            self._merge_tempdir = _materialize_partial_merge(model_path)
            model_path = self._merge_tempdir

        # LoRA-adapter auto-detection. When ``model_path`` is a PEFT
        # adapter directory (no base weights, just adapter_config.json +
        # adapter_model.safetensors), swap the engine's ``model`` arg to
        # the base referenced in ``adapter_config.json``, turn on vLLM's
        # LoRA path, and stash a ``LoRARequest`` to attach to every
        # generate / chat call below. Caller-supplied engine flags win
        # via ``setdefault``.
        self.lora_request = None
        if _is_lora_adapter_dir(model_path):
            from vllm.lora.request import LoRARequest
            base_model, lora_r = _read_adapter_config(model_path)
            self.lora_request = LoRARequest(
                lora_name="adapter",
                lora_int_id=1,
                lora_path=model_path,
            )
            model_path = base_model
            bk.setdefault("enable_lora", True)
            bk.setdefault("max_loras", 1)
            # vLLM's default ``max_lora_rank`` is 16; bump only when
            # the trained adapter is wider.
            bk.setdefault("max_lora_rank", max(16, lora_r))

        llm_kwargs: Dict[str, Any] = {"model": model_path}

        cache_dir = bk.pop("compilation_cache_dir", None)
        if cache_dir:
            try:
                from vllm.config import CompilationConfig
                llm_kwargs["compilation_config"] = CompilationConfig(cache_dir=cache_dir)
            except ImportError:
                pass

        llm_kwargs.update(bk)
        self.llm = LLM(**llm_kwargs)

        # Build the vocab-mask allow-list once: materialize the dense
        # bool mask against the loaded tokenizer's vocab size, then
        # collapse it to the sorted list of permitted token ids. vLLM's
        # V1 engine has no per-request ``SamplingParams.logits_processors``
        # field, so the hard mask is applied through ``allowed_token_ids``
        # (sampler retains only these ids, ``-inf``-masks the rest).
        self._allowed_token_ids = None
        if config.output_vocab_mask is not None:
            tok = self.llm.get_tokenizer()
            mask = config.output_vocab_mask.materialize(len(tok))
            warn_if_eos_masked(mask, getattr(tok, "eos_token_id", None))
            self._allowed_token_ids = allowed_token_ids_from_mask(mask)

    def generate(self, prompts, *, return_raw: bool = False, **kwargs):
        if not prompts:
            return []

        from vllm import SamplingParams

        normalized = self._normalize_prompts(prompts, self.sampling_params)

        # Data-parallel partitioning
        try:
            pc = self.llm.llm_engine.vllm_config.parallel_config
            dp_rank = getattr(pc, "data_parallel_rank", 0)
            dp_size = getattr(pc, "data_parallel_size", 1)
        except Exception:
            dp_rank, dp_size = 0, 1

        local = [p for p in normalized if p["id"] % dp_size == dp_rank]

        if not local:
            local_results: list = []
        else:
            sp_list = []
            for p in local:
                sp_kwargs = dict(p["sampling_params"])
                # Apply the vocab mask via ``allowed_token_ids``. Intersect
                # with any caller-supplied allow-list so both constraints
                # hold; absent one, just use the mask's list.
                if self._allowed_token_ids is not None:
                    existing = sp_kwargs.get("allowed_token_ids")
                    if existing:
                        sp_kwargs["allowed_token_ids"] = sorted(
                            set(existing) & set(self._allowed_token_ids)
                        )
                    else:
                        sp_kwargs["allowed_token_ids"] = self._allowed_token_ids
                sp_list.append(SamplingParams(**sp_kwargs))
            use_tqdm = dp_rank == 0

            # When a PEFT adapter was auto-detected at __init__ time,
            # every chat / generate call needs the corresponding
            # ``LoRARequest`` attached so the engine knows to fuse the
            # adapter on top of the base for this batch.
            lora_kwargs: Dict[str, Any] = (
                {"lora_request": self.lora_request} if self.lora_request else {}
            )

            if self.config.mode == "instruct":
                inputs = [
                    build_chat_messages(p["text"], self.config.system_prompt, self.config.prefill_assistant)
                    for p in local
                ]
                ct_kwargs = build_chat_template_kwargs(self.config)
                if _force_thinking_active(self.config):
                    # vLLM's ``chat()`` renders the chat template
                    # internally and exposes no post-template hook, so
                    # for force_thinking we render manually and feed
                    # the resulting strings to ``generate()`` instead.
                    tok = self.llm.get_tokenizer()
                    apply_kwargs: Dict[str, Any] = dict(
                        tokenize=False, add_generation_prompt=True, **ct_kwargs,
                    )
                    if self.config.prefill_assistant:
                        apply_kwargs.pop("add_generation_prompt")
                        apply_kwargs["continue_final_message"] = True
                    rendered = [
                        apply_force_thinking(
                            tok.apply_chat_template(msgs, **apply_kwargs),
                            self.config,
                        )
                        for msgs in inputs
                    ]
                    outputs = self.llm.generate(
                        rendered, sampling_params=sp_list, use_tqdm=use_tqdm,
                        **lora_kwargs,
                    )
                else:
                    chat_kwargs: Dict[str, Any] = dict(
                        messages=inputs,
                        sampling_params=sp_list,
                        use_tqdm=use_tqdm,
                    )
                    # vLLM forwards ``chat_template_kwargs`` to the
                    # tokenizer's ``apply_chat_template`` call. Only attach
                    # when there is something to pass — older vLLM builds
                    # may not accept the kwarg and an empty dict is wasted.
                    if ct_kwargs:
                        chat_kwargs["chat_template_kwargs"] = ct_kwargs
                    # Borrow a chat template from a sibling tokenizer (or
                    # raw jinja2 string) when the model's own tokenizer
                    # ships without one — e.g. Llama-3.1-8B base.
                    resolved_template = resolve_chat_template(self.config)
                    if resolved_template:
                        chat_kwargs["chat_template"] = resolved_template
                    chat_kwargs.update(lora_kwargs)
                    outputs = self.llm.chat(**chat_kwargs)
            else:
                inputs = [p["text"] for p in local]
                outputs = self.llm.generate(
                    inputs, sampling_params=sp_list, use_tqdm=use_tqdm,
                    **lora_kwargs,
                )

            if return_raw:
                # vLLM populates ``RequestOutput.prompt`` with the
                # rendered prompt string for both ``generate`` and
                # ``chat``. For raw response we re-decode the output
                # token ids without ``skip_special_tokens`` — the
                # tokenizer is fetched once per call.
                tok = self.llm.get_tokenizer()
                local_results = [
                    {
                        "id": p["id"],
                        "generated_text": o.outputs[0].text,
                        "raw_prompt": o.prompt,
                        "raw_response": format_raw_response(
                            tok.decode(
                                o.outputs[0].token_ids, skip_special_tokens=False,
                            ),
                            self.config,
                        ),
                    }
                    for p, o in zip(local, outputs)
                ]
            else:
                local_results = [
                    {"id": p["id"], "generated_text": o.outputs[0].text}
                    for p, o in zip(local, outputs)
                ]

        return self._gather(local_results, dp_size, return_raw=return_raw)

    @staticmethod
    def _gather(local_results: list, dp_size: int, return_raw: bool = False):
        def shape(r):
            if return_raw:
                return {
                    "text": r["generated_text"],
                    "raw_prompt": r["raw_prompt"],
                    "raw_response": r["raw_response"],
                }
            return r["generated_text"]

        if dp_size <= 1:
            local_results.sort(key=lambda x: x["id"])
            return [shape(r) for r in local_results]
        try:
            import torch.distributed as dist
            from vllm.distributed.parallel_state import get_world_group
            cpu_group = get_world_group().cpu_group
            all_results: list = [None] * dist.get_world_size(group=cpu_group)
            dist.all_gather_object(all_results, local_results, group=cpu_group)
            merged = [r for rank in all_results for r in rank]
            merged.sort(key=lambda x: x["id"])
            return [shape(r) for r in merged]
        except Exception:
            local_results.sort(key=lambda x: x["id"])
            return [shape(r) for r in local_results]


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------

@GenerationModel.register("hf")
class HFGenerationModel(GenerationModel):
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.sampling_params = config.sampling_params
        bk = config.backend_kwargs
        self.concurrency = bk.get("concurrency", 1)
        self.n_devices = bk.get("n_devices_per_instance", 1)
        self.batch_size = bk.get("batch_size", 4)
        self.model_kwargs = bk.get("model_kwargs", {})
        # Opt-in TP: when set (e.g. "auto"), spawn ``n_devices_per_instance``
        # ranks as a single torch.distributed group and load with
        # ``from_pretrained(tp_plan=...)``. Concurrency must be 1 in TP mode
        # — multiple parallel TP groups would each need their own process
        # group, which the current launcher doesn't manage.
        self.tp_plan = bk.get("tp_plan", None)
        if self.tp_plan is not None and self.concurrency != 1:
            raise ValueError(
                "tp_plan is incompatible with concurrency > 1; "
                "use a single TP group spanning n_devices_per_instance GPUs."
            )
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.model_kwargs.get("trust_remote_code", False),
            )
        return self._tokenizer

    def _apply_chat_template(self, text):
        tok = self._get_tokenizer()
        messages = build_chat_messages(text, self.config.system_prompt, self.config.prefill_assistant)
        ct_kwargs = build_chat_template_kwargs(self.config)
        resolved_template = resolve_chat_template(self.config)
        if self.config.prefill_assistant:
            rendered = tok.apply_chat_template(
                messages, chat_template=resolved_template,
                tokenize=False, continue_final_message=True, **ct_kwargs,
            )
        else:
            rendered = tok.apply_chat_template(
                messages, chat_template=resolved_template,
                tokenize=False, add_generation_prompt=True, **ct_kwargs,
            )
        return apply_force_thinking(rendered, self.config)

    def generate(self, prompts, *, return_raw: bool = False, **kwargs):
        if not prompts:
            return []
        normalized = self._normalize_prompts(prompts, self.sampling_params)
        preprocessed = []
        for item in normalized:
            text = self._apply_chat_template(item["text"]) if self.config.mode == "instruct" else item["text"]
            preprocessed.append({
                "id": item["id"],
                "text": text,
                "generation_kwargs": _translate_sampling_for_hf(item["sampling_params"]),
            })
        return self._run_workers(preprocessed, return_raw=return_raw)

    def _run_workers(self, prompts, return_raw: bool = False):
        import torch
        from multiprocessing import get_context, Manager
        from tqdm import tqdm

        total_gpus = torch.cuda.device_count()
        needed = self.concurrency * self.n_devices
        if total_gpus < needed:
            raise RuntimeError(f"Need {needed} GPUs, only {total_gpus} available")

        if self.tp_plan is not None:
            return self._run_tp_workers(prompts, return_raw=return_raw)

        gpu_groups = [
            list(range(i * self.n_devices, (i + 1) * self.n_devices))
            for i in range(self.concurrency)
        ]
        splits: list = [[] for _ in range(self.concurrency)]
        for j, p in enumerate(prompts):
            splits[j % self.concurrency].append(p)

        ctx = get_context("spawn")
        manager = Manager()
        shared_out = manager.dict()
        progress = manager.Value("i", 0)
        processes = []

        for gpu_group, subset in zip(gpu_groups, splits):
            if not subset:
                continue
            p = ctx.Process(
                target=_hf_generate_worker,
                kwargs=dict(
                    gpu_ids=gpu_group,
                    model_path=self.config.model_path,
                    prompts=subset,
                    batch_size=self.batch_size,
                    model_kwargs=self.model_kwargs,
                    return_dict=shared_out,
                    progress_counter=progress,
                    return_raw=return_raw,
                    vocab_mask_spec=self.config.output_vocab_mask,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)

        with tqdm(total=len(prompts), desc="Generating") as pbar:
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

        if return_raw:
            # ``prompts`` here is post-template, so its ``text`` is the
            # ``raw_prompt`` actually fed to the model. The worker
            # already produced ``raw_response`` with special tokens —
            # the force_thinking reconstruction happens here so the
            # worker stays oblivious to config-level knobs.
            id_to_raw_prompt = {p["id"]: p["text"] for p in prompts}
            results = []
            for i in range(len(prompts)):
                payload = shared_out[i]
                results.append({
                    "text": payload["text"],
                    "raw_prompt": id_to_raw_prompt[i],
                    "raw_response": format_raw_response(
                        payload["raw_response"], self.config,
                    ),
                })
            return results

        results = [{"id": i, "generated_text": shared_out[i]} for i in range(len(prompts))]
        return [r["generated_text"] for r in results]

    def _run_tp_workers(self, prompts, return_raw: bool = False):
        """Single TP group: spawn ``self.n_devices`` ranks on GPUs 0..N-1,
        all participating in one torch.distributed group via NCCL. All
        ranks process every batch; only rank 0 populates ``shared_out``.
        """
        from multiprocessing import get_context, Manager
        from tqdm import tqdm

        world_size = self.n_devices
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            master_port = s.getsockname()[1]

        ctx = get_context("spawn")
        manager = Manager()
        shared_out = manager.dict()
        progress = manager.Value("i", 0)
        processes = []

        for rank in range(world_size):
            p = ctx.Process(
                target=_hf_tp_generate_worker,
                kwargs=dict(
                    rank=rank,
                    world_size=world_size,
                    master_port=master_port,
                    model_path=self.config.model_path,
                    prompts=prompts,
                    batch_size=self.batch_size,
                    model_kwargs=self.model_kwargs,
                    tp_plan=self.tp_plan,
                    return_dict=shared_out,
                    progress_counter=progress,
                    return_raw=return_raw,
                    vocab_mask_spec=self.config.output_vocab_mask,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)

        with tqdm(total=len(prompts), desc="Generating (TP)") as pbar:
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
                raise RuntimeError(f"TP worker exited with code {p.exitcode}")

        if return_raw:
            id_to_raw_prompt = {p["id"]: p["text"] for p in prompts}
            results = []
            for i in range(len(prompts)):
                payload = shared_out[i]
                results.append({
                    "text": payload["text"],
                    "raw_prompt": id_to_raw_prompt[i],
                    "raw_response": format_raw_response(
                        payload["raw_response"], self.config,
                    ),
                })
            return results

        return [shared_out[i] for i in range(len(prompts))]


# ---------------------------------------------------------------------------
# Ray + VLLM backend
# ---------------------------------------------------------------------------

@GenerationModel.register("ray_vllm")
class RayVLLMGenerationModel(GenerationModel):
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.sampling_params = config.sampling_params
        bk = config.backend_kwargs
        self.engine_kwargs = bk.get("engine_kwargs", {})
        self.concurrency = bk.get("concurrency", 1)
        self.batch_size = bk.get("batch_size", 1)
        self.max_concurrent_batches = bk.get("max_concurrent_batches", 128)
        # Default actor runtime env. Just an env var — Ray 2.55+
        # already vendored the uvloop 0.21 fix and the vllm `runner`/
        # `convert` API switch (ray-project/ray#59009 + the
        # AsyncEngineArgs `task` kwarg removal upstream in vllm
        # 0.11+). For older Ray (≤ 2.53) the right move is to upgrade
        # the host conda env (`pip install 'ray[data]>=2.55'`) — a
        # per-actor pip override here can't bridge a Ray
        # major-version mismatch between driver and workers anyway.
        # ``VLLM_USE_V1=1`` is kept as belt-and-braces against the
        # V0 engine path being exercised in some niche flow.
        self.runtime_env = bk.get("runtime_env", {
            "env_vars": {"VLLM_USE_V1": "1"},
        })
        self._processor = None
        self._ray = None

        if config.mode == "instruct":
            from transformers import AutoTokenizer
            tok_path = bk.get("tokenizer", config.model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(tok_path)
        else:
            self._tokenizer = None

    def _build_mask_processor(self):
        """Build the vocab-mask allow-list on the driver.

        Returns the sorted list of permitted token ids (or ``None``).
        Captured by the preprocess closure and shipped to Ray workers,
        where it is set on each row's ``SamplingParams.allowed_token_ids``
        — vLLM's V1 engine has no per-request ``logits_processors`` field.

        Base mode doesn't pre-load a tokenizer, but we need one here
        just to read ``vocab_size`` + ``eos_token_id``; loading is cheap.
        """
        if self.config.output_vocab_mask is None:
            return None
        tok = self._tokenizer
        if tok is None:
            from transformers import AutoTokenizer
            tok_path = self.config.backend_kwargs.get(
                "tokenizer", self.config.model_path,
            )
            tok = AutoTokenizer.from_pretrained(tok_path)
        mask = self.config.output_vocab_mask.materialize(len(tok))
        warn_if_eos_masked(mask, getattr(tok, "eos_token_id", None))
        return allowed_token_ids_from_mask(mask)

    def _make_preprocess(self):
        cfg = self.config
        tok = self._tokenizer
        ct_kwargs = build_chat_template_kwargs(cfg)
        resolved_template = resolve_chat_template(cfg)
        allowed_token_ids = self._build_mask_processor()

        def preprocess(row):
            if cfg.mode == "instruct":
                msgs = build_chat_messages(row["text"], cfg.system_prompt, cfg.prefill_assistant)
                if cfg.prefill_assistant:
                    prompt = tok.apply_chat_template(
                        msgs, chat_template=resolved_template,
                        tokenize=False, continue_final_message=True, **ct_kwargs,
                    )
                else:
                    prompt = tok.apply_chat_template(
                        msgs, chat_template=resolved_template,
                        tokenize=False, add_generation_prompt=True, **ct_kwargs,
                    )
                prompt = apply_force_thinking(prompt, cfg)
            else:
                prompt = row["text"]
            # Apply the vocab mask via ``allowed_token_ids`` (V1 engine has
            # no per-request ``logits_processors``). Intersect with any
            # caller-supplied allow-list. Copy the dict so we don't mutate
            # the upstream row.
            sampling_params = row["sampling_params"]
            if allowed_token_ids is not None:
                sampling_params = dict(sampling_params)
                existing = sampling_params.get("allowed_token_ids")
                if existing:
                    sampling_params["allowed_token_ids"] = sorted(
                        set(existing) & set(allowed_token_ids)
                    )
                else:
                    sampling_params["allowed_token_ids"] = allowed_token_ids
            # ``raw_prompt`` rides along on the row so the post-processor
            # can pair it back with the generation. Ray's vLLM processor
            # passes unknown columns through untouched.
            return {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "id": row["id"],
                "raw_prompt": prompt,
            }

        return preprocess

    def _ensure_processor(self):
        if self._processor is not None:
            return

        import ray
        from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Force a FIXED actor pool. ray.data.llm normalizes a bare
        # int concurrency=N to a (1, N) tuple — i.e. "autoscale
        # between 1 and N actors based on demand". With our typical
        # batch sizes the autoscaler concludes 1 actor is enough and
        # the rest of the GPU allocation sits idle. Passing
        # (N, N) forces exactly N actors, which is what callers who
        # already sized their slurm allocation around this number
        # actually want. Tuples are passed through untouched.
        if isinstance(self.concurrency, int):
            forced_concurrency = (self.concurrency, self.concurrency)
        else:
            forced_concurrency = self.concurrency

        processor_cfg_kwargs = dict(
            model_source=self.config.model_path,
            engine_kwargs=self.engine_kwargs,
            concurrency=forced_concurrency,
            batch_size=self.batch_size,
            max_concurrent_batches=self.max_concurrent_batches,
            apply_chat_template=False,
        )
        if self.runtime_env:
            processor_cfg_kwargs["runtime_env"] = self.runtime_env
        processor_cfg = vLLMEngineProcessorConfig(**processor_cfg_kwargs)
        self._processor = build_llm_processor(
            processor_cfg,
            preprocess=self._make_preprocess(),
            postprocess=lambda row: row,
        )
        self._ray = ray

    def generate(self, prompts, *, return_raw: bool = False, **kwargs):
        if not prompts:
            return []

        self._ensure_processor()
        normalized = self._normalize_prompts(prompts, self.sampling_params)
        if return_raw:
            # Force ``skip_special_tokens=False`` per-row so the
            # processor returns the raw decoded text. The Ray vLLM
            # path does not surface output token ids, so the cleaned
            # ``text`` and ``raw_response`` are the same string when
            # this flag is on — callers that need stripped text can
            # postprocess.
            for item in normalized:
                sp = dict(item["sampling_params"])
                sp["skip_special_tokens"] = False
                item["sampling_params"] = sp
        ds = self._ray.data.from_items(normalized)
        results_ds = self._processor(ds)

        try:
            results = list(results_ds.take_all())
        except AttributeError:
            results = list(results_ds.iter_rows())

        results.sort(key=lambda x: x["id"])

        if return_raw:
            return [
                {
                    "text": r["generated_text"],
                    "raw_prompt": r["raw_prompt"],
                    "raw_response": format_raw_response(
                        r["generated_text"], self.config,
                    ),
                }
                for r in results
            ]
        return [r["generated_text"] for r in results]
