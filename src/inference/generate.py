from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


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
    """
    model_path: str
    backend: str = "vllm"           # "vllm", "hf", "ray_vllm"
    mode: str = "instruct"          # "instruct" or "base"
    system_prompt: Optional[str] = None
    prefill_assistant: Optional[str] = None
    chat_template: Optional[str] = None   # HF / Ray only
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def build_chat_messages(
    text: Union[str, List[Dict[str, str]]],
    system_prompt: Optional[str] = None,
    prefill_assistant: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build a chat-style message list from a string or existing messages."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if isinstance(text, str):
        messages.append({"role": "user", "content": text})
    elif isinstance(text, list):
        messages.extend(text)
    else:
        raise ValueError(f"Expected str or list of messages, got {type(text)}")
    if prefill_assistant:
        messages.append({"role": "assistant", "content": prefill_assistant})
    return messages


def _translate_sampling_for_hf(params: Dict[str, Any]) -> Dict[str, Any]:
    """Translate generic sampling param names to HF generate kwargs."""
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if k == "max_tokens":
            out["max_new_tokens"] = v
        else:
            out[k] = v
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
# HF worker (runs in spawned process)
# ---------------------------------------------------------------------------

def _hf_generate_worker(
    gpu_ids: List[int],
    model_path: str,
    prompts: List[Dict[str, Any]],
    batch_size: int,
    model_kwargs: Dict[str, Any],
    return_dict,
    progress_counter,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=model_kwargs.get("trust_remote_code", False),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    max_memory = _build_max_memory(gpu_ids, reserve_pct=0.25)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", max_memory=max_memory,
        low_cpu_mem_usage=False, **model_kwargs,
    )
    model.eval()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        ids = [x["id"] for x in batch]
        texts = [x["text"] for x in batch]

        gen_kwargs = batch[0].get("generation_kwargs", {}).copy()
        gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)

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
            return_dict[ids[b]] = tokenizer.decode(to_decode, skip_special_tokens=not include_special)

        progress_counter.value += len(batch)


# ---------------------------------------------------------------------------
# Base class + registry
# ---------------------------------------------------------------------------

class GenerationModel(ABC):
    _registry: Dict[str, type] = {}

    @abstractmethod
    def generate(self, prompts: Union[List[str], List[Dict[str, Any]]], **kwargs) -> List[str]:
        """Generate text for a list of prompts.

        Each prompt can be a plain string or a dict with ``text`` and
        optional ``sampling_params`` overrides.
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
        prompts: Union[List[str], List[Dict[str, Any]]],
        defaults: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for i, p in enumerate(prompts):
            if isinstance(p, str):
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
        from vllm import LLM

        self.config = config
        self.sampling_params = config.sampling_params

        llm_kwargs: Dict[str, Any] = {"model": config.model_path}
        bk = config.backend_kwargs.copy()

        cache_dir = bk.pop("compilation_cache_dir", None)
        if cache_dir:
            try:
                from vllm.config import CompilationConfig
                llm_kwargs["compilation_config"] = CompilationConfig(cache_dir=cache_dir)
            except ImportError:
                pass

        llm_kwargs.update(bk)
        self.llm = LLM(**llm_kwargs)

    def generate(self, prompts, **kwargs):
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
            sp_list = [SamplingParams(**p["sampling_params"]) for p in local]
            use_tqdm = dp_rank == 0

            if self.config.mode == "instruct":
                inputs = [
                    build_chat_messages(p["text"], self.config.system_prompt, self.config.prefill_assistant)
                    for p in local
                ]
                outputs = self.llm.chat(messages=inputs, sampling_params=sp_list, use_tqdm=use_tqdm)
            else:
                inputs = [p["text"] for p in local]
                outputs = self.llm.generate(inputs, sampling_params=sp_list, use_tqdm=use_tqdm)

            local_results = [
                {"id": p["id"], "generated_text": o.outputs[0].text}
                for p, o in zip(local, outputs)
            ]

        return self._gather(local_results, dp_size)

    @staticmethod
    def _gather(local_results: list, dp_size: int) -> List[str]:
        if dp_size <= 1:
            local_results.sort(key=lambda x: x["id"])
            return [r["generated_text"] for r in local_results]
        try:
            import torch.distributed as dist
            from vllm.distributed.parallel_state import get_world_group
            cpu_group = get_world_group().cpu_group
            all_results: list = [None] * dist.get_world_size(group=cpu_group)
            dist.all_gather_object(all_results, local_results, group=cpu_group)
            merged = [r for rank in all_results for r in rank]
            merged.sort(key=lambda x: x["id"])
            return [r["generated_text"] for r in merged]
        except Exception:
            local_results.sort(key=lambda x: x["id"])
            return [r["generated_text"] for r in local_results]


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
        if self.config.prefill_assistant:
            return tok.apply_chat_template(
                messages, chat_template=self.config.chat_template,
                tokenize=False, continue_final_message=True,
            )
        return tok.apply_chat_template(
            messages, chat_template=self.config.chat_template,
            tokenize=False, add_generation_prompt=True,
        )

    def generate(self, prompts, **kwargs):
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
        return self._run_workers(preprocessed)

    def _run_workers(self, prompts):
        import torch
        from multiprocessing import get_context, Manager
        from tqdm import tqdm

        total_gpus = torch.cuda.device_count()
        needed = self.concurrency * self.n_devices
        if total_gpus < needed:
            raise RuntimeError(f"Need {needed} GPUs, only {total_gpus} available")

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

        results = [{"id": i, "generated_text": shared_out[i]} for i in range(len(prompts))]
        return [r["generated_text"] for r in results]


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
        self._processor = None
        self._ray = None

        if config.mode == "instruct":
            from transformers import AutoTokenizer
            tok_path = bk.get("tokenizer", config.model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(tok_path)
        else:
            self._tokenizer = None

    def _make_preprocess(self):
        cfg = self.config
        tok = self._tokenizer

        def preprocess(row):
            if cfg.mode == "instruct":
                msgs = build_chat_messages(row["text"], cfg.system_prompt, cfg.prefill_assistant)
                if cfg.prefill_assistant:
                    prompt = tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True)
                else:
                    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            else:
                prompt = row["text"]
            return {"prompt": prompt, "sampling_params": row["sampling_params"], "id": row["id"]}

        return preprocess

    def _ensure_processor(self):
        if self._processor is not None:
            return

        import ray
        from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        processor_cfg = vLLMEngineProcessorConfig(
            model_source=self.config.model_path,
            engine_kwargs=self.engine_kwargs,
            concurrency=self.concurrency,
            batch_size=self.batch_size,
            max_concurrent_batches=self.max_concurrent_batches,
            apply_chat_template=False,
        )
        self._processor = build_llm_processor(
            processor_cfg,
            preprocess=self._make_preprocess(),
            postprocess=lambda row: row,
        )
        self._ray = ray

    def generate(self, prompts, **kwargs):
        if not prompts:
            return []

        self._ensure_processor()
        normalized = self._normalize_prompts(prompts, self.sampling_params)
        ds = self._ray.data.from_items(normalized)
        results_ds = self._processor(ds)

        try:
            results = list(results_ds.take_all())
        except AttributeError:
            results = list(results_ds.iter_rows())

        results.sort(key=lambda x: x["id"])
        return [r["generated_text"] for r in results]
