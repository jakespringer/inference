#!/usr/bin/env python3
"""Generate text from file or single prompt.

Usage examples:

    # From a config file
    python -m inference.scripts.generate --config config.json

    # Inline (vLLM, file input)
    python -m inference.scripts.generate \
        --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
        --backend vllm \
        --input-path data.jsonl \
        --output-path output.jsonl \
        --template "Answer the question: {question}" \
        --sampling-params '{"max_tokens": 512, "temperature": 0.7}' \
        --backend-kwargs '{"tensor_parallel_size": 4}'

    # Single prompt, multiple samples
    python -m inference.scripts.generate \
        --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
        --prompt "Write a haiku about coding" \
        --n-samples 10 \
        --output-path haikus.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np

from inference import (
    GenerationConfig,
    GenerationModel,
    apply_formatting,
    build_prompts,
    decode_escaped,
    load_records,
    load_records_multi,
    save_jsonl,
)


def _parse_json_arg(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    return json.loads(s)


def _set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _build_config_from_args(args, file_cfg: Dict[str, Any]) -> GenerationConfig:
    """Merge CLI args over file config to build GenerationConfig."""
    model_cfg = file_cfg.get("model", {})

    def pick(cli_val, cfg_key, default=None):
        if cli_val is not None:
            return cli_val
        return model_cfg.get(cfg_key, file_cfg.get(cfg_key, default))

    sp_cli = _parse_json_arg(args.sampling_params)
    sp_file = model_cfg.get("sampling_params", file_cfg.get("sampling_params", {}))
    sp = {**sp_file, **sp_cli}

    bk_cli = _parse_json_arg(args.backend_kwargs)
    bk_file = model_cfg.get("backend_kwargs", file_cfg.get("backend_kwargs", {}))
    bk = {**bk_file, **bk_cli}

    return GenerationConfig(
        model_path=pick(args.model_path, "model_path"),
        backend=pick(args.backend, "backend", "vllm"),
        mode=pick(args.mode, "mode", "instruct"),
        system_prompt=pick(args.system_prompt, "system_prompt"),
        prefill_assistant=pick(args.prefill_assistant, "prefill_assistant"),
        chat_template=pick(args.chat_template, "chat_template"),
        sampling_params=sp,
        backend_kwargs=bk,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate text from prompts")

    # Config
    parser.add_argument("--config", type=str, help="JSON config file (all other args override it)")

    # Input
    parser.add_argument("--input-path", type=str, nargs="+", help="Input JSONL file(s)")
    parser.add_argument("--input-format", type=str, default="jsonl")
    parser.add_argument("--prompt", type=str, help="Single prompt string (alternative to --input-path)")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of samples per prompt")

    # Output
    parser.add_argument("--output-path", type=str, nargs="+", help="Output JSONL file(s)")
    parser.add_argument("--output-key", type=str, default="output")
    parser.add_argument("--include-generation", action="store_true",
                        help="Include raw generation before postprocessing")

    # Model
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--system-prompt", type=str)
    parser.add_argument("--prefill-assistant", type=str)
    parser.add_argument("--chat-template", type=str)
    parser.add_argument("--sampling-params", type=str, help="JSON dict of sampling params")
    parser.add_argument("--backend-kwargs", type=str, help="JSON dict of backend kwargs")

    # Processing
    parser.add_argument("--template", type=str, help="Prompt template f-string, e.g. 'Answer: {question}'")
    parser.add_argument("--key", type=str, help="Record key to use as prompt text")
    parser.add_argument("--preprocessing", type=str, default="none")
    parser.add_argument("--postprocessing", type=str, default="none")
    parser.add_argument("--max-key-length", type=int)

    # Other
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    # Load file config
    file_cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config) as f:
            file_cfg = json.load(f)

    # Resolve remaining options from file config
    input_paths: Optional[List[str]] = args.input_path or file_cfg.get("input_path")
    prompt: Optional[str] = args.prompt or file_cfg.get("prompt")
    output_paths: List[str] = args.output_path or file_cfg.get("output_path", [])
    output_key: str = args.output_key if args.output_key != "output" else file_cfg.get("output_key", "output")
    template: Optional[str] = args.template or file_cfg.get("template")
    key: Optional[str] = args.key or file_cfg.get("key")
    preprocessing: str = args.preprocessing if args.preprocessing != "none" else file_cfg.get("preprocessing", "none")
    postprocessing: str = args.postprocessing if args.postprocessing != "none" else file_cfg.get("postprocessing", "none")
    max_key_length: Optional[int] = args.max_key_length or file_cfg.get("max_key_length")
    n_samples: int = args.n_samples if args.n_samples != 1 else file_cfg.get("n_samples", 1)
    seed: Optional[int] = args.seed if args.seed is not None else file_cfg.get("seed")

    if not output_paths:
        parser.error("--output-path is required (via CLI or config file)")

    if template:
        template = decode_escaped(template)

    _set_seed(seed)

    # Build model config and create model
    gen_config = _build_config_from_args(args, file_cfg)
    if gen_config.model_path is None:
        parser.error("--model-path is required (via CLI or config file)")

    model = GenerationModel.from_config(gen_config)

    # --- Single prompt mode ---
    if prompt is not None:
        if isinstance(output_paths, list) and len(output_paths) > 0:
            out_path = output_paths[0]
        else:
            out_path = output_paths

        prompts_list = [prompt] * n_samples
        outputs = model.generate(prompts_list)
        processed = [apply_formatting(o, postprocessing) for o in outputs]

        records = []
        for i, (raw, proc) in enumerate(zip(outputs, processed)):
            rec: Dict[str, Any] = {"prompt": prompt, output_key: proc, "sample_index": i}
            if args.include_generation:
                rec["generation"] = raw
            records.append(rec)

        save_jsonl(records, out_path)
        print(f"Wrote {len(records)} records to {out_path}")
        return

    # --- File input mode ---
    if input_paths is None:
        parser.error("Either --input-path or --prompt is required")

    if isinstance(input_paths, str):
        input_paths = [input_paths]
    if isinstance(output_paths, str):
        output_paths = [output_paths]

    if len(input_paths) != len(output_paths):
        if len(output_paths) == 1:
            base, ext = os.path.splitext(output_paths[0])
            output_paths = [f"{base}_{i}{ext}" for i in range(len(input_paths))]
        else:
            parser.error(f"Number of input paths ({len(input_paths)}) must match output paths ({len(output_paths)})")

    all_records, file_sizes = load_records_multi(input_paths, format=args.input_format)

    if template or key:
        prompts_list = build_prompts(
            all_records, template=template, key=key,
            formatting=preprocessing, max_key_length=max_key_length,
        )
    else:
        parser.error("Either --template or --key is required for file input")

    # Expand for n_samples > 1
    if n_samples > 1:
        expanded_records = []
        expanded_prompts = []
        for rec, p in zip(all_records, prompts_list):
            for _ in range(n_samples):
                expanded_records.append(rec)
                expanded_prompts.append(p)
        all_records = expanded_records
        prompts_list = expanded_prompts
        file_sizes = [s * n_samples for s in file_sizes]

    outputs = model.generate(prompts_list)
    processed = [apply_formatting(o, postprocessing) for o in outputs]

    # Distribute results back to per-file records
    offset = 0
    for out_path, size in zip(output_paths, file_sizes):
        file_records = []
        for i in range(size):
            rec = all_records[offset + i].copy()
            rec[output_key] = processed[offset + i]
            if args.include_generation:
                rec["generation"] = outputs[offset + i]
            file_records.append(rec)
        save_jsonl(file_records, out_path)
        print(f"Wrote {len(file_records)} records to {out_path}")
        offset += size


if __name__ == "__main__":
    main()
