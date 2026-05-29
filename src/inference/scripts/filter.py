#!/usr/bin/env python3
"""LLM-judge JSONL filter CLI.

Reads ``--input-path``, judges each record's ``--key`` field against a
filter system prompt (either supplied directly via ``--system-prompt`` /
``--system-prompt-file``, or built from the project-default template via
``--language``), and writes the surviving records to ``--output-path``.

The judge model is held to a strict KEEP / REJECT contract: outputs that
match neither are dropped by default (``--ambiguous-keep`` flips this).

Usage examples
==============

# Filter Spanish responses with the project-default 3-rule prompt:
python -m inference.scripts.filter \\
    --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \\
    --backend vllm \\
    --backend-kwargs '{"tensor_parallel_size": 8, "gpu_memory_utilization": 0.9}' \\
    --input-path responses_es.jsonl \\
    --output-path responses_es_kept.jsonl \\
    --rejected-path responses_es_dropped.jsonl \\
    --judgment-key judgment \\
    --key response \\
    --language Spanish

# Use a custom system prompt verbatim (overrides --language):
python -m inference.scripts.filter ... \\
    --system-prompt-file my_prompt.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

from inference.filter import (
    DecisionRules,
    build_filter_prompt,
    run_filter,
)


def _parse_json(arg: Optional[str]) -> Dict[str, Any]:
    if not arg:
        return {}
    return json.loads(arg)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- I/O ---
    p.add_argument("--input-path", required=True)
    p.add_argument("--output-path", required=True,
                   help="Where to write the kept records.")
    p.add_argument("--rejected-path", default=None,
                   help="Optional path to also dump rejected records to.")
    p.add_argument("--input-format", default="jsonl", choices=["jsonl", "json"])
    p.add_argument("--key", required=True,
                   help="Record field whose value is judged.")
    p.add_argument("--judgment-key", default=None,
                   help="If set, attach the raw judge output to each "
                        "kept/rejected record under this field.")

    # --- Prompt ---
    g = p.add_argument_group("filter prompt (one of these is required)")
    g.add_argument("--language", default=None,
                   help="Render the project-default 3-rule prompt for this language.")
    g.add_argument("--system-prompt", default=None,
                   help="Use this string verbatim as the judge's system prompt.")
    g.add_argument("--system-prompt-file", default=None,
                   help="Read the judge's system prompt from this file.")
    g.add_argument("--prompt-template", default=None,
                   help="Custom Python format string with a ``{language}`` "
                        "placeholder; combined with --language.")

    # --- Decision parsing ---
    p.add_argument("--keep-pattern", default=r"\bKEEP\b",
                   help="Case-insensitive regex; outputs matching this count as KEEP.")
    p.add_argument("--reject-pattern", default=r"\bREJECT\b",
                   help="Case-insensitive regex; outputs matching this count as REJECT.")
    p.add_argument("--ambiguous-keep", action="store_true",
                   help="If a judge output matches neither pattern, KEEP it. "
                        "Default is the strict 'reject on ambiguity'.")

    # --- Model ---
    p.add_argument("--model-path", required=True)
    p.add_argument("--backend", default="vllm")
    p.add_argument("--backend-kwargs", default=None,
                   help='JSON dict, e.g. \'{"tensor_parallel_size": 8}\'')
    p.add_argument("--sampling-params", default=None,
                   help='JSON dict; defaults to {"temperature": 0.0, "max_tokens": 8}')

    return p.parse_args()


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    sources = sum(
        1 for x in (args.language, args.system_prompt, args.system_prompt_file) if x
    )
    if sources == 0:
        raise SystemExit(
            "error: one of --language, --system-prompt, --system-prompt-file is required",
        )
    if args.system_prompt:
        return args.system_prompt
    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    # --language
    return build_filter_prompt(args.language, template=args.prompt_template)


def main() -> None:
    args = _parse_args()
    system_prompt = _resolve_system_prompt(args)

    summary = run_filter(
        input_path=args.input_path,
        output_path=args.output_path,
        rejected_path=args.rejected_path,
        input_format=args.input_format,
        key=args.key,
        judgment_key=args.judgment_key,
        system_prompt=system_prompt,
        model_path=args.model_path,
        backend=args.backend,
        backend_kwargs=_parse_json(args.backend_kwargs),
        sampling_params=_parse_json(args.sampling_params),
        rules=DecisionRules(
            keep_pattern=args.keep_pattern,
            reject_pattern=args.reject_pattern,
            ambiguous_keep=args.ambiguous_keep,
        ),
    )

    pct = (
        100.0 * summary["kept"] / summary["total"] if summary["total"] else 0.0
    )
    print(
        f"[filter] kept {summary['kept']:,} / {summary['total']:,} "
        f"({pct:.1f}%) -> {args.output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
