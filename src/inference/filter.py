"""LLM-judge data filter.

Generic primitive for keeping or rejecting JSONL records based on a one-
token verdict produced by a language model. The model is loaded once,
the entire input is judged in one batched ``llm.chat`` call, and the
input record order is preserved in the output.

Library entry points
====================

* :func:`build_filter_prompt` — render the canonical filter system
  prompt for a target language. Encodes the three project-wide rules
  (no code, substantive prose, single language).
* :func:`parse_decision` — turn a judge's raw text into a ``KEEP`` /
  ``REJECT`` boolean. Defaults to "strict": ambiguous outputs reject.
* :func:`filter_texts` — judge a list of strings and return a parallel
  list of keep-flags + the raw verdicts.
* :func:`run_filter` — full JSONL-in / JSONL-out pipeline. Used by both
  the CLI script in :mod:`inference.scripts.filter` and downstream
  callers (e.g. flexibility's alpaca filter).

Wire formatting
===============

Records are JSONL dicts. The text-to-judge is extracted by ``key`` (a
record field) — e.g. ``key="response"`` for the cg pipeline's response
JSONLs. If you need to judge a nested or composite field, do the
extraction yourself and call :func:`filter_texts` directly.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .generate import GenerationConfig, GenerationModel
from .io import load_records, save_jsonl


# ---------------------------------------------------------------------------
# Default filter prompt
# ---------------------------------------------------------------------------

# The system prompt below is the project-wide default for filtering
# multilingual SFT responses. It encodes three rules — no code, real
# prose (not just data), single language. Tweaks should usually be
# additive: keep the structure, append additional rules, and keep the
# strict KEEP / REJECT contract.

FILTER_PROMPT_TEMPLATE = """\
You are a strict data-quality filter for a multilingual instruction-tuning corpus. \
Your task: for each candidate response, decide whether it should be KEPT in the \
training set or REJECTED.

Apply ALL THREE rules below. If the response fails any single rule, REJECT.

────────────────────────────────────────────────────────────────────
RULE 1 — No code.
────────────────────────────────────────────────────────────────────
The response must contain no source code or pseudocode in any programming \
language. This includes, in any quantity:
  • Code fences (``` … ``` or indented code blocks).
  • Identifiable code statements: function or class definitions, control \
flow, imports, variable declarations, SQL, shell commands, regular \
expressions, etc.
  • Structured-data literals presented as code: JSON, YAML, XML, HTML tags.

Inline mentions of code-related concepts in prose are fine — e.g. \
"use Python's ``print`` function" is acceptable, but an actual \
``print(...)`` example is not.

────────────────────────────────────────────────────────────────────
RULE 2 — Substantive natural-language prose.
────────────────────────────────────────────────────────────────────
The response must contain meaningful natural-language explanation, not just \
bare data. REJECT if the response consists ENTIRELY of:
  • A bare number, date, or short numeric expression.
  • A table, chart, or otherwise structured tabular data with no narration.
  • A bulleted or numbered list of bare items with no explanatory sentences.
  • A single short phrase, fragment, or one-word answer.

A response that mixes narration with a list, table, or example is acceptable \
as long as the narration carries the meaning.

────────────────────────────────────────────────────────────────────
RULE 3 — Single language: {language}.
────────────────────────────────────────────────────────────────────
All natural-language text in the response must be written in {language}. \
REJECT if any non-{language} natural-language text appears, including \
isolated greetings, untranslated source-language words, or stray phrases in \
another language.

Permitted exceptions: universally-recognized proper nouns (people, places, \
brands), standard mathematical or scientific notation, and ISO terminology.

────────────────────────────────────────────────────────────────────
Output format
────────────────────────────────────────────────────────────────────
Output exactly one token:
  • KEEP   — if the response satisfies ALL three rules.
  • REJECT — if it fails any one rule.

No explanation. No punctuation. No quotes. No additional text.\
"""


def build_filter_prompt(language: str, *, template: Optional[str] = None) -> str:
    """Render the filter system prompt for ``language``.

    Pass a custom ``template`` (with the literal ``{language}`` placeholder)
    to override the default project-wide prompt.
    """
    return (template or FILTER_PROMPT_TEMPLATE).format(language=language)


# ---------------------------------------------------------------------------
# Decision parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionRules:
    """How to turn raw judge output text into a KEEP/REJECT boolean.

    Both patterns are anchored as case-insensitive whole-word matches.
    Any output that doesn't match either is treated according to
    ``ambiguous_keep`` — by default, ambiguous outputs are rejected
    (the strict, conservative posture).
    """
    keep_pattern: str = r"\bKEEP\b"
    reject_pattern: str = r"\bREJECT\b"
    ambiguous_keep: bool = False

    def compile(self) -> Tuple[re.Pattern, re.Pattern]:
        return (
            re.compile(self.keep_pattern, re.IGNORECASE),
            re.compile(self.reject_pattern, re.IGNORECASE),
        )


def parse_decision(text: str, rules: Optional[DecisionRules] = None) -> bool:
    """Return ``True`` if ``text`` is a KEEP verdict, else ``False``."""
    rules = rules or DecisionRules()
    keep_re, reject_re = rules.compile()
    s = (text or "").strip()
    if reject_re.search(s):
        return False
    if keep_re.search(s):
        return True
    return rules.ambiguous_keep


# ---------------------------------------------------------------------------
# Core filter primitive
# ---------------------------------------------------------------------------

def filter_texts(
    texts: Sequence[str],
    *,
    system_prompt: str,
    model_path: str,
    backend: str = "vllm",
    backend_kwargs: Optional[Dict[str, Any]] = None,
    sampling_params: Optional[Dict[str, Any]] = None,
    rules: Optional[DecisionRules] = None,
    model: Optional[GenerationModel] = None,
) -> Tuple[List[bool], List[str]]:
    """Judge each text and return ``(keep_mask, raw_judgments)``.

    Pass an already-built ``model`` to reuse a previously-loaded vLLM
    instance across multiple filter passes (e.g. when filtering several
    JSONL files in a single Python process).
    """
    if not texts:
        return [], []

    if model is None:
        sp = {"temperature": 0.0, "max_tokens": 8, "top_p": 1.0}
        if sampling_params:
            sp.update(sampling_params)
        config = GenerationConfig(
            model_path=model_path,
            backend=backend,
            mode="instruct",
            system_prompt=system_prompt,
            sampling_params=sp,
            backend_kwargs=dict(backend_kwargs or {}),
        )
        model = GenerationModel.from_config(config)
    else:
        # If reusing a model, the caller already pinned its system prompt.
        # We don't override it here; ``texts`` is judged under whatever
        # system prompt the model was built with.
        pass

    judgments = model.generate(list(texts))
    keep_mask = [parse_decision(j, rules) for j in judgments]
    return keep_mask, judgments


# ---------------------------------------------------------------------------
# JSONL pipeline
# ---------------------------------------------------------------------------

def run_filter(
    *,
    input_path: str,
    output_path: str,
    key: str,
    system_prompt: str,
    model_path: str,
    backend: str = "vllm",
    backend_kwargs: Optional[Dict[str, Any]] = None,
    sampling_params: Optional[Dict[str, Any]] = None,
    rules: Optional[DecisionRules] = None,
    judgment_key: Optional[str] = None,
    rejected_path: Optional[str] = None,
    input_format: str = "jsonl",
) -> Dict[str, int]:
    """Filter a JSONL file via an LLM judge.

    Reads ``input_path``, judges each record's ``record[key]`` field
    against ``system_prompt``, writes the kept records to ``output_path``
    and (optionally) the rejected records to ``rejected_path``.

    If ``judgment_key`` is set, every kept (and rejected) record gets an
    additional field with the raw judge output — useful for spot-checking
    later.

    Returns a small summary dict ``{"kept", "rejected", "total"}``.
    """
    records = load_records(input_path, format=input_format)
    if not records:
        save_jsonl([], output_path)
        return {"kept": 0, "rejected": 0, "total": 0}

    texts: List[str] = []
    for i, rec in enumerate(records):
        if key not in rec:
            raise KeyError(
                f"record {i} of {input_path!r} has no field {key!r} "
                f"(available: {list(rec.keys())[:8]}...)"
            )
        v = rec[key]
        if not isinstance(v, str):
            v = "" if v is None else str(v)
        texts.append(v)

    keep_mask, judgments = filter_texts(
        texts,
        system_prompt=system_prompt,
        model_path=model_path,
        backend=backend,
        backend_kwargs=backend_kwargs,
        sampling_params=sampling_params,
        rules=rules,
    )

    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for rec, keep, judgment in zip(records, keep_mask, judgments):
        out = dict(rec)
        if judgment_key:
            out[judgment_key] = judgment
        (kept if keep else rejected).append(out)

    save_jsonl(kept, output_path)
    if rejected_path:
        save_jsonl(rejected, rejected_path)

    return {"kept": len(kept), "rejected": len(rejected), "total": len(records)}
