from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from .formatting import apply_formatting

# ---------------------------------------------------------------------------
# Input format registry
# ---------------------------------------------------------------------------

INPUT_FORMATS: Dict[str, Callable[[str], List[Dict[str, Any]]]] = {}


def register_format(name: str):
    """Decorator to register a new input format loader."""
    def decorator(fn: Callable[[str], List[Dict[str, Any]]]):
        INPUT_FORMATS[name] = fn
        return fn
    return decorator


@register_format("jsonl")
def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@register_format("json")
def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_records(path: str, format: str = "jsonl") -> List[Dict[str, Any]]:
    if format not in INPUT_FORMATS:
        raise ValueError(f"Unknown format: {format}. Available: {list(INPUT_FORMATS.keys())}")
    return INPUT_FORMATS[format](path)


def load_records_multi(paths: List[str], format: str = "jsonl") -> tuple[List[Dict[str, Any]], List[int]]:
    """Load records from multiple files. Returns (all_records, file_sizes)."""
    all_records: List[Dict[str, Any]] = []
    sizes: List[int] = []
    for path in paths:
        records = load_records(path, format)
        all_records.extend(records)
        sizes.append(len(records))
    return all_records, sizes


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_jsonl(records: List[Dict[str, Any]], path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def decode_escaped(s: str) -> str:
    """Decode common escape sequences like \\n and \\t in a string."""
    if not s:
        return s
    return s.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")


def build_prompts(
    records: List[Dict[str, Any]],
    *,
    template: Optional[str] = None,
    key: Optional[str] = None,
    formatting: str = "none",
    max_key_length: Optional[int] = None,
) -> List[str]:
    """Build prompt strings from records using a template or key extraction."""
    if template is None and key is None:
        raise ValueError("Either template or key must be provided")

    prompts: List[str] = []
    for r in records:
        if template:
            values = {k: str(v)[:max_key_length] for k, v in r.items()} if max_key_length else r
            prompts.append(template.format(**values))
        else:
            prompts.append(str(r[key]))

    if formatting and formatting != "none":
        prompts = [apply_formatting(p, formatting) for p in prompts]

    return prompts
