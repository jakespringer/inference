#!/usr/bin/env python3
"""Compute embeddings from a file of texts.

Usage examples:

    # From a config file
    python -m inference.scripts.embed --config config.json

    # Inline
    python -m inference.scripts.embed \
        --model-path sentence-transformers/all-MiniLM-L6-v2 \
        --input-path docs.jsonl \
        --key text \
        --output-path embeddings.npy

    # With instruction and layer extraction
    python -m inference.scripts.embed \
        --model-path intfloat/e5-large-v2 \
        --input-path queries.jsonl \
        --key query \
        --instruction "Retrieve relevant passages" \
        --layer 12 \
        --pooling mean \
        --output-path query_embeddings.npy

    # Sharded output (directory)
    python -m inference.scripts.embed \
        --model-path sentence-transformers/all-MiniLM-L6-v2 \
        --input-path docs.jsonl \
        --key text \
        --output-path embeddings/ \
        --shard-size-mb 200
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from inference import (
    EmbeddingConfig,
    EmbeddingModel,
    build_prompts,
    load_records,
    save_jsonl,
)


def _parse_json_arg(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    return json.loads(s)


def _save_sharded(embeddings: np.ndarray, output_dir: str, shard_size_mb: int):
    """Save embeddings as sharded numpy memmap files with metadata."""
    os.makedirs(output_dir, exist_ok=True)

    n, dim = embeddings.shape
    bytes_per_row = dim * embeddings.dtype.itemsize
    rows_per_shard = max(1, (shard_size_mb * 1024 * 1024) // bytes_per_row)

    shard_sizes: List[int] = []
    shard_idx = 0
    offset = 0

    while offset < n:
        end = min(offset + rows_per_shard, n)
        chunk = embeddings[offset:end]
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.bin")
        fp = np.memmap(shard_path, dtype=embeddings.dtype, mode="w+", shape=chunk.shape)
        fp[:] = chunk
        fp.flush()
        del fp
        shard_sizes.append(end - offset)
        shard_idx += 1
        offset = end

    metadata = {
        "embedding_dim": dim,
        "dtype": str(embeddings.dtype),
        "n_embeddings": n,
        "n_shards": shard_idx,
        "shard_sizes": shard_sizes,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {n} embeddings in {shard_idx} shards to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compute embeddings from text")

    # Config
    parser.add_argument("--config", type=str, help="JSON config file")

    # Input
    parser.add_argument("--input-path", type=str, help="Input JSONL file")
    parser.add_argument("--input-format", type=str, default="jsonl")
    parser.add_argument("--key", type=str, help="Record key for text field")
    parser.add_argument("--template", type=str, help="Template for text (alternative to --key)")
    parser.add_argument("--preprocessing", type=str, default="none")

    # Output
    parser.add_argument("--output-path", type=str,
                        help="Output path: .npy file, .jsonl file, or directory for sharded output")
    parser.add_argument("--shard-size-mb", type=int, default=200, help="Shard size in MB (for directory output)")

    # Model
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--pooling", type=str, default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--n-devices-per-instance", type=int, default=None)
    parser.add_argument("--model-kwargs", type=str, help="JSON dict of model kwargs")

    args = parser.parse_args()

    # Load file config
    file_cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config) as f:
            file_cfg = json.load(f)
    model_cfg = file_cfg.get("model", {})

    def pick(cli_val, cfg_key, default=None):
        if cli_val is not None:
            return cli_val
        return model_cfg.get(cfg_key, file_cfg.get(cfg_key, default))

    mk_cli = _parse_json_arg(args.model_kwargs)
    mk_file = model_cfg.get("model_kwargs", {})
    model_kwargs = {**mk_file, **mk_cli}

    embed_config = EmbeddingConfig(
        model_path=pick(args.model_path, "model_path"),
        pooling=pick(args.pooling, "pooling", "mean"),
        normalize=pick(args.normalize, "normalize", True),
        layer=pick(args.layer, "layer"),
        batch_size=pick(args.batch_size, "batch_size", 32),
        max_length=pick(args.max_length, "max_length"),
        instruction=pick(args.instruction, "instruction"),
        concurrency=pick(args.concurrency, "concurrency", 1),
        n_devices_per_instance=pick(args.n_devices_per_instance, "n_devices_per_instance", 1),
        model_kwargs=model_kwargs,
    )

    if embed_config.model_path is None:
        parser.error("--model-path is required (via CLI or config file)")

    # Resolve I/O options
    input_path: str = args.input_path or file_cfg.get("input_path")
    output_path: str = args.output_path or file_cfg.get("output_path")
    key: Optional[str] = args.key or file_cfg.get("key")
    template: Optional[str] = args.template or file_cfg.get("template")
    preprocessing: str = args.preprocessing if args.preprocessing != "none" else file_cfg.get("preprocessing", "none")

    if output_path is None:
        parser.error("--output-path is required (via CLI or config file)")
    if input_path is None:
        parser.error("--input-path is required")

    # Load and prepare texts
    records = load_records(input_path, format=args.input_format)

    if template or key:
        texts = build_prompts(records, template=template, key=key, formatting=preprocessing)
    else:
        parser.error("Either --key or --template is required")

    # Embed
    model = EmbeddingModel.from_config(embed_config)
    embeddings = model.embed(texts)

    # Save output
    if output_path.endswith(".npy"):
        parent = os.path.dirname(os.path.abspath(output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.save(output_path, embeddings)
        print(f"Saved {embeddings.shape} embeddings to {output_path}")

    elif output_path.endswith(".jsonl"):
        out_records = []
        for rec, emb in zip(records, embeddings):
            rec = rec.copy()
            rec["embedding"] = emb.tolist()
            out_records.append(rec)
        save_jsonl(out_records, output_path)
        print(f"Saved {len(out_records)} records to {output_path}")

    else:
        _save_sharded(embeddings, output_path, shard_size_mb=args.shard_size_mb)


if __name__ == "__main__":
    main()
