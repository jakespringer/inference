#!/usr/bin/env python3
"""Build a sparse JSON vocab mask file for inference.

Composes a mask from a tokenizer plus one or more additive sources, then
writes the sorted, deduplicated token IDs to a JSON file consumed by
``GenerationConfig.output_vocab_mask`` (and by ``--vocab-mask`` in the
chat CLI).

Output schema::

    {"vocab_size": <int>, "mode": "allow" | "deny", "tokens": [<int>, ...]}

Under ``--mode allow`` the listed tokens are the ONLY ones the sampler
may choose. Under ``--mode deny`` the listed tokens are forbidden and
everything else is permitted. The ``--allow-*`` source flags name
membership in the listed set under either mode — they always say "put
these tokens in the list"; ``--mode`` decides what the list means.

Example: digits-only generation for Qwen3-1.7B::

    python -m inference.scripts.build_vocab_mask \\
        --tokenizer Qwen/Qwen3-1.7B \\
        --mode allow \\
        --allow-chars "0123456789" \\
        --allow-eos \\
        --output digits.json

Example: forbid a specific token + all special tokens::

    python -m inference.scripts.build_vocab_mask \\
        --tokenizer Qwen/Qwen3-1.7B \\
        --mode deny \\
        --allow-token-ids 42 \\
        --allow-special \\
        --output no_specials.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List, Set, Tuple


def _ids_from_chars(tokenizer, chars: str) -> List[int]:
    """Tokenize each character individually; collect ALL resulting IDs.

    Multi-byte characters may tokenize to multiple sub-tokens — we
    include every sub-token. This matches the intuitive reading of
    "allow these characters": any token that participates in producing
    one of these characters is allowed.
    """
    ids: List[int] = []
    for ch in chars:
        ids.extend(tokenizer.encode(ch, add_special_tokens=False))
    return ids


def _ids_from_strings(tokenizer, strings: Iterable[str]) -> List[int]:
    """Tokenize each string as a unit; collect all IDs across strings."""
    ids: List[int] = []
    for s in strings:
        ids.extend(tokenizer.encode(s, add_special_tokens=False))
    return ids


def _ids_from_mask_file(path: str) -> Tuple[List[int], int]:
    """Load token IDs + vocab_size from an existing mask file.

    Dense masks are converted to a sparse ID list on the fly. Returns
    the IDs that are SET in the underlying bool tensor — i.e. the
    indices listed under whatever mode the source file uses; callers
    are responsible for matching modes.
    """
    from inference.vocab_mask import VocabMaskSpec
    spec = VocabMaskSpec.from_file(path)
    vocab_size = spec.vocab_size
    if vocab_size is None:
        raise ValueError(f"--from-mask source {path!r} carries no vocab_size")
    if spec.tokens is not None:
        return list(spec.tokens), vocab_size
    import torch
    indices = torch.nonzero(spec.dense.to(torch.bool), as_tuple=True)[0]
    return indices.tolist(), vocab_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tokenizer", required=True,
        help="HF id or local path of the tokenizer to encode against.",
    )
    parser.add_argument(
        "--mode", required=True, choices=("allow", "deny"),
        help="Polarity of the output mask: 'allow' = listed tokens are "
             "the only ones permitted; 'deny' = listed tokens are forbidden, "
             "everything else permitted.",
    )
    parser.add_argument(
        "--allow-chars", default="",
        help="String of characters; each character is tokenized individually "
             "and all resulting sub-token IDs are added to the list.",
    )
    parser.add_argument(
        "--allow-strings", nargs="*", default=[],
        help="Strings to tokenize whole (without special tokens); all "
             "resulting IDs are added to the list.",
    )
    parser.add_argument(
        "--allow-token-ids", nargs="*", type=int, default=[],
        help="Raw token IDs to add to the list.",
    )
    parser.add_argument(
        "--allow-special", action="store_true",
        help="Add all of tokenizer.all_special_ids to the list.",
    )
    parser.add_argument(
        "--allow-eos", action="store_true",
        help="Add tokenizer.eos_token_id to the list.",
    )
    parser.add_argument(
        "--from-mask", type=str, default=None,
        help="Seed the list from an existing mask file (extends it). "
             "The source file's vocab_size must match the tokenizer's.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON path. Must end in '.json'.",
    )
    args = parser.parse_args()

    if not args.output.endswith(".json"):
        parser.error("--output must end in .json")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab_size = len(tokenizer)

    collected: Set[int] = set()

    if args.from_mask:
        seed_ids, seed_vocab = _ids_from_mask_file(args.from_mask)
        if seed_vocab != vocab_size:
            parser.error(
                f"--from-mask {args.from_mask!r} declares vocab_size="
                f"{seed_vocab}, but the tokenizer reports vocab_size="
                f"{vocab_size}"
            )
        collected.update(seed_ids)

    if args.allow_chars:
        collected.update(_ids_from_chars(tokenizer, args.allow_chars))

    if args.allow_strings:
        collected.update(_ids_from_strings(tokenizer, args.allow_strings))

    if args.allow_token_ids:
        for tid in args.allow_token_ids:
            if not 0 <= tid < vocab_size:
                parser.error(
                    f"--allow-token-ids: id {tid} outside "
                    f"[0, vocab_size={vocab_size})"
                )
        collected.update(args.allow_token_ids)

    if args.allow_special:
        collected.update(getattr(tokenizer, "all_special_ids", None) or [])

    if args.allow_eos:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            parser.error("--allow-eos: tokenizer has no eos_token_id")
        collected.add(eos)

    if not collected:
        parser.error(
            "no token IDs collected; pass at least one of --allow-chars / "
            "--allow-strings / --allow-token-ids / --allow-special / "
            "--allow-eos / --from-mask"
        )

    for tid in collected:
        if not 0 <= tid < vocab_size:
            parser.error(
                f"collected token id {tid} is outside [0, vocab_size="
                f"{vocab_size}) — tokenizer/vocab_size mismatch"
            )

    tokens_sorted = sorted(collected)
    spec = {
        "vocab_size": vocab_size,
        "mode": args.mode,
        "tokens": tokens_sorted,
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(spec, f, indent=2)

    pct = 100.0 * len(tokens_sorted) / vocab_size
    print(
        f"Wrote {args.output}: mode={args.mode}, "
        f"{len(tokens_sorted)} / {vocab_size} tokens listed ({pct:.2f}%)"
    )


if __name__ == "__main__":
    main()
