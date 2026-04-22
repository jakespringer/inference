from __future__ import annotations
from typing import List


def _split_escaped(s: str, delim: str) -> List[str]:
    if not s:
        return []
    parts: List[str] = []
    buf: List[str] = []
    in_quotes = False
    escape = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if escape:
            buf.append(ch)
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            in_quotes = not in_quotes
            buf.append(ch)
        elif not in_quotes and ch == delim:
            if i + 1 < n and s[i + 1] == delim:
                buf.append(delim)
                i += 1
            else:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        part = "".join(buf).strip()
        if part:
            parts.append(part)
    return parts


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    out: List[str] = []
    escape = False
    for ch in s:
        if escape:
            out.append(ch)
            escape = False
        elif ch == "\\":
            escape = True
        else:
            out.append(ch)
    result = "".join(out)
    return result.replace("::", ":").replace(",,", ",")


def apply_formatting(text: str, spec: str) -> str:
    """Apply comma-separated formatting operations to text.

    Supported: first_line, after:{pat}, before:{pat}, between:{pat1}:{pat2}, none.
    Strings may be double-quoted; use ``,,`` for literal comma, ``::`` for literal colon.
    """
    if not spec or spec.strip().lower() in ("", "none"):
        return text

    ops = _split_escaped(spec, ",")
    result = text
    for raw_op in ops:
        op = raw_op.strip()
        if not op or op.lower() == "none":
            continue
        lower = op.lower()

        if lower == "first_line":
            for line in result.split("\n"):
                if line.strip():
                    result = line.strip()
                    break
            continue

        if lower.startswith("after:"):
            pattern = _unquote(op[len("after:"):])
            idx = result.rfind(pattern)
            if idx != -1:
                result = result[idx + len(pattern):]
            continue

        if lower.startswith("before:"):
            pattern = _unquote(op[len("before:"):])
            idx = result.find(pattern)
            if idx != -1:
                result = result[:idx]
            continue

        if lower.startswith("between:"):
            parts = _split_escaped(op[len("between:"):], ":")
            if len(parts) != 2:
                raise ValueError(f"between expects two arguments, got {len(parts)} in '{raw_op}'")
            start_pat = _unquote(parts[0])
            end_pat = _unquote(parts[1])
            si = result.find(start_pat)
            ei = result.rfind(end_pat)
            if si != -1 and ei != -1 and ei >= si + len(start_pat):
                result = result[si + len(start_pat):ei]
            continue

        raise ValueError(f"Unknown formatting operation: {raw_op}")
    return result
