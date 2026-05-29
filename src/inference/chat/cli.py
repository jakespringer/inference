#!/usr/bin/env python3
"""Streaming CLI chatbot backed by vLLM.

Generation runs in a separate process (a vLLM AsyncLLMEngine), while this
process owns the terminal: the raw-mode line editor and the token printing.
The two communicate over multiprocessing queues, so the editor/streaming
never share a GIL with the engine.

Usage:
    python -m inference.chat.cli --model Qwen/Qwen3-1.7B

Slash commands:
    /exit                  quit
    /reset                 clear chat history (keeps system prompt)
    /system [message]      set (no arg = clear) the system prompt
    /temperature [value]   show or set sampling temperature (0 = greedy)
    /top_p [value]         show or set top_p
    /max_tokens [n]        show or set max_new_tokens
    /thinking [bool]       show or toggle the model's thinking mode (if any)
    /sampling              print all current sampling parameters
    /history               print the current chat history
    /help                  show command list

Ctrl+C during generation interrupts the current reply. At an empty
prompt, a single Ctrl+C arms exit; a second Ctrl+C quits.

Enter submits the prompt. Ctrl+Enter (or Alt+Enter / Shift+Enter on
terminals that don't encode Ctrl+Enter) inserts a newline so you can
compose multi-line prompts; continuation lines are indented to line up
under the text of the first line. Backspace, Delete, arrow keys, Home/End
and Up/Down history all work across the multi-line buffer.
"""
from __future__ import annotations

import argparse
import atexit
import multiprocessing as mp
import os
import re
import sys
import time

try:
    import readline  # noqa: F401 — enables Up/Down history + line editing in input()
except ImportError:
    pass


# --- ANSI helpers ---------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(code: str) -> str:
    return code if _USE_COLOR else ""


RESET = _c("\x1b[0m")
DIM = _c("\x1b[2m")
BOLD = _c("\x1b[1m")
CYAN = _c("\x1b[36m")
GREEN = _c("\x1b[32m")
YELLOW = _c("\x1b[33m")
RED = _c("\x1b[31m")
MAGENTA = _c("\x1b[35m")


def _readline_safe(s: str) -> str:
    """Wrap ANSI escapes in \\001..\\002 so readline counts cursor cols
    correctly when computing line length for the input prompt."""
    if not _USE_COLOR or not s:
        return s
    out = []
    i = 0
    while i < len(s):
        if s[i] == "\x1b":
            j = s.find("m", i)
            if j != -1:
                out.append("\001" + s[i : j + 1] + "\002")
                i = j + 1
                continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _print_banner(model_name: str) -> None:
    bar = DIM + "─" * 60 + RESET
    print(bar)
    print(f"  {BOLD}{MAGENTA}chat{RESET}  {DIM}·{RESET}  {model_name}")
    print(f"  {DIM}/help for commands · Ctrl+Enter for a new line · Ctrl+C interrupts · Ctrl+D exits{RESET}")
    print(bar)


# --- input editor ---------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(s: str) -> int:
    """Number of printed columns in *s*, ignoring SGR color escapes.

    Assumes every visible character is one column wide, which holds for the
    prompt strings this CLI uses (ASCII plus a single-width '›')."""
    return len(_ANSI_RE.sub("", s))


def _read_input(prompt: str, history: list[str]) -> str:
    """Read one (possibly multi-line) prompt with a raw-mode line editor.

    Plain Enter submits. Ctrl+Enter inserts a newline so you can compose
    multi-line input; Alt+Enter and Shift+Enter do the same on terminals
    that don't encode Ctrl+Enter. Continuation lines are indented to line up
    under the first line's text. Backspace/Delete edit across line breaks,
    arrows + Home/End move the cursor, and Up/Down browse *history* once the
    cursor is on the buffer's top/bottom line.

    Raises KeyboardInterrupt on Ctrl+C and EOFError on Ctrl+D at an empty
    buffer, mirroring the builtin input() so the caller's handling is reused.
    """
    import codecs
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    plen = _visible_len(prompt)
    indent = " " * plen

    buf: list[str] = []      # the text being edited, '\n' marks line breaks
    cursor = 0               # insert position, an index into buf
    hist_idx = len(history)  # len(history) == "current (unsaved) buffer"
    saved = ""               # buffer stashed when stepping back into history
    decoder = codecs.getincrementaldecoder("utf-8")()

    def _text() -> str:
        return "".join(buf)

    def _pos(idx: int) -> tuple[int, int]:
        """(row, screen-col) of buf index *idx*."""
        before = _text()[:idx]
        row = before.count("\n")
        col = plen + (len(before) - (before.rfind("\n") + 1))
        return row, col

    def _index_at(row: int, col: int) -> int:
        """buf index nearest to a (row, screen-col), clamped to that row."""
        lines = _text().split("\n")
        row = max(0, min(row, len(lines) - 1))
        tcol = max(0, col - plen)
        start = sum(len(lines[r]) + 1 for r in range(row))
        return start + min(tcol, len(lines[row]))

    def _render(last_row: int) -> int:
        """Repaint the input region and place the cursor. Returns its row.

        Lines are assumed to fit the terminal width (no soft-wrap math)."""
        crow, ccol = _pos(cursor)
        total_rows = 1 + _text().count("\n")
        out = []
        if last_row > 0:
            out.append(f"\x1b[{last_row}A")
        out.append("\r\x1b[J")                # back to region top, clear it
        out.append(prompt)
        out.append(_text().replace("\n", "\r\n" + indent))
        up = (total_rows - 1) - crow          # from print end back to cursor
        if up > 0:
            out.append(f"\x1b[{up}A")
        out.append("\r")
        if ccol > 0:
            out.append(f"\x1b[{ccol}C")
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        return crow

    def _read_after_esc() -> bytes:
        """One byte following ESC, or b'' if none arrives promptly."""
        r, _, _ = select.select([fd], [], [], 0.03)
        return os.read(fd, 1) if r else b""

    old_attr = termios.tcgetattr(fd)
    # Kitty keyboard protocol (disambiguate) lets us see Ctrl+Enter; bracketed
    # paste keeps newlines in pasted text from being read as a submit.
    sys.stdout.write("\x1b[>1u\x1b[?2004h")
    sys.stdout.flush()
    try:
        tty.setraw(fd)
        last_row = _render(0)
        while True:
            ch = os.read(fd, 1)
            if not ch:
                continue
            o = ch[0]

            if o == 0x03:               # Ctrl+C
                raise KeyboardInterrupt
            if o == 0x04:               # Ctrl+D
                if not buf:
                    raise EOFError
                continue
            if o in (0x0D, 0x0A):       # Enter -> submit
                break
            if o in (0x7F, 0x08):       # Backspace
                if cursor > 0:
                    del buf[cursor - 1]
                    cursor -= 1
                    hist_idx = len(history)
                    last_row = _render(last_row)
                continue

            if o == 0x1B:               # escape sequence
                nb = _read_after_esc()
                if not nb:
                    continue            # lone ESC
                if nb in (b"\r", b"\n"):    # Alt+Enter -> newline
                    buf.insert(cursor, "\n")
                    cursor += 1
                    hist_idx = len(history)
                    last_row = _render(last_row)
                    continue
                if nb not in (b"[", b"O"):
                    continue
                params = b""
                final = b""
                while True:
                    c2 = _read_after_esc()
                    if not c2:
                        break
                    if 0x30 <= c2[0] <= 0x3F:       # params / digits / ';'
                        params += c2
                    elif 0x40 <= c2[0] <= 0x7E:     # final byte
                        final = c2
                        break
                    else:
                        break
                ps = params.split(b";")
                try:
                    key = int(ps[0]) if ps[0] else 0
                except ValueError:
                    key = 0
                try:
                    mod = int(ps[1]) if len(ps) > 1 and ps[1] else 1
                except ValueError:
                    mod = 1

                if final == b"u":               # kitty key event
                    if key in (13, 10):          # Enter
                        if mod != 1:             # any modifier -> newline
                            buf.insert(cursor, "\n")
                            cursor += 1
                            hist_idx = len(history)
                            last_row = _render(last_row)
                        else:
                            break                # bare Enter -> submit
                    elif key == 99 and mod != 1:    # Ctrl+C reported as key
                        raise KeyboardInterrupt
                    elif key == 100 and mod != 1 and not buf:  # Ctrl+D
                        raise EOFError
                    continue

                crow, ccol = _pos(cursor)
                if final == b"D":                       # Left
                    cursor = max(0, cursor - 1)
                elif final == b"C":                     # Right
                    cursor = min(len(buf), cursor + 1)
                elif final == b"A":                     # Up
                    if crow > 0:
                        cursor = _index_at(crow - 1, ccol)
                    elif hist_idx > 0:
                        if hist_idx == len(history):
                            saved = _text()
                        hist_idx -= 1
                        buf[:] = list(history[hist_idx])
                        cursor = len(buf)
                elif final == b"B":                     # Down
                    total_rows = 1 + _text().count("\n")
                    if crow < total_rows - 1:
                        cursor = _index_at(crow + 1, ccol)
                    elif hist_idx < len(history):
                        hist_idx += 1
                        text = saved if hist_idx == len(history) else history[hist_idx]
                        buf[:] = list(text)
                        cursor = len(buf)
                elif final in (b"H",):                  # Home
                    cursor = _index_at(crow, plen)
                elif final in (b"F",):                  # End
                    cursor = _index_at(crow, 10**9)
                elif final == b"~":
                    if params == b"3" and cursor < len(buf):    # Delete
                        del buf[cursor]
                    elif params in (b"1", b"7"):                # Home
                        cursor = _index_at(crow, plen)
                    elif params in (b"4", b"8"):                # End
                        cursor = _index_at(crow, 10**9)
                    elif params == b"200":                      # bracketed paste
                        data = bytearray()
                        end = b"\x1b[201~"
                        while not data.endswith(end):
                            data += os.read(fd, 1)
                        pasted = data[: -len(end)].decode("utf-8", "replace")
                        pasted = pasted.replace("\r\n", "\n").replace("\r", "\n")
                        for c in pasted:
                            buf.insert(cursor, c)
                            cursor += 1
                        hist_idx = len(history)
                last_row = _render(last_row)
                continue

            if o < 0x20:                # other control bytes: ignore
                continue

            # printable byte, possibly the start of a UTF-8 sequence
            text = decoder.decode(ch)
            while not text:
                text = decoder.decode(os.read(fd, 1))
            for c in text:
                buf.insert(cursor, c)
                cursor += 1
            hist_idx = len(history)
            last_row = _render(last_row)

        # submit: drop the cursor below the whole buffer, then a fresh line
        total_rows = 1 + _text().count("\n")
        down = (total_rows - 1) - last_row
        tail = (f"\x1b[{down}B" if down > 0 else "") + "\r\n"
        sys.stdout.write(tail)
        sys.stdout.flush()
        return _text()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        sys.stdout.write("\x1b[<u\x1b[?2004l")  # pop kitty mode, end paste mode
        sys.stdout.flush()


# --- streaming consumer (this process) ------------------------------------


def _consume_reply(req_q, out_q, interrupt_event, worker, messages, sampling, thinking):
    """Stream one reply from the generation process to stdout.

    Hands the turn to the worker, then prints token deltas as they arrive on
    *out_q*. Ctrl+C signals the worker to abort (via *interrupt_event*) and
    stops echoing, but keeps draining until the worker's 'done' so the queue
    stays in sync for the next turn. Returns (text, interrupted)."""
    import queue as _queue

    interrupt_event.clear()
    req_q.put(("generate", messages, sampling, thinking))

    pieces: list[str] = []
    n_tokens = 0
    interrupted = False
    t0 = time.time()
    while True:
        try:
            kind, payload = out_q.get(timeout=0.1)
        except _queue.Empty:
            if not worker.is_alive():
                sys.stdout.write(f"\n{RED}[generation process exited]{RESET}\n")
                sys.stdout.flush()
                break
            continue
        except KeyboardInterrupt:
            interrupted = True
            interrupt_event.set()
            continue

        if kind == "tok":
            n_tokens += 1
            pieces.append(payload)
            if not interrupted:
                sys.stdout.write(payload)
                sys.stdout.flush()
        elif kind == "error":
            sys.stdout.write(f"\n{RED}[generation error: {payload}]{RESET}\n")
            sys.stdout.flush()
        elif kind == "done":
            interrupted = interrupted or payload.get("interrupted", False)
            break

    dt = max(time.time() - t0, 1e-6)
    tag = "interrupted · " if interrupted else ""
    sys.stdout.write(f"\n{DIM}[{tag}{n_tokens} tok · {n_tokens / dt:.1f} tok/s]{RESET}\n")
    sys.stdout.flush()
    return "".join(pieces), interrupted


# --- commands -------------------------------------------------------------


def _parse_bool(s: str) -> bool:
    t = s.strip().lower()
    if t in ("true", "t", "yes", "y", "1", "on"):
        return True
    if t in ("false", "f", "no", "n", "0", "off"):
        return False
    raise ValueError(f"expected true/false, got {s!r}")


def _print_msg(role: str, content: str) -> None:
    color = {"user": CYAN, "assistant": GREEN, "system": MAGENTA}.get(role, "")
    label = f"{BOLD}{color}{role}{RESET}{DIM}:{RESET} "
    indent = " " * (len(role) + 2)
    lines = content.splitlines() or [""]
    print(label + lines[0])
    for line in lines[1:]:
        print(indent + line)


def _cmd_exit(arg, state):
    state["exit"] = True


def _cmd_reset(arg, state):
    n = len(state["messages"])
    state["messages"].clear()
    suffix = "" if n == 1 else "s"
    print(f"{DIM}[reset · cleared {n} message{suffix}]{RESET}")


def _cmd_system(arg, state):
    if arg.strip():
        state["system"] = arg
        print(f"{DIM}[system prompt set]{RESET}")
    else:
        state["system"] = None
        print(f"{DIM}[system prompt cleared]{RESET}")


def _set_numeric(arg, state, key, parser, validate, label):
    if not arg.strip():
        print(f"{DIM}{label} = {state['sampling'][key]}{RESET}")
        return
    try:
        v = parser(arg.strip())
    except ValueError:
        print(f"{RED}/{label}: cannot parse {arg.strip()!r}{RESET}")
        return
    err = validate(v)
    if err:
        print(f"{RED}/{label}: {err}{RESET}")
        return
    state["sampling"][key] = v
    print(f"{DIM}{label} = {v}{RESET}")


def _cmd_temperature(arg, state):
    _set_numeric(
        arg, state, "temperature", float,
        lambda v: "must be ≥ 0 (0 = greedy)" if v < 0 else None,
        "temperature",
    )


def _cmd_top_p(arg, state):
    _set_numeric(
        arg, state, "top_p", float,
        lambda v: "must be in (0, 1]" if not (0 < v <= 1) else None,
        "top_p",
    )


def _cmd_max_tokens(arg, state):
    _set_numeric(
        arg, state, "max_new_tokens", int,
        lambda v: "must be > 0" if v <= 0 else None,
        "max_new_tokens",
    )


def _cmd_thinking(arg, state):
    if not state["supports_thinking"]:
        print(f"{YELLOW}/thinking: this model's chat template has no thinking mode{RESET}")
        return
    if not arg.strip():
        print(f"{DIM}thinking = {state['thinking']}{RESET}")
        return
    try:
        v = _parse_bool(arg)
    except ValueError as e:
        print(f"{RED}/thinking: {e}{RESET}")
        return
    state["thinking"] = v
    print(f"{DIM}thinking = {v}{RESET}")


def _cmd_sampling(arg, state):
    s = state["sampling"]
    print(f"{DIM}sampling:{RESET}")
    print(f"  {CYAN}temperature{RESET}     = {s['temperature']}")
    print(f"  {CYAN}top_p{RESET}           = {s['top_p']}")
    print(f"  {CYAN}max_new_tokens{RESET}  = {s['max_new_tokens']}")
    if state["supports_thinking"]:
        print(f"  {CYAN}thinking{RESET}        = {state['thinking']}")


def _cmd_history(arg, state):
    if state["system"]:
        _print_msg("system", state["system"])
    if not state["messages"]:
        if not state["system"]:
            print(f"{DIM}(no messages){RESET}")
        return
    for m in state["messages"]:
        _print_msg(m["role"], m["content"])


def _cmd_help(arg, state):
    rows = [
        ("/exit", "quit"),
        ("/reset", "clear chat history (keeps system prompt)"),
        ("/system [msg]", "set system prompt; no arg = clear"),
        ("/temperature [v]", "show or set sampling temperature (0 = greedy)"),
        ("/top_p [v]", "show or set top_p"),
        ("/max_tokens [n]", "show or set max_new_tokens"),
    ]
    if state["supports_thinking"]:
        rows.append(("/thinking [bool]", "show or toggle the model's thinking mode"))
    rows += [
        ("/sampling", "print all sampling parameters"),
        ("/history", "print the current chat history"),
        ("/help", "show this list"),
    ]
    width = max(len(name) for name, _ in rows)
    print(f"{DIM}commands:{RESET}")
    for name, desc in rows:
        print(f"  {CYAN}{name.ljust(width)}{RESET}  {desc}")


_COMMANDS = {
    "exit": _cmd_exit,
    "quit": _cmd_exit,
    "q": _cmd_exit,
    "reset": _cmd_reset,
    "clear": _cmd_reset,
    "system": _cmd_system,
    "temperature": _cmd_temperature,
    "temp": _cmd_temperature,
    "top_p": _cmd_top_p,
    "topp": _cmd_top_p,
    "max_new_tokens": _cmd_max_tokens,
    "max_tokens": _cmd_max_tokens,
    "thinking": _cmd_thinking,
    "think": _cmd_thinking,
    "sampling": _cmd_sampling,
    "params": _cmd_sampling,
    "history": _cmd_history,
    "help": _cmd_help,
    "?": _cmd_help,
}


def _handle_command(line: str, state: dict) -> None:
    parts = line[1:].split(None, 1)
    if not parts:
        return
    raw = parts[0]
    cmd = raw.lower().replace("-", "_")
    arg = parts[1] if len(parts) > 1 else ""
    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(f"{RED}unknown command:{RESET} /{raw}  {DIM}(try /help){RESET}")
        return
    handler(arg, state)


# --- generation worker (separate process) ---------------------------------
#
# The worker owns the vLLM engine and an HF tokenizer (for chat templating).
# Protocol on the queues:
#   req_q  <- ("generate", messages, sampling, thinking) | ("shutdown",)
#   out_q  -> ("ready", info) once, then per turn:
#             ("tok", delta_text) * N, optional ("error", msg), ("done", info)
# Spawn (not fork) is required so CUDA is only ever initialised in the child.
# Assumes a vLLM new enough to expose AsyncLLMEngine.from_engine_args and the
# `generate(prompt, sampling_params, request_id)` async-iterator API (>= 0.6).


def _generation_worker(model_path, dtype, device, vocab_mask, max_model_len,
                       gpu_mem_util, req_q, out_q, interrupt_event):
    """Process entry point: run the async engine loop, reporting load errors."""
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")  # vLLM is chatty
    import asyncio

    try:
        asyncio.run(_worker_main(model_path, dtype, device, vocab_mask,
                                 max_model_len, gpu_mem_util,
                                 req_q, out_q, interrupt_event))
    except BaseException as e:  # last resort: a clean message, not a traceback
        try:
            out_q.put(("ready", {"error": f"{type(e).__name__}: {e}"}))
        except Exception:
            pass


async def _worker_main(model_path, dtype, device, vocab_mask, max_model_len,
                       gpu_mem_util, req_q, out_q, interrupt_event):
    import asyncio

    try:
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    except BaseException as e:
        out_q.put(("ready", {"error": f"import failed ({e}); is vllm installed?"}))
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except BaseException as e:
        out_q.put(("ready", {"error": f"tokenizer: {type(e).__name__}: {e}"}))
        return
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        out_q.put(("ready", {"error": f"{model_path!r} has no chat_template "
                             "— this CLI only supports instruct-style models."}))
        return

    engine_args = dict(model=model_path, dtype=dtype, trust_remote_code=True,
                       disable_log_requests=True)
    if device:
        engine_args["device"] = device
    if max_model_len:
        engine_args["max_model_len"] = max_model_len
    if gpu_mem_util:
        engine_args["gpu_memory_utilization"] = gpu_mem_util
    try:
        engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))
    except BaseException as e:
        out_q.put(("ready", {"error": f"engine init: {type(e).__name__}: {e}"}))
        return

    # A vocab mask maps cleanly to vLLM's native `allowed_token_ids`, which
    # works on both the v0 and v1 engines (unlike custom logits processors).
    allowed_token_ids = None
    mask_info = None
    if vocab_mask:
        try:
            from inference.vocab_mask import VocabMaskSpec, warn_if_eos_masked
            spec = VocabMaskSpec.from_file(vocab_mask)
            mask = spec.materialize(len(tokenizer))
            warn_if_eos_masked(mask, getattr(tokenizer, "eos_token_id", None))
            allowed_token_ids = mask.nonzero().flatten().tolist()
            mask_info = {"n_allow": len(allowed_token_ids),
                         "vocab_size": len(tokenizer), "path": vocab_mask}
        except BaseException as e:
            out_q.put(("ready", {"error": f"vocab mask: {type(e).__name__}: {e}"}))
            return

    out_q.put(("ready", {
        "error": None,
        "supports_thinking": "enable_thinking" in template,
        "mask_info": mask_info,
        "dtype": dtype,
    }))

    loop = asyncio.get_running_loop()
    req_id = 0
    while True:
        # Block off-loop for the next request so the engine loop stays free.
        req = await loop.run_in_executor(None, req_q.get)
        if not req or req[0] == "shutdown":
            return
        if req[0] != "generate":
            continue
        _, messages, sampling, thinking = req
        req_id += 1
        rid = str(req_id)

        chat_kwargs = {}
        if "enable_thinking" in template:
            chat_kwargs["enable_thinking"] = thinking
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_kwargs,
            )
            # Chat template already inserts special tokens; don't add them twice.
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        except BaseException as e:
            out_q.put(("error", f"templating: {type(e).__name__}: {e}"))
            out_q.put(("done", {"interrupted": False}))
            continue

        greedy = sampling["temperature"] <= 0
        sampling_params = SamplingParams(
            max_tokens=sampling["max_new_tokens"],
            temperature=0.0 if greedy else sampling["temperature"],
            top_p=1.0 if greedy else sampling["top_p"],
            allowed_token_ids=allowed_token_ids,
        )

        prev = ""
        interrupted = False
        try:
            # The async engine yields cumulative text by default (emit the new
            # tail); tolerate delta-mode builds where .text is already the tail.
            async for req_out in engine.generate(
                {"prompt_token_ids": prompt_ids}, sampling_params, rid,
            ):
                text = req_out.outputs[0].text
                delta = text[len(prev):] if text.startswith(prev) else text
                prev = text
                if delta:
                    out_q.put(("tok", delta))
                if interrupt_event.is_set():
                    interrupted = True
                    await engine.abort(rid)
                    break
        except BaseException as e:
            out_q.put(("error", f"{type(e).__name__}: {e}"))
        out_q.put(("done", {"interrupted": interrupted or interrupt_event.is_set()}))


# --- main -----------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Streaming CLI chatbot backed by a vLLM generation process.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--device", default=None,
                    help="vLLM device (default: auto-detect, normally cuda)")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Engine dtype passed to vLLM (default: bfloat16)",
    )
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="vLLM max context length (default: from model config)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="Fraction of GPU memory for the vLLM KV cache (vLLM default ~0.9)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--system", default=None, help="Initial system prompt")
    ap.add_argument(
        "--thinking",
        action="store_true",
        help="For models whose chat template supports it (e.g. Qwen3), enable the "
        "<think>...</think> reasoning block (off by default — keeps replies snappy).",
    )
    ap.add_argument(
        "--vocab-mask",
        default=None,
        help="Path to a vocabulary mask file (.json sparse spec / .npy dense "
        "bool array / .safetensors bool tensor named 'mask'). When set, only "
        "the allowed tokens can be generated (via vLLM allowed_token_ids). "
        "Build one with 'python -m inference.scripts.build_vocab_mask'.",
    )
    args = ap.parse_args()

    # Spawn the vLLM generation process. Spawn (not fork) keeps CUDA out of
    # this process entirely — only the child ever touches the GPU.
    ctx = mp.get_context("spawn")
    req_q = ctx.Queue()
    out_q = ctx.Queue()
    interrupt_event = ctx.Event()
    worker = ctx.Process(
        target=_generation_worker,
        args=(args.model, args.dtype, args.device, args.vocab_mask,
              args.max_model_len, args.gpu_memory_utilization,
              req_q, out_q, interrupt_event),
        # Not daemonic: vLLM's engine may spawn its own subprocesses, which a
        # daemonic parent is forbidden from doing. We reap it explicitly below
        # and via atexit so it never outlives this process.
        daemon=False,
    )
    print(f"{DIM}loading {args.model} (vllm)...{RESET}", flush=True)
    worker.start()

    def _reap_worker():
        if worker.is_alive():
            try:
                req_q.put_nowait(("shutdown",))
            except Exception:
                pass
            worker.terminate()
    atexit.register(_reap_worker)

    # Wait for the engine to come up. Poll so a worker that dies during load
    # (e.g. CUDA OOM, SIGKILL) surfaces instead of hanging, and so Ctrl+C aborts.
    import queue as _queue
    kind, info = None, None
    try:
        while True:
            try:
                kind, info = out_q.get(timeout=0.5)
                break
            except _queue.Empty:
                if not worker.is_alive():
                    print(f"{RED}error:{RESET} generation process died during startup")
                    raise SystemExit(1)
    except KeyboardInterrupt:
        worker.terminate()
        raise SystemExit(130)
    if kind != "ready" or info.get("error"):
        msg = info.get("error") if kind == "ready" else f"unexpected {kind!r}"
        print(f"{RED}error:{RESET} {msg}")
        worker.terminate()
        raise SystemExit(1)
    print(f"{DIM}loaded ({info['dtype']}){RESET}")
    if info.get("mask_info"):
        mi = info["mask_info"]
        print(f"{DIM}vocab mask: {mi['n_allow']} / {mi['vocab_size']} tokens "
              f"allowed  ({mi['path']}){RESET}")

    _print_banner(args.model)

    state = {
        "system": args.system,
        "messages": [],  # list[{"role", "content"}]
        "exit": False,
        "supports_thinking": info["supports_thinking"],
        "thinking": bool(args.thinking),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
    }

    prompt_raw = f"{BOLD}{CYAN}you{RESET} {DIM}›{RESET} "
    user_prompt = _readline_safe(prompt_raw)  # for the input() fallback
    # The raw-mode editor (Ctrl+Enter multi-line) needs a real TTY on both
    # ends and POSIX termios; otherwise fall back to the builtin input().
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if interactive:
        try:
            import termios  # noqa: F401
            import tty  # noqa: F401
        except ImportError:
            interactive = False
    history: list[str] = []
    armed_exit = False

    while not state["exit"]:
        try:
            line = _read_input(prompt_raw, history) if interactive else input(user_prompt)
            armed_exit = False
        except KeyboardInterrupt:
            if armed_exit:
                print()
                break
            armed_exit = True
            print(f"\n{DIM}(press Ctrl+C again to quit, or keep typing){RESET}")
            continue
        except EOFError:
            print()
            break

        stripped = line.strip()
        if not stripped:
            continue

        # Up/Down browse past prompts; store the raw (multi-line) text and
        # skip immediate duplicates.
        if interactive and (not history or history[-1] != line):
            history.append(line)

        if stripped.startswith("/"):
            _handle_command(stripped, state)
            continue

        # Build the full message list each turn so /system updates take
        # effect immediately and survive /reset.
        messages = []
        if state["system"]:
            messages.append({"role": "system", "content": state["system"]})
        messages.extend(state["messages"])
        messages.append({"role": "user", "content": stripped})

        # Pass sampling + thinking by value each turn so /temperature, /top_p,
        # /max_tokens and /thinking take effect mid-conversation. The worker
        # builds the actual SamplingParams and applies the chat template.
        print(f"{BOLD}{GREEN}bot{RESET} {DIM}›{RESET} ", end="", flush=True)
        reply, _ = _consume_reply(
            req_q, out_q, interrupt_event, worker,
            messages, dict(state["sampling"]), state["thinking"],
        )

        # Even on interrupt, keep partial assistant output in history so
        # follow-ups have continuity. Users can /reset to wipe.
        state["messages"].append({"role": "user", "content": stripped})
        state["messages"].append({"role": "assistant", "content": reply})

    # Tell the worker to stop, then reap it.
    try:
        req_q.put(("shutdown",))
        worker.join(timeout=10)
    except Exception:
        pass
    if worker.is_alive():
        worker.terminate()
    print(f"{DIM}bye{RESET}")


if __name__ == "__main__":
    main()
