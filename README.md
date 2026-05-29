# inference

A simple, modular library for **text generation** and **embedding** with large language models. One backend-agnostic API drives [vLLM](https://github.com/vllm-project/vllm), HuggingFace `transformers`, and Ray Data, so the same config runs single-GPU, multi-GPU (tensor-parallel), or sharded across a cluster without changing your code.

On top of generation it ships three batteries-included tools: an **LLM-judge data filter**, a **hard vocabulary-masking** layer for constrained decoding, and a **streaming terminal chat CLI**.

## Features

- **Backends**: `vllm`, `hf` (transformers), and `ray_vllm` (Ray Data) behind one `GenerationConfig` — switch with a single field.
- **Generation**: batched generation from files or single prompts, multi-sample decoding, chat-template handling, system prompts, and assistant prefill.
- **Thinking-mode control**: toggle, force, or budget reasoning blocks for models that support them (auto-detected for Qwen3+ families).
- **Adapters**: loads PEFT/LoRA adapter directories and partial-finetune checkpoints transparently (PEFT-merge for HF, runtime `LoRARequest` for vLLM).
- **Embedding**: mean / CLS / last-token pooling (extensible via a registry), optional normalization, arbitrary hidden-layer extraction, instruction prefixes, and sharded `.npy` output.
- **Vocab masking**: hard constraint on which token IDs may be sampled (`-inf` logits, zero probability), honored by all three backends and the chat CLI. Sparse JSON, dense `.npy`, or `.safetensors` mask formats.
- **LLM-judge filter**: keep/reject JSONL records via a one-token verdict from a judge model, with a strict `KEEP` / `REJECT` contract.
- **Chat CLI**: streaming terminal chatbot backed by a vLLM `AsyncLLMEngine`, with a multi-line line editor, history, and runtime slash-commands.

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .

# Backend extras (install at least one):
pip install -e ".[vllm]"   # vLLM backend + the chat CLI
pip install -e ".[ray]"    # Ray Data backend
pip install -e ".[all]"    # both
```

The core install pulls in `torch`, `transformers`, `numpy`, and `tqdm`. `vllm` and `ray[data]` are optional because they are large and only needed for their respective backends — the `hf` backend works with the core install alone.

## Quickstart

### Generate

```bash
# vLLM, file input, with a prompt template
python -m inference.scripts.generate \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --backend vllm \
    --input-path data.jsonl \
    --output-path output.jsonl \
    --template "Answer the question: {question}" \
    --sampling-params '{"max_tokens": 512, "temperature": 0.7}' \
    --backend-kwargs '{"tensor_parallel_size": 4}'

# Single prompt, many samples
python -m inference.scripts.generate \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --prompt "Write a haiku about coding" \
    --n-samples 10 \
    --output-path haikus.jsonl
```

### Embed

```bash
python -m inference.scripts.embed \
    --model-path intfloat/e5-large-v2 \
    --input-path queries.jsonl \
    --key query \
    --instruction "Retrieve relevant passages" \
    --layer 12 \
    --pooling mean \
    --output-path query_embeddings.npy
```

### Filter (LLM judge)

```bash
python -m inference.scripts.filter \
    --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --backend vllm \
    --backend-kwargs '{"tensor_parallel_size": 8, "gpu_memory_utilization": 0.9}' \
    --input-path responses_es.jsonl \
    --output-path responses_es_kept.jsonl \
    --rejected-path responses_es_dropped.jsonl \
    --key response \
    --language Spanish
```

### Build a vocab mask

```bash
# Digits-only generation
python -m inference.scripts.build_vocab_mask \
    --tokenizer Qwen/Qwen3-1.7B \
    --mode allow \
    --allow-chars "0123456789" \
    --allow-eos \
    --output digits.json
```

Pass the resulting mask to generation via `GenerationConfig(output_vocab_mask="digits.json")` or `--vocab-mask` in the chat CLI.

### Chat (streaming CLI)

```bash
python -m inference.chat.cli --model Qwen/Qwen3-1.7B
```

Slash-commands include `/system`, `/temperature`, `/top_p`, `/max_tokens`, `/thinking`, `/sampling`, `/history`, `/reset`, and `/exit`. Enter submits; Ctrl+Enter (or Alt/Shift+Enter) inserts a newline for multi-line prompts.

## Python API

```python
from inference import GenerationConfig, GenerationModel

cfg = GenerationConfig(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    backend="vllm",
    sampling_params={"max_tokens": 256, "temperature": 0.7},
    backend_kwargs={"tensor_parallel_size": 4},
)
model = GenerationModel(cfg)
outputs = model.generate(["Explain attention in one sentence."])

from inference import EmbeddingConfig, EmbeddingModel
emb = EmbeddingModel(EmbeddingConfig(model_path="intfloat/e5-large-v2", pooling="mean"))
vectors = emb.embed(["hello world"])
```

## Package layout

```
inference/
├── src/inference/
│   ├── generate.py        # GenerationConfig + GenerationModel (vllm / hf / ray_vllm)
│   ├── embed.py           # EmbeddingConfig + EmbeddingModel + pooling registry
│   ├── filter.py          # LLM-judge keep/reject filter primitives
│   ├── vocab_mask.py       # VocabMaskSpec: hard vocabulary constraints
│   ├── formatting.py      # Prompt-template / escaped-string formatting
│   ├── io.py              # JSONL/JSON record loading + output writers (registries)
│   ├── chat/cli.py        # Streaming terminal chatbot (vLLM AsyncLLMEngine)
│   └── scripts/           # CLI entry points (generate / embed / filter / build_vocab_mask)
└── pyproject.toml
```

## Configuration

`GenerationConfig` is backend-agnostic — the same object is consumed by every backend:

- `model_path`, `backend` (`vllm` / `hf` / `ray_vllm`), `mode` (`instruct` / `base`)
- `system_prompt`, `prefill_assistant`, `chat_template`, `chat_template_tokenizer`
- `sampling_params` — backend-agnostic names (`max_tokens`, `temperature`, `top_p`, `top_k`); `max_tokens` is auto-translated to `max_new_tokens` for HF
- `backend_kwargs` — passed straight to the backend constructor (e.g. `tensor_parallel_size`, `gpu_memory_utilization` for vLLM; `concurrency` for HF)
- `enable_thinking` / `force_thinking` / `thinking_budget` — reasoning-mode controls (auto-detected for Qwen3+ families)
- `output_vocab_mask` — a mask file path or a `VocabMaskSpec`

`EmbeddingConfig` exposes `model_path`, `pooling` (any registered strategy), `normalize`, and `layer` (which hidden layer to pull, `None` = last hidden state).

## Data format

Inputs are JSONL (one record per line) or JSON (a list, or a single object). The `--template` / `--key` flags select and format fields per record. New input formats and output writers can be registered via the `@register_format` and `@register_output` decorators in [io.py](src/inference/io.py).

## Notes & caveats

- **Backend availability**: `vllm` and `ray[data]` are optional extras. Importing the library works without them; constructing a `GenerationModel` with a backend you haven't installed will fail at construction.
- **`trust_remote_code`**: the adapter/partial-merge load paths call HuggingFace loaders with `trust_remote_code=True`. Only point them at model repos you trust.
- **Temp space for partial merges**: the vLLM partial-finetune path materializes a merged checkpoint to `$TMPDIR`. For large (10B+) models, set `$TMPDIR` to a disk with room to spare; the temp dir is cleaned up on exit.

## License

No license file is currently included. Add one before publishing if you intend others to reuse this code.
