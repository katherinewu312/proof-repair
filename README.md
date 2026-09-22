# Proof repair

This project uses Hugging Face Transformers for Qwen inference and training and
PEFT for LoRA adapters. Lean candidates are evaluated independently with the
pinned project in `verifier/`.

## Lean environment

Lean is installed through `elan`, the Lean toolchain manager. Elan reads each
directory's `lean-toolchain` file and automatically uses the version pinned by
that project.

First make sure `git` and `curl` are installed. On macOS or Linux, install elan
with:

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
source "$HOME/.elan/env"
```

Choose the default installation when prompted. On Windows, follow the official
[Lean installation instructions](https://lean-lang.org/install/) or install
elan from PowerShell as described in the
[manual installation guide](https://lean-lang.org/install/manual/).

For interactive proof development, install VS Code and the official **Lean 4**
extension published by `leanprover`.

Confirm that the command-line tools are available:

```bash
elan --version
lean --version
lake --version
```

From the repository root, install the root project's pinned Lean 4.28 and
Mathlib environment:

```bash
lake update
lake exe cache get
```

Compile the example file through Lake so that Lean can find Mathlib:

```bash
lake env lean proofs.lean
```

Do not invoke `lean proofs.lean` directly; doing so can produce an `unknown
module prefix 'Mathlib'` error because it bypasses the project's Lake
environment.

APRIL examples are checked in a separate verifier pinned to Lean 4.22.0-rc4.
Install that environment independently:

```bash
cd verifier
lake update
lake exe cache get
lake env lean --version
cd ..
```

## Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Download the Hugging Face model

The old converted 4-bit checkpoint is not compatible with this training stack.
Download the standard Hugging Face checkpoint without converting it:

```bash
mkdir -p models
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir models/Qwen3-4B-Instruct-2507
```

You may alternatively pass the Hub ID `Qwen/Qwen3-4B-Instruct-2507` directly to
the scripts, but a local directory makes runs reproducible after downloading.

## Fine-tune a PEFT LoRA adapter

```bash
python scripts/train.py --config configs/qwen3_tme_lora.yaml
```

The final adapter is written to `runs/tme_lora/adapters`. Checkpoints used for
resuming training remain under `runs/tme_lora/checkpoint-*`.

The configured 2,048-token limit matches the previous experiment. The trainer
prints how many rows were truncated and reserves supervised repair tokens when
a prompt is longer than the available context window.

## Generate repairs

Zero-shot baseline:

```bash
python scripts/generate.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --input data/sft/test.jsonl \
  --output runs/zero_shot/predictions.jsonl \
  --max-tokens 1024
```

Fine-tuned adapter:

```bash
python scripts/generate.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter-path runs/tme_lora/adapters \
  --input data/sft/test.jsonl \
  --output runs/fine_tuned/predictions.jsonl \
  --max-tokens 1024
```

Both commands use greedy decoding, save each response immediately, and resume
from existing prediction files.

## Compile generated repairs

```bash
python scripts/evaluate_predictions.py \
  --predictions runs/zero_shot/predictions.jsonl \
  --results runs/zero_shot/compilation_results.jsonl \
  --summary runs/zero_shot/score.json
```

Use the corresponding `runs/fine_tuned/` paths for the adapter run.
