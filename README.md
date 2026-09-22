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

## Download and prepare APRIL dataset

Download the TME portions of the APRIL dataset:

```bash
mkdir -p data/raw/april

hf download uw-math-ai/APRIL \
  train/tme_train.jsonl \
  val/tme_val.jsonl \
  test/tme_test.jsonl \
  --repo-type dataset \
  --local-dir data/raw/april
```

This produces:

```text
data/raw/april/
├── train/tme_train.jsonl
├── val/tme_val.jsonl
└── test/tme_test.jsonl
```

Create the deterministic 500/100/100 benchmark and chat-format SFT files:

```bash
python scripts/prepare_data.py
```

Before training, compile-audit the selected examples with the pinned APRIL
verifier:

```bash
python scripts/verify_dataset.py --workers 1
```

## SIMPLE MODEL

This uses Qwen3-0.6B, 16 training examples, 4
validation examples, one epoch, 512-token sequences, and a rank-8 LoRA adapter.

### 1. Download the model

Download the standard Hugging Face checkpoint without converting it:

```bash
mkdir -p models
hf download Qwen/Qwen3-0.6B \
  --local-dir models/Qwen3-0.6B
```

### 2. Create the dataset

The full sampled data remains unchanged under `data/sft/`. Create a separate,
deterministic subset for the smoke test:

```bash
mkdir -p data/sft-smoke

head -n 16 data/sft/train.jsonl \
  > data/sft-smoke/train.jsonl

head -n 4 data/sft/valid.jsonl \
  > data/sft-smoke/valid.jsonl
```

Confirm the counts:

```bash
wc -l data/sft-smoke/*.jsonl
```

Expected result:

```text
16 data/sft-smoke/train.jsonl
 4 data/sft-smoke/valid.jsonl
20 total
```

The generated `data/sft-smoke/` directory is ignored by Git.

### 3. Train the adapter

The settings are stored in `configs/qwen3_tme_lora_smoke.yaml`:

```bash
python scripts/train.py \
  --config configs/qwen3_tme_lora_smoke.yaml
```

The final PEFT adapter is written to:

```text
runs/tme_lora_smoke_hf/adapters/
```

Check that it was created:

```bash
ls -lh runs/tme_lora_smoke_hf/adapters
```

### 4. Generate the zero-shot baseline

```bash
python scripts/generate.py \
  --model models/Qwen3-0.6B \
  --input data/sft/test.jsonl \
  --output runs/smoke_zero_shot/predictions.jsonl \
  --max-tokens 1024
```

`generate.py` currently expects all 100 test examples. It saves every response
immediately, so the command can be stopped with `Ctrl-C` and resumed by running
the identical command again.

### 5. Compile the zero-shot repairs

After all 100 predictions have been generated:

```bash
python scripts/evaluate_predictions.py \
  --predictions runs/smoke_zero_shot/predictions.jsonl \
  --results runs/smoke_zero_shot/compilation_results.jsonl \
  --summary runs/smoke_zero_shot/score.json \
  --expected 100
```

### 6. Generate with the smoke-test adapter

```bash
python scripts/generate.py \
  --model models/Qwen3-0.6B \
  --adapter-path runs/tme_lora_smoke_hf/adapters \
  --input data/sft/test.jsonl \
  --output runs/smoke_fine_tuned/predictions.jsonl \
  --max-tokens 1024
```

### 7. Compile the fine-tuned repairs

```bash
python scripts/evaluate_predictions.py \
  --predictions runs/smoke_fine_tuned/predictions.jsonl \
  --results runs/smoke_fine_tuned/compilation_results.jsonl \
  --summary runs/smoke_fine_tuned/score.json \
  --expected 100
```

Compare the resulting scores:

```text
runs/smoke_zero_shot/score.json
runs/smoke_fine_tuned/score.json
```

 <!--
## Full Qwen3-4B experiment

The old converted MLX 4-bit checkpoint is not compatible with this Transformers
and PEFT training stack. Download the standard Hugging Face checkpoint:

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir models/Qwen3-4B-Instruct-2507
```

This experiment is substantially more demanding than the smoke test and may
not fit comfortably in 16 GB of unified memory.

### Fine-tune a PEFT LoRA adapter

```bash
python scripts/train.py --config configs/qwen3_tme_lora.yaml
```

The final adapter is written to `runs/tme_lora/adapters`. Checkpoints used for
resuming training remain under `runs/tme_lora/checkpoint-*`.

The configured 2,048-token limit matches the previous experiment. The trainer
prints how many rows were truncated and reserves supervised repair tokens when
a prompt is longer than the available context window.

### Generate repairs

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

### Compile generated repairs

```bash
python scripts/evaluate_predictions.py \
  --predictions runs/zero_shot/predictions.jsonl \
  --results runs/zero_shot/compilation_results.jsonl \
  --summary runs/zero_shot/score.json
```

Use the corresponding `runs/fine_tuned/` paths for the adapter run.
-->