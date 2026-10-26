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

Download the APRIL dataset. For example, for the TME (Tactic mutation error) portion of the dataset, run:

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

Create the training / validation / test examples directly from the raw data, including chat-format SFT files:

```bash
python scripts/prepare_data.py
```

The outputs are `data/benchmark/{train,valid,test}.jsonl` (original records),
`data/sft/{train,valid,test}.jsonl` (chat format), and
`data/benchmark/manifest.json` (selection metadata). Rerunning preparation
replaces these files.

The default split size is 16 training / 4 validation / 100 test. To choose different split sizes, pass:

```bash
python scripts/prepare_data.py --train-size 32 --valid-size 8 --test-size 50
```

Before training, verify that each reference proof compiles and each broken proof fails. Examples that fail these checks are replaced with verified examples from the raw dataset, keeping the requested training, validation, and test sizes unchanged:

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

### 2. Train the adapter

Training reads the prepared `data/sft/` files directly. The settings are stored
in `configs/qwen3_tme_lora.yaml`:

```bash
python scripts/train.py \
  --config configs/qwen3_tme_lora.yaml
```

The final PEFT adapter is written to:

```text
runs/tme_lora/adapters/
```

Check that it was created:

```bash
ls -lh runs/tme_lora/adapters
```

### 3. Generate repairs with the Base model (zero-shot)

```bash
python scripts/generate.py \
  --model models/Qwen3-0.6B \
  --input data/sft/test.jsonl \
  --output runs/zero_shot/predictions.jsonl \
  --max-tokens 1024
```

`generate.py` processes all examples in the input file. It saves every response
immediately, so the command can be stopped with `Ctrl-C` and resumed by running
the identical command again.

### 4. Compile repairs from the Base model (zero-shot)

After all 100 predictions have been generated:

```bash
python scripts/evaluate_predictions.py \
  --predictions runs/zero_shot/predictions.jsonl \
  --results runs/zero_shot/compilation_results.jsonl \
  --summary runs/zero_shot/score.json \
  --expected 100
```

### 5. Generate repairs with the LoRA fine-tuned model

```bash
python scripts/generate.py \
  --model models/Qwen3-0.6B \
  --adapter-path runs/tme_lora/adapters \
  --input data/sft/test.jsonl \
  --output runs/fine_tuned/predictions.jsonl \
  --max-tokens 1024
```

### 6. Compile repairs from the LoRA fine-tuned model

```bash
python scripts/evaluate_predictions.py \
  --predictions runs/fine_tuned/predictions.jsonl \
  --results runs/fine_tuned/compilation_results.jsonl \
  --summary runs/fine_tuned/score.json \
  --expected 100
```

Compare the Base model and LoRA fine-tuned model scores below.
Both use the same prompt format, the LoRA fine-tuned model also loads the
trained adapter:

```text
runs/zero_shot/score.json
runs/fine_tuned/score.json
```

If you change `--test-size`, set `--expected` in both evaluation commands to
that same size. (Use new run output directories after changing the dataset so
previous predictions and adapters remain associated with their original data.)
