#!/usr/bin/env python3
"""Fine-tune Qwen for APRIL proof repair with Transformers and PEFT LoRA."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen3_tme_lora.yaml"),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="latest",
        help="Resume from a checkpoint path, or the latest checkpoint if omitted",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            selected_id = row.get("selected_id")
            if not isinstance(selected_id, str) or not selected_id:
                raise ValueError(f"{path}:{line_number}: missing selected_id")
            if selected_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate {selected_id}")
            seen.add(selected_id)
            rows.append(row)
    return rows


def conversation_parts(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"{row['selected_id']}: expected three chat messages")
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(
            f"{row['selected_id']}: expected system, user, assistant roles"
        )
    if not all(isinstance(message.get("content"), str) for message in messages):
        raise ValueError(f"{row['selected_id']}: message content must be text")
    prompt = [
        {"role": message["role"], "content": message["content"]}
        for message in messages[:2]
    ]
    return prompt, messages[2]["content"]


@dataclass
class EncodedSplit:
    rows: list[dict[str, list[int]]]
    truncated: int
    left_truncated_prompts: int


def encode_split(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    min_response_tokens: int,
) -> EncodedSplit:
    encoded_rows: list[dict[str, list[int]]] = []
    truncated = 0
    left_truncated_prompts = 0

    for row in rows:
        prompt_messages, response = conversation_parts(row)
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            response_ids.append(tokenizer.eos_token_id)
        if not response_ids:
            raise ValueError(f"{row['selected_id']}: empty assistant target")

        original_length = len(prompt_ids) + len(response_ids)
        if original_length > max_length:
            truncated += 1
            response_reserve = min(
                len(response_ids), max(1, min_response_tokens, max_length - 1)
            )
            prompt_budget = max_length - response_reserve
            if len(prompt_ids) > prompt_budget:
                prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
                left_truncated_prompts += 1
            response_budget = max_length - len(prompt_ids)
            response_ids = response_ids[:response_budget]

        if not response_ids:
            raise ValueError(
                f"{row['selected_id']}: max_seq_length leaves no supervised tokens"
            )
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids.copy()
        encoded_rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )

    return EncodedSplit(encoded_rows, truncated, left_truncated_prompts)


class ListDataset:
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class CausalLMCollator:
    def __init__(self, torch: Any, pad_token_id: int) -> None:
        self.torch = torch
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        longest = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = longest - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": self.torch.tensor(input_ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(
                attention_masks, dtype=self.torch.long
            ),
            "labels": self.torch.tensor(labels, dtype=self.torch.long),
        }


def require(config: dict[str, Any], key: str, expected_type: type) -> Any:
    value = config.get(key)
    if not isinstance(value, expected_type):
        raise ValueError(f"config field {key!r} must be {expected_type.__name__}")
    return value


def resolve_dtype(torch: Any, name: str) -> Any:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {name}") from error


def check_device(torch: Any, requested: str) -> str:
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def write_run_config(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)

    # Imports are lazy so configuration/help checks do not initialize Torch.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    model_name = require(config, "model_name_or_path", str)
    data_dir = Path(require(config, "data_dir", str))
    output_dir = Path(require(config, "output_dir", str))
    final_adapter_dir = Path(require(config, "final_adapter_dir", str))
    max_length = int(config.get("max_seq_length", 2048))
    min_response_tokens = int(config.get("min_response_tokens", 256))
    if max_length < 2 or not 1 <= min_response_tokens < max_length:
        raise ValueError("invalid max_seq_length/min_response_tokens configuration")

    local_model = Path(model_name)
    if model_name.startswith((".", "/", "models/")) and not local_model.is_dir():
        raise FileNotFoundError(local_model)

    seed = int(config.get("seed", 20260921))
    set_seed(seed)
    device = check_device(torch, str(config.get("device", "auto")))
    dtype_name = str(config.get("dtype", "bfloat16"))
    dtype = resolve_dtype(torch, dtype_name)
    print(f"Training backend: Transformers + PEFT on {device} ({dtype_name})")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    train_source = read_jsonl(data_dir / "train.jsonl")
    valid_source = read_jsonl(data_dir / "valid.jsonl")
    train = encode_split(
        train_source,
        tokenizer,
        max_length=max_length,
        min_response_tokens=min_response_tokens,
    )
    valid = encode_split(
        valid_source,
        tokenizer,
        max_length=max_length,
        min_response_tokens=min_response_tokens,
    )
    print(
        f"train: {len(train.rows)} rows, {train.truncated} truncated "
        f"({train.left_truncated_prompts} prompts left-truncated)"
    )
    print(
        f"valid: {len(valid.rows)} rows, {valid.truncated} truncated "
        f"({valid.left_truncated_prompts} prompts left-truncated)"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    lora = require(config, "lora", dict)
    target_modules = lora.get("target_modules")
    if not isinstance(target_modules, list) or not all(
        isinstance(item, str) for item in target_modules
    ):
        raise ValueError("lora.target_modules must be a list of strings")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=int(lora.get("rank", 32)),
        lora_alpha=int(lora.get("alpha", 64)),
        lora_dropout=float(lora.get("dropout", 0.1)),
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    gradient_checkpointing = bool(config.get("gradient_checkpointing", True))
    if gradient_checkpointing:
        model.enable_input_require_grads()

    num_train_epochs = float(config.get("num_train_epochs", 3))
    train_batch_size = int(config.get("per_device_train_batch_size", 1))
    gradient_accumulation_steps = int(
        config.get("gradient_accumulation_steps", 8)
    )
    optimizer_steps_per_epoch = math.ceil(
        math.ceil(len(train.rows) / train_batch_size)
        / gradient_accumulation_steps
    )
    estimated_optimizer_steps = math.ceil(
        optimizer_steps_per_epoch * num_train_epochs
    )
    warmup_ratio = float(config.get("warmup_ratio", 0.1))
    if not 0 <= warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be between 0 and 1")
    warmup_steps = round(estimated_optimizer_steps * warmup_ratio)

    # Transformers currently validates bf16 as a CUDA/XLA feature and rejects
    # its AMP flag on MPS. The model is already loaded in the requested dtype,
    # so MPS should not ask Trainer to enable a second mixed-precision layer.
    use_bf16_amp = dtype_name == "bfloat16" and device != "mps"
    use_fp16_amp = dtype_name == "float16" and device == "cuda"
    if device == "mps" and dtype_name in {"bfloat16", "float16"}:
        print(
            f"MPS: model weights use {dtype_name}; Trainer AMP flags are disabled"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(config.get("learning_rate", 1.0e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        warmup_steps=warmup_steps,
        lr_scheduler_type=str(config.get("lr_scheduler_type", "cosine")),
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=int(config.get("logging_steps", 10)),
        eval_strategy="steps",
        eval_steps=int(config.get("eval_steps", 25)),
        save_strategy="steps",
        save_steps=int(config.get("save_steps", 25)),
        save_total_limit=int(config.get("save_total_limit", 3)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16_amp,
        fp16=use_fp16_amp,
        use_cpu=device == "cpu",
        optim="adamw_torch",
        report_to=str(config.get("report_to", "none")),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ListDataset(train.rows),
        eval_dataset=ListDataset(valid.rows),
        data_collator=CausalLMCollator(torch, tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    resume: bool | str | None = args.resume_from_checkpoint
    if resume == "latest":
        resume = True
    trainer.train(resume_from_checkpoint=resume)

    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_adapter_dir)
    write_run_config(
        output_dir / "run_config.json",
        {
            "backend": "transformers_peft",
            "source_config": str(args.config),
            "config": config,
            "resolved_device": device,
            "train_rows": len(train.rows),
            "valid_rows": len(valid.rows),
            "train_truncated": train.truncated,
            "valid_truncated": valid.truncated,
            "estimated_optimizer_steps": estimated_optimizer_steps,
            "warmup_steps": warmup_steps,
        },
    )
    print(f"Saved final PEFT adapter to {final_adapter_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
