#!/usr/bin/env python3
"""Generate resumable, single-shot Qwen repairs with Transformers and PEFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "models/Qwen3-4B-Instruct-2507"
DEFAULT_INPUT = Path("data/sft/test.jsonl")
DEFAULT_OUTPUT = Path("runs/zero_shot/predictions.jsonl")
DEFAULT_MAX_TOKENS = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run greedy Qwen proof repair over the APRIL test prompts."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local Transformers model directory or Hugging Face Hub ID",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="Optional PEFT adapter directory for the fine-tuned run",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-tokens",
        type=int,
        choices=(1024, 2048),
        default=DEFAULT_MAX_TOKENS,
        help="Maximum new tokens per repair (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Torch device (default: choose CUDA, then MPS, then CPU)",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Model weight dtype (default: model configuration)",
    )
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            selected_id = value.get("selected_id")
            if not isinstance(selected_id, str) or not selected_id:
                raise ValueError(f"{path}:{line_number}: missing selected_id")
            if selected_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate selected_id {selected_id}")
            seen.add(selected_id)
            rows.append(value)
    return rows


def inference_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Return only system/user messages; never expose the assistant gold target."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{row['selected_id']}: messages must be a list")

    prompt_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"{row['selected_id']}: invalid message")
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            continue
        if role not in {"system", "user"} or not isinstance(content, str):
            raise ValueError(f"{row['selected_id']}: invalid prompt message")
        prompt_messages.append({"role": role, "content": content})

    if [message["role"] for message in prompt_messages] != ["system", "user"]:
        raise ValueError(
            f"{row['selected_id']}: expected exactly one system and one user prompt"
        )
    return prompt_messages


def strip_obvious_markdown_fence(raw_response: str) -> tuple[str, str]:
    """Remove a single outer Markdown fence, but do not rewrite the response."""
    stripped = raw_response.strip()
    lines = stripped.splitlines()
    if len(lines) >= 2:
        opening = lines[0].strip().lower()
        closing = lines[-1].strip()
        if opening in {"```", "```lean", "```lean4"} and closing == "```":
            candidate = "\n".join(lines[1:-1]).strip()
            return (candidate + "\n" if candidate else "", "outer_markdown_fence")
    return (stripped + "\n" if stripped else "", "none")


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["selected_id"]: row for row in read_jsonl(path)}


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_or_check_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"Run configuration differs from {path}; use a new output directory"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested, but CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("--device mps requested, but MPS is unavailable")
    return requested


def resolve_dtype(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[requested]


def main() -> int:
    args = parse_args()
    local_model = Path(args.model)
    if args.model.startswith((".", "/", "models/")) and not local_model.is_dir():
        raise FileNotFoundError(local_model)
    if args.adapter_path is not None and not args.adapter_path.is_dir():
        raise FileNotFoundError(args.adapter_path)

    rows = read_jsonl(args.input)
    if len(rows) != 100:
        raise ValueError(f"Expected exactly 100 test prompts, found {len(rows)}")
    prompt_messages = {
        row["selected_id"]: inference_messages(row) for row in rows
    }

    config = {
        "backend": "transformers_peft",
        "model": args.model,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "device": args.device,
        "dtype": args.dtype,
        "decoding": "greedy_argmax",
        "num_samples": 1,
        "temperature": 0.0,
        "max_new_tokens": args.max_tokens,
        "chat_template": "model_tokenizer",
        "enable_thinking": False,
        "assistant_gold_in_prompt": False,
    }
    write_or_check_config(args.output.parent / "generation_config.json", config)

    completed = load_completed(args.output)
    input_ids = {row["selected_id"] for row in rows}
    unknown = set(completed) - input_ids
    if unknown:
        raise ValueError(f"Predictions contain IDs absent from test set: {sorted(unknown)}")
    pending = [row for row in rows if row["selected_id"] not in completed]
    print(f"Resuming with {len(completed)}/100 complete; {len(pending)} remaining", flush=True)
    if not pending:
        print(f"All predictions already exist in {args.output}", flush=True)
        return 0

    # Import the large ML stack only after validating inputs and resume state.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)
    print(f"Loading Transformers model {args.model} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if args.adapter_path is not None:
        from peft import PeftModel

        print(f"Loading PEFT adapter from {args.adapter_path}...", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for index, row in enumerate(rows, start=1):
        selected_id = row["selected_id"]
        if selected_id in completed:
            print(f"[{index}/100] {selected_id}: cached", flush=True)
            continue

        messages = prompt_messages[selected_id]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]

        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        generated_tokens = generated[0, prompt_length:]
        raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        candidate, extraction = strip_obvious_markdown_fence(raw_response)
        prediction = {
            "selected_id": selected_id,
            "src_hash": row.get("src_hash"),
            "raw_response": raw_response,
            "candidate": candidate,
            "extraction": extraction,
            "prompt_sha256": sha256_text(
                json.dumps(messages, ensure_ascii=False, sort_keys=True)
            ),
            "candidate_sha256": sha256_text(candidate),
            "max_new_tokens": args.max_tokens,
            "decoding": "greedy_argmax",
            "backend": "transformers_peft",
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "elapsed_seconds": round(elapsed, 6),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(args.output, prediction)
        completed[selected_id] = prediction
        print(
            f"[{index}/100] {selected_id}: saved "
            f"({len(candidate)} chars, {elapsed:.1f}s)",
            flush=True,
        )

    print(f"Saved 100/100 predictions to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
