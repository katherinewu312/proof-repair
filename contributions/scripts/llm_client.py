"""External language-model adapters used by mutation pipelines."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_HUGGINGFACE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
PAPER_MODEL = "deepseek-ai/DeepSeek-V3-0324"
DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1"


def sanitize_endpoint(base_url: str | None) -> str:
    """Return endpoint provenance without credentials, queries, or fragments."""
    value = urlsplit(base_url or DEFAULT_OPENAI_ENDPOINT)
    hostname = value.hostname or ""
    if value.port:
        hostname += f":{value.port}"
    return urlunsplit((value.scheme, hostname, value.path, "", ""))


class OpenAIModel:
    """Responses API adapter implementing ``mutate_line.CompletionModel``."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        reasoning_effort: str = "medium",
        timeout: float = 180.0,
        retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("install requirements.txt to use the model API") from error
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"required API key environment variable {api_key_env} is unset")
        self.model = model
        self.reasoning_effort = reasoning_effort
        kwargs = {"api_key": api_key, "timeout": timeout, "max_retries": retries}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def complete(self, messages: Sequence[Mapping[str, str]], max_tokens: int) -> str:
        return self._create(input=list(messages), max_output_tokens=max_tokens)

    def complete_json(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        max_tokens: int,
        *,
        name: str = "response",
    ) -> str:
        """Structured output; returns the raw JSON text."""
        return self._create(
            input=list(messages),
            max_output_tokens=max_tokens,
            text={"format": {"type": "json_schema", "name": name, "schema": dict(schema), "strict": True}},
        )

    def _create(self, **request: Any) -> str:
        try:
            response = self.client.responses.create(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                **request,
            )
        except Exception as error:
            # Surface only stable API metadata. Avoid dumping response headers or
            # request details, which do not belong in pipeline logs.
            details = [type(error).__name__]
            status = getattr(error, "status_code", None)
            code = getattr(error, "code", None)
            if status is not None:
                details.append(f"status={status}")
            if code:
                details.append(f"code={code}")
            raise RuntimeError(f"model API request failed ({', '.join(details)})") from error
        content = response.output_text
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("model returned an empty completion")
        return content


class HuggingFaceLocalModel:
    """Run an instruction-tuned Hugging Face model in the current process."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_HUGGINGFACE_MODEL,
        device: str = "auto",
        revision: str = "main",
    ) -> None:
        try:
            import torch
            from huggingface_hub import try_to_load_from_cache
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "install torch, transformers, and accelerate to use a local model"
            ) from error

        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        if device not in {"mps", "cuda", "cpu"}:
            raise ValueError(f"unsupported Hugging Face device: {device}")

        dtype = {
            "mps": torch.float16,
            "cuda": torch.bfloat16,
            "cpu": torch.float32,
        }[device]
        self.model_name = model
        self.device = device
        self.requested_revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        cached_config = try_to_load_from_cache(model, "config.json", revision=revision)
        self.revision = (
            Path(cached_config).parent.name
            if isinstance(cached_config, str)
            else getattr(self.model.config, "_commit_hash", None)
        )

    def complete(self, messages: Sequence[Mapping[str, str]], max_tokens: int) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_tokens = encoded["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        content = self.tokenizer.decode(
            generated[0, input_tokens:],
            skip_special_tokens=True,
        ).strip()
        if not content:
            raise RuntimeError("local Hugging Face model returned an empty completion")
        return content
