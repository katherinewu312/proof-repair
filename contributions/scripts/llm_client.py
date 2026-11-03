"""External language-model adapters used by mutation pipelines."""
from __future__ import annotations

import os
from typing import Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


DEFAULT_OPENAI_MODEL = "gpt-6-luna"
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
        try:
            response = self.client.responses.create(
                model=self.model,
                input=list(messages),
                max_output_tokens=max_tokens,
                reasoning={"effort": self.reasoning_effort},
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
