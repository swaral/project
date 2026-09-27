"""Small LLM client adapters used by the Step 4 baseline runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class LLMResponse:
    """Raw model output and request metadata."""

    raw_text: str
    model: str
    request_id: str | None
    latency_ms: float | None
    usage: dict[str, object] | None


class LLMRequestError(RuntimeError):
    """Raised when an LLM request cannot produce a usable response."""


class StubLLMClient:
    """Deterministic no-network client for testing the full data path.

    It returns candidates in their presented order. This is intentionally not
    a recommendation model; it only verifies prompt rendering, parsing, and
    trial logging without requiring an API key.
    """

    model = "stub-v1"

    @property
    def config(self) -> dict[str, object]:
        return {
            "provider": "stub",
            "model": self.model,
            "temperature": 0.0,
            "top_p": 1.0,
        }

    def generate(self, system_message: str, user_message: str) -> LLMResponse:
        del system_message
        started = time.perf_counter()
        marker_match = re.search(
            r"Candidate positions to rank(?: \(exactly \d+ candidates\))?:\n",
            user_message,
        )
        if marker_match is None:
            raise LLMRequestError("stub could not locate the candidate section")
        candidate_section = user_message[marker_match.end() :].split(
            "\n\nRules:", 1
        )[0]
        candidate_positions = [
            int(match)
            for match in re.findall(r"^(\d+)\. title=", candidate_section, re.MULTILINE)
        ]
        if not candidate_positions:
            raise LLMRequestError("stub could not parse candidate positions")
        # Give earlier presented candidates higher scores so the stub ranks
        # positions in their original order after deterministic sorting.
        raw_text = json.dumps(
            {"scores": list(range(len(candidate_positions), 0, -1))}
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return LLMResponse(
            raw_text=raw_text,
            model=self.model,
            request_id=None,
            latency_ms=latency_ms,
            usage=None,
        )


class ChatCompletionsClient:
    """Client for APIs exposing an OpenAI-compatible chat-completions route.

    The HTTP implementation uses only the Python standard library. Set a
    provider-specific base URL, model, and API key through the runner's CLI or
    environment variables. Local OpenAI-compatible servers may omit the key.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
        timeout_seconds: int = 120,
        json_mode: bool = True,
        candidate_count: int = 20,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")

        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.json_mode = json_mode
        self.candidate_count = candidate_count

    @staticmethod
    def _scores_schema(candidate_count: int) -> dict[str, object]:
        """Return the strict score schema used by Ollama structured outputs."""

        return {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 100},
                    "minItems": candidate_count,
                    "maxItems": candidate_count,
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        }

    @property
    def config(self) -> dict[str, object]:
        return {
            "provider": "chat-completions-compatible",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
            "structured_output": (
                "candidate_scores_schema" if self.json_mode else None
            ),
            "candidate_count": self.candidate_count,
        }

    @staticmethod
    def _content(choice: Mapping[str, object]) -> str:
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise LLMRequestError("response choice has no message object")
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, Mapping)
            ]
            return "".join(text_parts)
        raise LLMRequestError("response message has no text content")

    def generate(self, system_message: str, user_message: str) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "candidate_scores",
                    "strict": True,
                    "schema": self._scores_schema(self.candidate_count),
                },
            }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise LLMRequestError(f"LLM HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise LLMRequestError(f"LLM connection failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise LLMRequestError("LLM returned a non-JSON API response") from exc

        if not isinstance(response_payload, Mapping):
            raise LLMRequestError("LLM API response must be a JSON object")
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMRequestError("LLM API response contains no choices")
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise LLMRequestError("LLM API choice is not an object")

        usage = response_payload.get("usage")
        usage_dict = dict(usage) if isinstance(usage, Mapping) else None
        return LLMResponse(
            raw_text=self._content(first_choice),
            model=str(response_payload.get("model", self.model)),
            request_id=(
                str(response_payload["id"])
                if response_payload.get("id") is not None
                else None
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
            usage=usage_dict,
        )


def create_client(
    provider: str,
    *,
    model: str | None,
    base_url: str | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_seconds: int,
    json_mode: bool,
    candidate_count: int,
):
    if provider == "stub":
        return StubLLMClient()

    if provider == "openrouter":
        model = model or os.environ.get("OPENROUTER_MODEL")
        if not model:
            raise ValueError("Set OPENROUTER_MODEL or pass --model for OpenRouter")
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY for OpenRouter authentication")
        base_url = (
            base_url
            or os.environ.get("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
    elif provider == "chat-completions":
        model = model or os.environ.get("LLM_MODEL", "qwen2.5:3b-instruct")
        api_key = os.environ.get("LLM_API_KEY")
        base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
        )
    else:
        raise ValueError(f"Unknown provider {provider!r}")

    return ChatCompletionsClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        json_mode=json_mode,
        candidate_count=candidate_count,
    )
