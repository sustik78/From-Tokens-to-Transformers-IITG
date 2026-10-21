"""Small OpenAI-compatible LLM client with retries and a free fallback.

No key is required for the default `llm7` and `pollinations` providers.  The
provider is fully configurable in `.env`; hosted providers use LLM_API_KEY.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from . import config

PROVIDER_PROFILES = {
    "llm7": {
        "base_url": "https://api.llm7.io/v1",
        "default_model": "DeepSeek-V4-Flash-0731",
        "needs_key": False,
        "label": "LLM7.io (free, no key)",
    },
    "pollinations": {
        "base_url": "https://text.pollinations.ai/openai",
        "default_model": "openai-fast",
        "needs_key": False,
        "label": "Pollinations (free, no key)",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "needs_key": True,
        "label": "OpenAI",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "needs_key": True,
        "label": "Groq",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.0-flash",
        "needs_key": True,
        "label": "Google Gemini (OpenAI-compatible)",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "needs_key": True,
        "label": "OpenRouter",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen2.5:0.5b-instruct",
        "needs_key": False,
        "label": "Ollama (local)",
    },
    "none": {
        "base_url": "",
        "default_model": "",
        "needs_key": False,
        "label": "Disabled",
    },
}


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict]
    provider: str
    model: str
    raw: dict | None = None


def _profile(provider: str) -> dict:
    return PROVIDER_PROFILES.get(provider, PROVIDER_PROFILES["llm7"])


def _endpoint(provider: str) -> str:
    explicit = config.LLM_BASE_URL
    if provider == config.LLM_PROVIDER and explicit:
        base = explicit
    else:
        base = _profile(provider)["base_url"]
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _model(provider: str) -> str:
    if provider == config.LLM_PROVIDER and config.LLM_MODEL:
        return config.LLM_MODEL
    if provider == config.LLM_FALLBACK_PROVIDER and config.LLM_FALLBACK_MODEL:
        return config.LLM_FALLBACK_MODEL
    return _profile(provider)["default_model"]


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        temperature: float | None = None,
    ):
        self.primary = (provider or config.LLM_PROVIDER or "llm7").lower()
        self.model = model or config.LLM_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT
        self.temperature = (
            config.LLM_TEMPERATURE if temperature is None else temperature
        )
        self.fallback = config.LLM_FALLBACK_PROVIDER
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return self.primary != "none"

    def _headers(self, provider: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = config.LLM_API_KEY
        if key:
            headers["Authorization"] = f"Bearer {key}"
        elif provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/"
            headers["X-Title"] = "VoiceOps 2026"
        return headers

    def _call(
        self,
        provider: str,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        profile = _profile(provider)
        if provider == "none":
            raise LLMError("LLM provider is disabled")
        if profile["needs_key"] and not config.LLM_API_KEY:
            raise LLMError(f"{profile['label']} requires LLM_API_KEY in .env")

        payload: dict[str, Any] = {
            "model": _model(provider),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = requests.post(
            _endpoint(provider),
            headers=self._headers(provider),
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code in (429, 500, 502, 503, 504):
            raise LLMError(f"{provider} HTTP {response.status_code}: {response.text[:300]}")
        if response.status_code >= 400:
            raise LLMError(f"{provider} HTTP {response.status_code}: {response.text[:500]}")
        body = response.json()
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Malformed LLM response: {body}") from exc
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        return LLMResponse(
            content=content.strip(),
            tool_calls=tool_calls,
            provider=provider,
            model=str(body.get("model") or _model(provider)),
            raw=body,
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 500,
        temperature: float | None = None,
        json_mode: bool = False,
        allow_fallback: bool = True,
    ) -> LLMResponse:
        providers = [self.primary]
        if allow_fallback and self.fallback and self.fallback not in providers:
            providers.append(self.fallback)

        last_error: Exception | None = None
        for provider in providers:
            delays = [0, 2, 6]
            for attempt, delay in enumerate(delays):
                if delay:
                    time.sleep(delay)
                try:
                    return self._call(
                        provider,
                        messages,
                        tools,
                        max_tokens,
                        self.temperature if temperature is None else temperature,
                        json_mode,
                    )
                except Exception as exc:  # noqa: BLE001 - provider errors are varied
                    last_error = exc
                    self.last_error = str(exc)
                    if "requires LLM_API_KEY" in str(exc) or provider == "none":
                        break
        raise LLMError(str(last_error or "No LLM provider available"))

    def chat_text(
        self,
        system: str,
        user: str,
        max_tokens: int = 400,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        response = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
        return response.content


def parse_json_object(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def get_llm() -> LLMClient:
    return LLMClient()


# Imported lazily in parse_json_object to keep the module import cheap.
import re  # noqa: E402
