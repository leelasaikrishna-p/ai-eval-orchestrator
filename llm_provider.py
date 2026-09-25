"""
Pluggable LLM provider layer.

Today: Ollama (local, free, no API key, no billing account).
Planned: Gemini, Groq, Anthropic -- each is a drop-in class implementing
the same `generate(prompt) -> str` interface, registered in `get_provider()`
below. Nothing else in the project (the four eval tools, the orchestrator)
needs to know or care which provider is active.

Swap providers via the PROVIDER env var (defaults to "ollama"), e.g.:
    PROVIDER=gemini python tools/llm_judge.py
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Return the model's raw text response to `prompt`."""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Local inference via Ollama's REST API. No API key required.

    Setup:
        brew install ollama      # or download from https://ollama.com
        ollama serve             # runs in the background on :11434
        ollama pull llama3.1:8b  # one-time model download (~4.7GB)
    """

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("response", "").strip()
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}.\n"
                f"Is it running? Try: `ollama serve` (in another terminal), "
                f"and make sure the model is pulled: `ollama pull {self.model}`.\n"
                f"Original error: {e}"
            ) from e


class MockProvider(LLMProvider):
    """Deterministic canned responses for tests -- no network, no model.

    Lets the orchestration logic (JSON parsing, branching, error handling)
    be verified without Ollama installed or running.
    """

    def __init__(self, response: str = '{"score": 0.9, "reasoning": "mock"}'):
        self.response = response
        self.calls: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        return self.response


def get_provider(name: str | None = None) -> LLMProvider:
    """Factory -- selects a provider by name (or the PROVIDER env var)."""
    name = (name or os.environ.get("PROVIDER", "ollama")).lower()
    if name == "ollama":
        return OllamaProvider()
    if name == "mock":
        return MockProvider()
    # Future providers -- add the class above, then register here:
    # if name == "gemini":
    #     from providers.gemini_provider import GeminiProvider
    #     return GeminiProvider()
    # if name == "groq":
    #     from providers.groq_provider import GroqProvider
    #     return GroqProvider()
    # if name == "anthropic":
    #     from providers.anthropic_provider import AnthropicProvider
    #     return AnthropicProvider()
    raise ValueError(f"Unknown provider: {name!r}. Available now: ollama, mock.")


def extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in prose or code fences -- pull out the
    first {...} block and parse it, raising a clear error if none is found."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output:\n{text!r}")
    return json.loads(text[start : end + 1])
