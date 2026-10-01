"""
Pluggable LLM provider layer.

Today: Ollama (local, free, no API key) and Gemini (Google's REST API, free
tier, needs an API key).
Planned: Groq, Anthropic -- each is a drop-in class implementing the same
`generate(prompt) -> str` interface (plus `chat()` for the orchestrator),
registered in `get_provider()` below. Nothing else in the project (the four
eval tools, the orchestrator) needs to know or care which provider is active.

Swap providers via the PROVIDER env var (defaults to "ollama"), e.g.:
    PROVIDER=gemini python tools/llm_judge.py

`chat()` speaks one message format everywhere -- Ollama's (role/content/
tool_calls). Providers with a different wire format translate at the edge,
so the orchestrator never branches on provider.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name = "base"

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

    name = "ollama"

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

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.0) -> dict:
        """Native tool-calling via Ollama's /api/chat. Returns the raw
        `message` dict -- may contain `content` (plain text) and/or
        `tool_calls` (a list of {id, function: {name, arguments}})."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["message"]
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it running? Original error: {e}"
            ) from e


class MockProvider(LLMProvider):
    """Deterministic canned responses for tests -- no network, no model.

    Lets the orchestration logic (JSON parsing, branching, error handling)
    be verified without Ollama installed or running.
    """

    name = "mock"

    def __init__(self, response: str = '{"score": 0.9, "reasoning": "mock"}'):
        self.response = response
        self.calls: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        return self.response


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_RETRYABLE = (429, 500, 503)


class GeminiProvider(LLMProvider):
    """Google Gemini via its REST API -- standard library only, no SDK.

    Setup:
        1. Create an API key in Google AI Studio: https://aistudio.google.com/apikey
        2. export GEMINI_API_KEY=...   # never commit it -- .env is gitignored
        3. PROVIDER=gemini python3 orchestrator.py

    GEMINI_MODEL picks the model (default below); list what your key can use:
        curl -H "x-goog-api-key: $GEMINI_API_KEY" \\
             https://generativelanguage.googleapis.com/v1beta/models
    """

    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey, "
                "then `export GEMINI_API_KEY=...` in the shell you run this from."
            )

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        body = self._post({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        })
        return from_gemini_response(body)["content"].strip()

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.0) -> dict:
        request = to_gemini_request(messages, tools)
        request["generationConfig"] = {"temperature": temperature}
        return from_gemini_response(self._post(request))

    def _post(self, payload: dict, retries: int = 5) -> dict:
        # Key goes in a header, not the URL, so it never shows up in logs or tracebacks.
        url = f"{GEMINI_API_BASE}/models/{self.model}:generateContent"
        data = json.dumps(payload).encode("utf-8")
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                if e.code in GEMINI_RETRYABLE and attempt < retries:
                    # Rate limits (429) are per minute and "high demand" (503) spikes
                    # can last a while: honor Retry-After if sent, else back off
                    # 5s, 10s, 20s, 40s, 60s (~2 min total).
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else min(60, 5 * 2**attempt)
                    print(f"  (Gemini {e.code}, retrying in {wait:.0f}s -- attempt {attempt + 1}/{retries})", file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Gemini API error {e.code} (model {self.model}): {detail[:500]}") from e
            except urllib.error.URLError as e:
                raise RuntimeError(f"Could not reach the Gemini API: {e}") from e
        raise RuntimeError("unreachable")


# Gemini accepts an OpenAPI subset for function parameters; MCP's generated
# JSON Schema carries extra keys (e.g. "title") it rejects, so keep only these.
_GEMINI_SCHEMA_KEYS = {"type", "description", "properties", "required", "items", "enum", "format", "nullable"}


def _gemini_schema(schema: dict) -> dict:
    out = {}
    for key, value in schema.items():
        if key not in _GEMINI_SCHEMA_KEYS:
            continue
        if key == "properties":
            out[key] = {prop: _gemini_schema(sub) for prop, sub in value.items()}
        elif key == "items" and isinstance(value, dict):
            out[key] = _gemini_schema(value)
        else:
            out[key] = value
    return out


def to_gemini_request(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Translate Ollama-style chat messages + tool schemas into a Gemini
    generateContent request. Pure function -- unit-tested offline."""
    system_parts: list[dict] = []
    contents: list[dict] = []
    pending_call_names: list[str] = []

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_parts.append({"text": msg.get("content", "")})
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": msg.get("content", "")}]})
        elif role == "assistant":
            calls = msg.get("tool_calls") or []
            pending_call_names = [c["function"]["name"] for c in calls]
            # Replay Gemini's own parts verbatim when we have them: they can carry
            # thought signatures that multi-turn function calling needs back.
            parts = msg.get("_gemini_parts")
            if not parts:
                parts = [{"text": msg["content"]}] if msg.get("content") else []
                parts += [
                    {"functionCall": {"name": c["function"]["name"], "args": c["function"].get("arguments") or {}}}
                    for c in calls
                ]
            contents.append({"role": "model", "parts": parts})
        elif role == "tool":
            name = msg.get("tool_name") or (pending_call_names[0] if pending_call_names else "unknown_tool")
            if name in pending_call_names:
                pending_call_names.remove(name)
            try:
                response = json.loads(msg.get("content") or "{}")
            except json.JSONDecodeError:
                response = {"result": msg.get("content")}
            if not isinstance(response, dict):
                response = {"result": response}
            part = {"functionResponse": {"name": name, "response": response}}
            # Results for one batch of calls go back together in a single turn.
            last = contents[-1] if contents else None
            if last and last["role"] == "user" and all("functionResponse" in p for p in last["parts"]):
                last["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})

    request: dict = {"contents": contents}
    if system_parts:
        request["systemInstruction"] = {"parts": system_parts}
    if tools:
        request["tools"] = [{
            "functionDeclarations": [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "parameters": _gemini_schema(t["function"].get("parameters") or {"type": "object", "properties": {}}),
                }
                for t in tools
            ]
        }]
    return request


def from_gemini_response(body: dict) -> dict:
    """Translate a Gemini generateContent response into an Ollama-style
    assistant message ({role, content, tool_calls}). Pure function."""
    candidates = body.get("candidates") or []
    if not candidates:
        reason = (body.get("promptFeedback") or {}).get("blockReason", "no candidates returned")
        raise RuntimeError(f"Gemini returned no answer: {reason}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
    calls = [
        {"function": {"name": p["functionCall"]["name"], "arguments": p["functionCall"].get("args") or {}}}
        for p in parts
        if "functionCall" in p
    ]
    message = {"role": "assistant", "content": text, "_gemini_parts": parts}
    if calls:
        message["tool_calls"] = calls
    return message


def get_provider(name: str | None = None) -> LLMProvider:
    """Factory -- selects a provider by name (or the PROVIDER env var)."""
    name = (name or os.environ.get("PROVIDER", "ollama")).lower()
    if name == "ollama":
        return OllamaProvider()
    if name == "gemini":
        return GeminiProvider()
    if name == "mock":
        response = os.environ.get("MOCK_RESPONSE")
        return MockProvider(response) if response else MockProvider()
    # Future providers -- add the class above, then register here (groq, anthropic).
    raise ValueError(f"Unknown provider: {name!r}. Available now: ollama, gemini, mock.")


def extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in prose or code fences -- pull out the
    first {...} block and parse it, raising a clear error if none is found."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output:\n{text!r}")
    return json.loads(text[start : end + 1])
