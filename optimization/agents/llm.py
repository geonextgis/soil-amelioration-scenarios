#!/usr/bin/env python
"""Local LLM backends. Nothing here leaves the machine.

Ollama is reached over its HTTP API with the standard library only — no
``ollama`` package, no ``requests``. That is deliberate: the cluster nodes this
runs on have no outbound network, so a dependency that has to be installed is a
dependency that will not be there. urllib is.

    backend = build_backend(cfg["llm"], agent="lai")
    ok, detail = backend.available()
    reply = backend.json_chat([...])       # dict, or LLMError

Two backends:

  ``OllamaBackend``  the real thing, POST /api/chat with format=json.
  ``MockBackend``    scripted replies. Used by the test suite to exercise the
                     whole agent loop — context building, JSON parsing, the
                     repair round-trip, validation, the ledger — without a model
                     server. It is a test double, not a fallback: a missing
                     Ollama is reported as an error, never silently mocked.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

DEFAULT_HOST = "http://localhost:11434"


def normalise_host(host: str) -> str:
    """``OLLAMA_HOST`` is conventionally bare (``127.0.0.1:11434``); urllib needs a scheme."""
    host = (host or DEFAULT_HOST).strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


class LLMError(RuntimeError):
    """The model could not be reached, or did not answer usefully."""


class LLMUnavailable(LLMError):
    """No local model server is reachable."""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull one JSON object out of a model reply.

    Even with ``format=json`` a local model will sometimes wrap the object in a
    code fence or prepend a sentence of commentary. Rather than fail the whole
    iteration on that, take the outermost balanced ``{...}``.
    """
    text = (text or "").strip()
    if not text:
        raise LLMError("model returned an empty reply")

    for candidate in (text, *(m.group(1).strip() for m in _FENCE.finditer(text))):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    start = text.find("{")
    if start >= 0:
        depth, in_string, escaped = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break

    raise LLMError(f"no JSON object in the reply (first 300 chars): {text[:300]!r}")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
@dataclass
class OllamaBackend:
    """Chat against a local Ollama server."""

    model: str
    host: str = DEFAULT_HOST
    temperature: float = 0.2
    num_ctx: int = 32768
    timeout: float = 900.0
    seed: int | None = 7  # fixed by default: a calibration decision should repeat
    # Reasoning models (qwen3.x, deepseek-r1, …) emit a separate `thinking` field
    # and, with a large prompt, spend the entire generation budget on it — leaving
    # `content` empty or a truncated fragment of JSON. The chain of thought buys
    # nothing here: the schema already demands `analysis` and `reasoning`, which
    # land in the *output*, where the ledger can record them. So thinking is off.
    think: bool = False
    # Guarantees room for the answer instead of letting the prompt consume it all.
    num_predict: int = 2048

    name: str = field(default="ollama", init=False)

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        url = f"{self.host.rstrip('/')}{path}"
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            raise LLMError(f"Ollama {path} returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMUnavailable(f"cannot reach Ollama at {self.host}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMError(
                f"Ollama did not answer within {timeout or self.timeout:.0f}s. A cold "
                f"local model can exceed this — raise llm.request_timeout.") from exc

    def _get(self, path: str, timeout: float = 10.0) -> dict:
        url = f"{self.host.rstrip('/')}{path}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.URLError as exc:
            raise LLMUnavailable(f"cannot reach Ollama at {self.host}: {exc.reason}") from exc

    def models(self) -> list[str]:
        return sorted(m.get("name", "") for m in self._get("/api/tags").get("models", []))

    def available(self) -> tuple[bool, str]:
        """``(reachable and model present, human-readable detail)``. Never raises."""
        try:
            installed = self.models()
        except LLMError as exc:
            return False, str(exc)
        if not installed:
            return False, f"Ollama is running at {self.host} but has no models pulled"
        # Ollama tags carry an explicit ":latest" that users normally omit.
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if self.model in installed or wanted in installed:
            return True, f"{self.model} available at {self.host}"
        return False, (f"model {self.model!r} is not pulled on {self.host}. "
                       f"Available: {', '.join(installed)}. "
                       f"Pull it with: ollama pull {self.model}")

    def chat(self, messages: list[dict], json_mode: bool = True) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict},
        }
        if self.seed is not None:
            payload["options"]["seed"] = self.seed
        if json_mode:
            payload["format"] = "json"

        response = self._post("/api/chat", payload)
        message = response.get("message") or {}
        content = message.get("content") or ""
        thinking = message.get("thinking") or ""
        done_reason = response.get("done_reason")

        if content:
            if done_reason == "length":
                # Generation hit the ceiling: whatever came back is a fragment, and
                # letting it through produces the useless "no JSON object" error one
                # layer up instead of naming the actual cause.
                raise LLMError(
                    f"the reply was cut off after {response.get('eval_count', '?')} tokens "
                    f"(done_reason=length). The prompt is using most of the {self.num_ctx}-token "
                    f"window. Raise llm.num_ctx, or lower llm.history_iterations to shorten "
                    f"the prompt.")
            return content

        # Empty content. If the model reasoned itself out of a budget, the JSON is
        # sometimes still in the thinking text — worth a look before giving up.
        if thinking:
            try:
                return json.dumps(extract_json(thinking))
            except LLMError:
                pass
            raise LLMError(
                f"the model produced {len(thinking)} characters of `thinking` and no answer. "
                f"It is a reasoning model and thinking is consuming the generation budget — "
                f"set llm.think: false (and/or raise llm.num_ctx above {self.num_ctx}).")

        raise LLMError(f"Ollama returned no content: {json.dumps(response)[:300]}")

    def json_chat(self, messages: list[dict]) -> dict:
        return extract_json(self.chat(messages, json_mode=True))


# ---------------------------------------------------------------------------
# Mock (tests only)
# ---------------------------------------------------------------------------
@dataclass
class MockBackend:
    """Scripted replies, in order. A callable gets the message list instead."""

    replies: list[str] | Callable[[list[dict]], str]
    model: str = "mock"
    calls: list[list[dict]] = field(default_factory=list)

    name: str = field(default="mock", init=False)

    def available(self) -> tuple[bool, str]:
        return True, "mock backend (no model server involved)"

    def models(self) -> list[str]:
        return ["mock"]

    def chat(self, messages: list[dict], json_mode: bool = True) -> str:
        self.calls.append(messages)
        if callable(self.replies):
            return self.replies(messages)
        if not self.replies:
            raise LLMError("MockBackend ran out of scripted replies")
        return self.replies.pop(0)

    def json_chat(self, messages: list[dict]) -> dict:
        return extract_json(self.chat(messages))


# ---------------------------------------------------------------------------
def build_backend(llm_cfg: dict, agent: str, model: str | None = None,
                  replies=None):
    """Backend for one agent, from the ``llm:`` block of calibration.yaml.

    ``OLLAMA_HOST`` overrides the configured host — that is the variable the
    Ollama CLI itself uses, so a machine that already points somewhere else keeps
    working without editing the repo.
    """
    llm_cfg = llm_cfg or {}
    provider = str(llm_cfg.get("provider", "ollama")).lower()
    per_agent = (llm_cfg.get("agents") or {}).get(agent) or {}

    if provider == "mock":
        return MockBackend(replies=replies if replies is not None else [])
    if provider != "ollama":
        raise LLMError(f"unknown llm provider {provider!r}; expected 'ollama' or 'mock'")

    chosen = model or per_agent.get("model")
    if not chosen:
        raise LLMError(
            f"no model configured for agent {agent!r}. Add it under llm.agents.{agent}.model "
            f"in calibration.yaml, or pass --model.")

    return OllamaBackend(
        model=chosen,
        host=normalise_host(os.environ.get("OLLAMA_HOST") or llm_cfg.get("host", DEFAULT_HOST)),
        temperature=float(per_agent.get("temperature", 0.2)),
        num_ctx=int(llm_cfg.get("num_ctx", 32768)),
        timeout=float(llm_cfg.get("request_timeout", 900)),
        seed=per_agent.get("seed", llm_cfg.get("seed", 7)),
        think=bool(per_agent.get("think", llm_cfg.get("think", False))),
        num_predict=int(per_agent.get("num_predict", llm_cfg.get("num_predict", 2048))),
    )
