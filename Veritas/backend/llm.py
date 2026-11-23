"""Thin Ollama client: structured-JSON chat and embeddings. Nothing here calls a paid API."""
from __future__ import annotations

import json
import re

import numpy as np
import requests

from . import config


class LLMError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: int):
    try:
        r = requests.post(f"{config.OLLAMA_HOST}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise LLMError(f"Ollama call failed ({config.OLLAMA_HOST}{path}): {e}") from e


def parse_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    raise LLMError("Model did not return valid JSON")


def chat_json(system: str, user: str, schema: dict | None = None, retries: int = 1):
    """Deterministic (temperature 0) chat that must return JSON. `think:false` skips the
    reasoning phase of Qwen3.5 - much faster, and the schema keeps the output well-formed."""
    payload = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "format": schema or "json",
        "options": {"temperature": 0, "num_ctx": config.NUM_CTX},
    }
    last: Exception | None = None
    for _ in range(retries + 1):
        data = _post("/api/chat", payload, config.LLM_TIMEOUT)
        try:
            return parse_json(data["message"]["content"])
        except (LLMError, KeyError) as e:
            last = e
    raise LLMError(str(last))


def embed(texts: list[str], batch: int = 16) -> np.ndarray:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        data = _post("/api/embed", {"model": config.EMBED_MODEL, "input": texts[i:i + batch]}, 600)
        out.extend(data["embeddings"])
    arr = np.array(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-9, None)


def health() -> dict:
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException as e:
        return {"ok": False, "error": str(e), "host": config.OLLAMA_HOST}
    def has(want: str) -> bool:
        return any(n == want or n.split(":")[0] == want for n in names) if ":" not in want else want in names

    return {"ok": True, "host": config.OLLAMA_HOST, "llm": config.LLM_MODEL, "embed": config.EMBED_MODEL,
            "llm_present": has(config.LLM_MODEL),
            "embed_present": has(config.EMBED_MODEL), "models": names}
