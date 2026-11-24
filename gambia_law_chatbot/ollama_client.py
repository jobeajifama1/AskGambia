"""Thin client for the local Ollama HTTP API."""

from __future__ import annotations

import logging
from typing import Any

import requests

from .config import GENERATE_TIMEOUT, MAX_EMBED_CHARS, REQUEST_TIMEOUT, OllamaUnavailableError, Config

LOGGER = logging.getLogger("gambia_law_chatbot.ollama")


class OllamaClient:
    """Sends generation and embedding requests to a local Ollama server."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def _post(self, url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(f"Cannot reach Ollama at {self.config.ollama_url}") from exc
        except requests.RequestException as exc:
            raise OllamaUnavailableError(f"Ollama request failed: {exc}") from exc

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Return the model's answer for a prompt."""
        data = self._post(
            self.config.generate_url,
            {
                "model": self.config.chat_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": 8192},
            },
            GENERATE_TIMEOUT,
        )
        return data.get("response", "").strip()

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a piece of text."""
        data = self._post(
            self.config.embeddings_url,
            {"model": self.config.embedding_model, "prompt": text[:MAX_EMBED_CHARS]},
            REQUEST_TIMEOUT,
        )
        return data.get("embedding", [])


def verify_ollama(config: Config) -> None:
    """Fail fast if Ollama is down or a required model has not been pulled."""
    try:
        response = requests.get(config.tags_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {config.ollama_url}. Install it from "
            "https://ollama.com/download and run: ollama serve"
        ) from exc

    installed = {model["name"] for model in response.json().get("models", [])}

    def has(model_name: str) -> bool:
        base = model_name.split(":")[0]
        return any(name == model_name or name.startswith(base + ":") for name in installed)

    if not has(config.chat_model):
        raise OllamaUnavailableError(f"Chat model missing. Run: ollama pull {config.chat_model}")
    if not has(config.embedding_model):
        raise OllamaUnavailableError(f"Embedding model missing. Run: ollama pull {config.embedding_model}")
