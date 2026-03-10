from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    backend: str
    model: str
    base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding-v3/invoke"
    vllm_base_url: str = "http://10.20.49.150:8001/v1"
    api_key_env: str = "OPENAI_API_KEY"


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._local_model: SentenceTransformer | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.config.backend == "dashscope":
            return self._dashscope_embed(texts)
        if self.config.backend == "vllm":
            return self._vllm_embed(texts)
        if self.config.backend == "openai":
            return self._openai_sdk_embed(texts)
        return self._sentence_transformers_embed(texts)

    def _sentence_transformers_embed(self, texts: list[str]) -> np.ndarray:
        if self._local_model is None:
            self._local_model = SentenceTransformer(self.config.model)
        return self._local_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def _dashscope_embed(self, texts: list[str]) -> np.ndarray:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": {"texts": texts},
        }
        response = requests.post(self.config.base_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        embeddings = [item["embedding"] for item in data["output"]["embeddings"]]
        return np.array(embeddings, dtype="float32")

    def _vllm_embed(self, texts: list[str]) -> np.ndarray:
        api_key = os.getenv(self.config.api_key_env) or "EMPTY"
        client = OpenAI(api_key=api_key, base_url=self.config.vllm_base_url)
        response = client.embeddings.create(model=self.config.model, input=texts)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype="float32")

    def _openai_sdk_embed(self, texts: list[str]) -> np.ndarray:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.config.api_key_env} not set")
        client = OpenAI(api_key=api_key, base_url=self.config.base_url)
        response = client.embeddings.create(model=self.config.model, input=texts)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype="float32")
