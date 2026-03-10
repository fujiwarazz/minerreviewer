from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    backend: str
    model: str
    temperature: float = 0.2
    base_url: str = "https://api.openai.com/v1"
    dashscope_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    api_key_env: str = "OPENAI_API_KEY"


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def generate(self, prompt: str) -> str:
        if self.config.backend == "dummy":
            return "{" "\"response\": \"dummy\"" "}"
        if self.config.backend == "openai":
            return self._openai_sdk_generate(prompt)
        if self.config.backend == "dashscope":
            return self._dashscope_generate(prompt)
        return self._openai_generate(prompt)

    def generate_json(self, prompt: str) -> dict[str, Any]:
        if self.config.backend == "dummy":
            return {"response": "dummy"}
        if self.config.backend == "dashscope":
            raw = self._dashscope_generate(prompt)
        elif self.config.backend == "openai":
            raw = self._openai_sdk_generate(prompt, json_mode=True)
        else:
            raw = self._openai_generate(prompt, json_mode=True)
        return self._parse_json(raw)

    def _openai_generate(self, prompt: str, json_mode: bool = False) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": "You are a precise research assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = requests.post(f"{self.config.base_url}/chat/completions", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _openai_sdk_generate(self, prompt: str, json_mode: bool = False) -> str:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.config.api_key_env} not set")
        client = OpenAI(api_key=api_key, base_url=self.config.base_url)
        response_format = {"type": "json_object"} if json_mode else None
        response = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            response_format=response_format,
            messages=[
                {"role": "system", "content": "You are a precise research assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def _dashscope_generate(self, prompt: str) -> str:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": {"prompt": prompt},
            "parameters": {"temperature": self.config.temperature},
        }
        response = requests.post(self.config.dashscope_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["output"]["text"]

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                candidate = match.group(0)
                candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                return json.loads(candidate)
            raise
