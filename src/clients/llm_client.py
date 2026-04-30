from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import subprocess
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
    api_key: str | None = None  # Direct API key from config
    max_retries: int = 5  # 最大重试次数
    retry_base_delay: float = 2.0  # 重试基础延迟（秒）


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """检查是否是429限流错误"""
        error_str = str(error).lower()
        return "429" in error_str or "rate" in error_str or "负载" in error_str or "saturated" in error_str

    def _retry_call(self, func, *args, **kwargs):
        """带retry的调用包装"""
        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay * (2 ** attempt)  # 递增延迟
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}), waiting {delay}s before retry...")
                    time.sleep(delay)
                else:
                    raise
        raise RuntimeError(f"Max retries ({self.config.max_retries}) exceeded")

    def generate(self, prompt: str) -> str:
        if self.config.backend == "dummy":
            return "{" "\"response\": \"dummy\"" "}"
        return self._retry_call(self._generate_impl, prompt)

    def _generate_impl(self, prompt: str) -> str:
        """实际生成实现"""
        if self.config.backend == "openai":
            return self._openai_sdk_generate(prompt)
        if self.config.backend == "curl":
            return self._curl_generate(prompt)
        if self.config.backend == "dashscope":
            return self._dashscope_generate(prompt)
        return self._openai_generate(prompt)

    def generate_json(self, prompt: str) -> dict[str, Any]:
        if self.config.backend == "dummy":
            return {"response": "dummy"}
        raw = self._retry_call(self._generate_json_impl, prompt)
        return self._parse_json(raw)

    def _generate_json_impl(self, prompt: str) -> str:
        """JSON模式生成实现"""
        if self.config.backend == "dashscope":
            return self._dashscope_generate(prompt)
        elif self.config.backend == "openai":
            return self._openai_sdk_generate(prompt, json_mode=True)
        elif self.config.backend == "curl":
            return self._curl_generate(prompt, json_mode=True)
        else:
            return self._openai_generate(prompt, json_mode=True)

    def _openai_generate(self, prompt: str, json_mode: bool = False) -> str:
        api_key = self.config.api_key or os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"API key not set (config.api_key or {self.config.api_key_env})")
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

    def _curl_generate(self, prompt: str, json_mode: bool = False) -> str:
        """使用 curl subprocess 调用，绕过 Python SSL 问题"""
        api_key = self.config.api_key or os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"API key not set (config.api_key or {self.config.api_key_env})")

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

        url = f"{self.config.base_url}/chat/completions"

        # 超时设置
        connect_timeout = 60  # 连接超时
        max_time = 300  # 总超时（5分钟）

        result = subprocess.run([
            'curl', '-s', '-X', 'POST', url,
            '-H', f'Authorization: Bearer {api_key}',
            '-H', 'Content-Type: application/json',
            '-d', json.dumps(payload),
            '--connect-timeout', str(connect_timeout),
            '--max-time', str(max_time),
            '-w', '\n%{http_code}\n%{time_total}\n%{time_connect}'  # 添加响应码和时间信息
        ], capture_output=True, text=True, timeout=max_time + 60)

        # 解析 curl 输出（最后一行是 http_code, time_total, time_connect）
        parts = result.stdout.strip().split('\n')
        if len(parts) >= 3:
            json_response = '\n'.join(parts[:-3])
            http_code = parts[-3]
            time_total = parts[-2]
            time_connect = parts[-1]
        else:
            json_response = result.stdout
            http_code = "unknown"
            time_total = "unknown"
            time_connect = "unknown"

        if result.returncode != 0:
            # 记录详细超时原因
            error_detail = {
                "returncode": result.returncode,
                "stderr": result.stderr,
                "http_code": http_code,
                "time_total": time_total,
                "time_connect": time_connect,
                "stdout_preview": json_response[:500]
            }
            raise RuntimeError(f"curl failed: {json.dumps(error_detail)}")

        if not json_response:
            raise RuntimeError(f"curl returned empty response (http_code={http_code}, time_total={time_total})")

        try:
            data = json.loads(json_response)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"curl returned invalid JSON (http_code={http_code}, time_total={time_total}): {json_response[:500]}")

        if "error" in data:
            raise RuntimeError(f"API error (http_code={http_code}): {data['error']}")

        return data["choices"][0]["message"]["content"]

    def _openai_sdk_generate(self, prompt: str, json_mode: bool = False) -> str:
        api_key = self.config.api_key or os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"API key not set (config.api_key or {self.config.api_key_env})")
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
