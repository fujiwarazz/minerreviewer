from __future__ import annotations

from dataclasses import dataclass

from clients.llm_client import LLMClient


@dataclass
class AgentConfig:
    name: str
    llm: LLMClient
