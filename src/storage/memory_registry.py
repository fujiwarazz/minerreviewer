"""MemoryRegistry: 管理多个热插拔记忆库

支持：
1. 按会议/年份独立管理记忆库
2. 动态加载/卸载记忆库
3. 列出所有可用记忆库
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("data/processed/registry.json")
MEMORY_ROOT = Path("data/processed/memory")


class MemoryRegistry:
    """记忆库注册表，管理多个记忆库的热插拔"""

    def __init__(self, registry_path: Path | str | None = None) -> None:
        self.registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry: dict[str, Any] = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """加载注册表"""
        if self.registry_path.exists():
            with open(self.registry_path, encoding="utf-8") as f:
                return json.load(f)
        return {"memories": {}, "active_memories": []}

    def _save_registry(self) -> None:
        """保存注册表"""
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def list_memories(self) -> list[dict[str, Any]]:
        """列出所有记忆库"""
        result = []
        for memory_id, info in self.registry.get("memories", {}).items():
            result.append({
                "memory_id": memory_id,
                "venue": info.get("venue"),
                "year": info.get("year"),
                "cases_count": info.get("cases_count", 0),
                "cards_count": info.get("cards_count", 0),
                "active": memory_id in self.registry.get("active_memories", []),
                "path": info.get("path"),
            })
        return result

    def get_active_memories(self) -> list[str]:
        """获取所有活跃的记忆库 ID"""
        return self.registry.get("active_memories", [])

    def get_memory_info(self, memory_id: str) -> dict[str, Any] | None:
        """获取指定记忆库的信息"""
        return self.registry.get("memories", {}).get(memory_id)

    def get_memory_path(self, memory_id: str) -> Path | None:
        """获取指定记忆库的路径"""
        info = self.get_memory_info(memory_id)
        if info:
            return MEMORY_ROOT / memory_id
        return None

    def activate_memory(self, memory_id: str) -> bool:
        """激活记忆库"""
        if memory_id not in self.registry.get("memories", {}):
            logger.warning("Memory %s not found", memory_id)
            return False

        active_memories = self.registry.get("active_memories", [])
        if memory_id not in active_memories:
            active_memories.append(memory_id)
            self.registry["active_memories"] = active_memories
            self._save_registry()
            logger.info("Activated memory %s", memory_id)
        return True

    def deactivate_memory(self, memory_id: str) -> bool:
        """停用记忆库"""
        active_memories = self.registry.get("active_memories", [])
        if memory_id in active_memories:
            active_memories.remove(memory_id)
            self.registry["active_memories"] = active_memories
            self._save_registry()
            logger.info("Deactivated memory %s", memory_id)
            return True
        return False

    def register_memory(
        self,
        memory_id: str,
        venue: str,
        year: int,
        cases_count: int,
        cards_count: int,
    ) -> None:
        """注册新记忆库"""
        self.registry["memories"][memory_id] = {
            "path": f"memory/{memory_id}",
            "venue": venue,
            "year": year,
            "cases_count": cases_count,
            "cards_count": cards_count,
            "created_at": datetime.now().isoformat(),
            "active": True,
        }
        if memory_id not in self.registry.get("active_memories", []):
            self.registry.setdefault("active_memories", []).append(memory_id)
        self._save_registry()
        logger.info("Registered memory %s (%s %d)", memory_id, venue, year)

    def unregister_memory(self, memory_id: str) -> bool:
        """注销记忆库"""
        if memory_id in self.registry.get("memories", {}):
            del self.registry["memories"][memory_id]
            self.deactivate_memory(memory_id)
            self._save_registry()
            logger.info("Unregistered memory %s", memory_id)
            return True
        return False

    def get_memories_for_venue(self, venue: str, before_year: int | None = None) -> list[str]:
        """获取指定会议的所有活跃记忆库"""
        result = []
        for memory_id in self.get_active_memories():
            info = self.get_memory_info(memory_id)
            if info and info.get("venue") == venue:
                if before_year is None or info.get("year", 0) < before_year:
                    result.append(memory_id)
        return result

    def get_memories_for_year(self, before_year: int) -> list[str]:
        """获取指定年份之前的所有活跃记忆库"""
        result = []
        for memory_id in self.get_active_memories():
            info = self.get_memory_info(memory_id)
            if info and info.get("year", 0) < before_year:
                result.append(memory_id)
        return result


# 导入 datetime（放在文件末尾避免循环导入）
from datetime import datetime