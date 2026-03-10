from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from common.utils import read_json, write_json


class FaissIndex:
    def __init__(self, index_path: str | Path, meta_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.index: faiss.Index | None = None
        self.meta: dict[str, list[str]] = {}

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        normalized = self._normalize(embeddings.astype("float32"))
        index.add(normalized)
        self.index = index
        self.meta = {"ids": ids}

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Index not built")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        write_json(self.meta_path, self.meta)

    def load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        self.meta = read_json(self.meta_path)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, list[str]]:
        if self.index is None:
            raise RuntimeError("Index not loaded")
        query = self._normalize(query.astype("float32"))
        scores, indices = self.index.search(query, top_k)
        ids = [self.meta["ids"][idx] for idx in indices[0] if idx >= 0]
        return scores[0], ids
