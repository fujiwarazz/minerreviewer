from __future__ import annotations

import logging
from dataclasses import dataclass

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

logger = logging.getLogger(__name__)


@dataclass
class MilvusConfig:
    host: str
    port: int
    papers_collection: str
    reviews_collection: str


class MilvusStore:
    def __init__(self, config: MilvusConfig, batch_size: int = 1000) -> None:
        self.config = config
        self.batch_size = batch_size
        connections.connect(host=config.host, port=str(config.port))

    def upsert_embeddings(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str] | None = None,
    ) -> None:
        if not embeddings:
            return
        dim = len(embeddings[0])
        collection = self._get_or_create_collection(collection_name, dim, store_text=bool(texts))
        total = len(ids)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_ids = ids[start:end]
            batch_emb = embeddings[start:end]
            if texts:
                batch_texts = texts[start:end]
                data = [batch_ids, batch_texts, batch_emb]
            else:
                data = [batch_ids, batch_emb]
            collection.insert(data)
        collection.flush()
        logger.info("Upserted %s vectors into %s (batch_size=%s)", total, collection_name, self.batch_size)

    def search_ids(self, collection_name: str, query_embedding: list[float], top_k: int) -> list[str]:
        if not utility.has_collection(collection_name):
            return []
        collection = Collection(collection_name)
        collection.load()
        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["doc_id"],
        )
        hits = results[0] if results else []
        return [hit.entity.get("doc_id") for hit in hits if hit.entity]

    def search_texts(self, collection_name: str, query_embedding: list[float], top_k: int) -> list[str]:
        if not utility.has_collection(collection_name):
            return []
        collection = Collection(collection_name)
        collection.load()
        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["doc_id", "text"],
        )
        hits = results[0] if results else []
        return [hit.entity.get("text") for hit in hits if hit.entity and hit.entity.get("text")]

    def _get_or_create_collection(self, collection_name: str, dim: int, store_text: bool = False) -> Collection:
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            return collection
        fields = [
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
        ]
        if store_text:
            fields.append(FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096))
        fields.append(FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim))
        schema = CollectionSchema(fields=fields, description="Vector store")
        collection = Collection(name=collection_name, schema=schema)
        index_params = {"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
        collection.create_index(field_name="embedding", index_params=index_params)
        return collection
