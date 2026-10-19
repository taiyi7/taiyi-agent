from collections.abc import Iterable
from typing import Any

from taiyi_agent.memory.rag.models import TextChunk
from taiyi_agent.memory.rag.ports.storage import RAG_CHUNK_TYPE
from taiyi_agent.memory.storage.models import VectorRecord
from taiyi_agent.memory.storage.qdrant_store import QdrantStore
from taiyi_agent.memory.rag.models import chunk_hash_to_uuid

class QdrantRAGIndex:
    """使用 QdrantStore 实现 VectorIndexPort。"""

    def __init__(self, store: QdrantStore):
        self.store = store

    def upsert_chunks(
        self,
        chunks: Iterable[TextChunk],
        vectors: Iterable[list[float]],
        *,
        namespace: str,
    ) -> None:
        records = [
            VectorRecord(
                record_id=str(chunk_hash_to_uuid(chunk.chunk_id)),
                vector=vector,
                data_type=RAG_CHUNK_TYPE,
                namespace=namespace,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source_uri": chunk.metadata.get("source_uri"),
                    "source_type": chunk.metadata.get("source_type"),
                    "title": chunk.metadata.get("title"),
                    "page_number": chunk.metadata.get("page_number"),
                    "section_path": chunk.metadata.get("section_path"),
                    "chunk_index": chunk.index,
                    "tenant_id": chunk.metadata.get("tenant_id"),
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        self.store.upsert_many(records)

    def search_chunks(
        self,
        vector: list[float],
        *,
        namespace: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ):
        return self.store.search(
            vector,
            data_type=RAG_CHUNK_TYPE,
            namespace=namespace,
            limit=limit,
            extra_filter=filters,
        )

    def delete_document(
        self,
        doc_id: str,
        *,
        namespace: str,
    ) -> None:
        self.store.delete_filter(
            data_type=RAG_CHUNK_TYPE,
            namespace=namespace,
            extra_filter={"doc_id", doc_id}
        )

    def health_check(self)-> bool:
        return self.store.healthcheck()