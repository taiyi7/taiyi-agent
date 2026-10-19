from typing import Iterable, Protocol
from typing import Any

from taiyi_agent.memory.rag.models import (
    SourceDocument,
    TextChunk,
)
from taiyi_agent.memory.storage.models import VectorHit


RAG_DOCUMENT_TYPE = "rag_document"
RAG_CHUNK_TYPE = "rag_chunk"


class RecordStorePort(Protocol):
    """RAG 完整文档和 Chunk 的存储接口。"""

    def save_document(
        self,
        document: SourceDocument,
        *,
        namespace: str,
    ) -> None:
        ...

    def save_chunks(
        self,
        chunks: Iterable[TextChunk],
        *,
        namespace: str,
    ) -> None:
        ...

    def get_chunks(
        self,
        chunk_ids: Iterable[str],
        *,
        namespace: str,
    ) -> list[TextChunk]:
        ...

    def sparse_search(
        self,
        query: str,
        *,
        namespace: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        ...

    def healthcheck(self) -> bool:
        ...


class VectorIndexPort(Protocol):
    """RAG 向量写入和稠密检索接口。"""

    def upsert_chunks(
        self,
        chunks: Iterable[TextChunk],
        vectors: Iterable[list[float]],
        *,
        namespace: str,
    ) -> None:
        ...

    def search_chunks(
        self,
        vector: list[float],
        *,
        namespace: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        ...

    def delete_document(
        self,
        doc_id: str,
        *,
        namespace: str,
    ) -> None:
        ...

    def healthcheck(self) -> bool:
        ...