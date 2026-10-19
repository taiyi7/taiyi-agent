from taiyi_agent.memory.rag.models import RetrievalHit
from taiyi_agent.memory.rag.ports.storage import (
    RecordStorePort,
    VectorIndexPort,
)
from taiyi_agent.memory.rag.ports.embedder import EmbedderPort
from taiyi_agent.memory.rag.config import RAGConfig


class HybridRetriever:
    """RAG 稠密 + 稀疏混合检索。"""

    def __init__(
        self,
        record_store: RecordStorePort,
        vector_index: VectorIndexPort,
        embedder: EmbedderPort,
        config: RAGConfig,
    ):
        self.record_store = record_store
        self.vector_index = vector_index
        self.embedder = embedder
        self.config = config

    def search(
        self,
        query: str,
        *,
        namespace: str | None = None,
        limit: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievalHit]:
        namespace = namespace or self.config.namespace
        if limit is None:
            limit = self.config.retrieval.default_limit

        # 1. 稠密检索
        query_vector = self.embedder.embed_query(query)

        dense_hits = self.vector_index.search_chunks(
            query_vector,
            namespace=namespace,
            limit=limit,
            filters=filters,
        )

        # 2. 稀疏检索
        sparse_hits = self.record_store.sparse_search(
            query,
            namespace=namespace,
            limit=limit,
            filters=filters,
        )

        # 3. RRF 融合
        return self._fuse(
            dense_ids=[hit.payload.get("chunk_id", hit.record_id) for hit in dense_hits],
            sparse_ids=[chunk_id for chunk_id, _ in sparse_hits],
            namespace=namespace,
        )

    def _fuse(
        self,
        *,
        dense_ids: list[str],
        sparse_ids: list[str],
        namespace: str,
    ) -> list[RetrievalHit]:
        """
        RRF融合，输出综合排名前 k 个结果
        """
        rrf_k = self.config.retrieval.rrf_k
        scores: dict[str, float] = {}

        for rank, chunk_id in enumerate(dense_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0)
            scores[chunk_id] += 1.0 / (rrf_k + rank)

        for rank, chunk_id in enumerate(sparse_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0)
            scores[chunk_id] += 1.0 / (rrf_k + rank)

        ordered_ids = [
            chunk_id
            for chunk_id, _ in sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        # 一次性从 SQLite 读取完整 Chunk
        chunks = self.record_store.get_chunks(
            ordered_ids,
            namespace=namespace,
        )
        chunk_map = {
            chunk.chunk_id: chunk
            for chunk in chunks
        }

        return [
            RetrievalHit(
                chunk=chunk_map[chunk_id],
                score=scores[chunk_id],
                source="hybrid",
                rank=rank,
            )
            for rank, chunk_id in enumerate(ordered_ids, start=1)
            if chunk_id in chunk_map
        ]