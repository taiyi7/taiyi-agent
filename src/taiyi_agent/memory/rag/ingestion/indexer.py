from taiyi_agent.memory.rag.models import SourceDocument, TextChunk
from taiyi_agent.memory.rag.ports.storage import (
    RecordStorePort,
    VectorIndexPort,
)
from taiyi_agent.memory.rag.ports.embedder import EmbedderPort
from taiyi_agent.memory.rag.config import RAGConfig

class Indexer:
    """RAG 离线索引服务。"""

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

    def index(
        self,
        document: SourceDocument,
        chunks: list[TextChunk],
        *,
        namespace: str | None = None,
    ) -> None:
        if not chunks:
            return

        if any(chunk.doc_id != document.doc_id for chunk in chunks):
            raise ValueError("chunks 必须属于同一个 document")

        vectors = self.embedder.embed_documents(
            [chunk.content for chunk in chunks]
        )

        if len(vectors) != len(chunks):
            raise ValueError("embedding 数量必须与 chunk 数量一致")

        namespace = namespace or self.config.namespace
        # embedding 成功后再写存储
        self.record_store.save_document(document, namespace=namespace)
        self.record_store.save_chunks(chunks, namespace=namespace)
        self.vector_index.upsert_chunks(chunks, vectors, namespace=namespace)
        print("[indexer]成功")