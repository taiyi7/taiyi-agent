"""RAG 模块的配置模型与环境变量加载。

集中管理文档Load、Chunk、Retrieval、Embedding、Rerank 及 SQLite/Qdrant
存储的各项配置；RAGConfig.from_env() 从环境变量覆盖默认值。
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


class LoaderConfig(BaseModel):
    encoding: str = "utf-8"


class ChunkingConfig(BaseModel):
    # 当前 DocumentChunker 按字符长度切分，不是 token 数
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 50
    separators: list[str] = Field(
        default_factory=lambda: [
            "\n\n",
            "\n",
            "。",
            ".",
            " ",
        ]
    )
    min_chunk_length: int = 50
    merge_max_length: int = 512


class RetrievalConfig(BaseModel):
    candidate_k: int = 20
    answer_k: int = 5
    max_candidate_k: int = 100
    default_limit: int = 20
    rrf_k: int = 60


class EmbeddingConfig(BaseModel):
    model: str | None = None
    api_key: str | None = None
    url: str | None = None
    timeout: float = 60.0
    batch_size: int = 16


class RerankerConfig(BaseModel):
    url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout: float = 30.0
    top_k: int = 5


class SQLiteConfig(BaseModel):
    path: Path = Path(r"C:\code\agent\taiyi-agent\test\rag_test\data\db\rag_test.db")
    timeout: float = 30.0
    busy_timeout_ms: int = 30000
    journal_mode: Literal["WAL", "DELETE"] = "WAL"
    synchronous: Literal["OFF", "NORMAL", "FULL", "EXTRA"] = "NORMAL"
    foreign_keys: bool = True


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection_name: str = "taiyi_vectors"
    vector_size: int = 1024
    distance: Literal["Cosine", "Dot", "Euclid", "Manhattan"] = "Cosine"


class RAGConfig(BaseModel):
    namespace: str = "knowledge_base"

    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            namespace=os.getenv("RAG_NAMESPACE","knowledge_base",),
            loader=LoaderConfig(
                encoding=os.getenv("RAG_LOADER_ENCODING", "utf-8"),
            ),
            chunking=ChunkingConfig(
                chunk_size_tokens=_env_int("RAG_CHUNK_SIZE_TOKENS", 256),
                chunk_overlap_tokens=_env_int("RAG_CHUNK_OVERLAP_TOKENS", 50),
                min_chunk_length=_env_int("RAG_MIN_CHUNK_LENGTH", 50),
                merge_max_length=_env_int("RAG_MERGE_MAX_TOKENS", 512),
            ),
            retrieval=RetrievalConfig(
                candidate_k=_env_int("RAG_CANDIDATE_K", 20),
                answer_k=_env_int("RAG_ANSWER_K", 5),
                max_candidate_k=_env_int("RAG_MAX_CANDIDATE_K", 100),
                default_limit=_env_int("RAG_DEFAULT_LIMIT", 20),
                rrf_k=_env_int("RAG_RRF_K", 60),
            ),
            embedding=EmbeddingConfig(
                model=os.getenv("EMBEDDER_MODEL"),
                api_key=os.getenv("EMBEDDER_API_KEY"),
                url=os.getenv("EMBEDDER_URL"),
                timeout=_env_float("EMBEDDER_TIMEOUT", 60.0),
                batch_size=_env_int("EMBEDDER_BATCH_SIZE", 16),
            ),
            reranker=RerankerConfig(
                url=os.getenv("QWEN_RERANK_URL"),
                api_key=os.getenv("QWEN_RERANK_API_KEY"),
                model=os.getenv("QWEN_RERANK_MODEL"),
                timeout=_env_float("QWEN_RERANK_TIMEOUT", 30.0),
                top_k=os.getenv("QWEN_RERANK_TOP_K", 5),
            ),
            sqlite=SQLiteConfig(
                path=Path(
                    os.getenv("RAG_SQLITE_PATH", r"C:\code\agent\taiyi-agent\test\rag_test\data\db\rag_test.db")
                ),
            ),
            qdrant=QdrantConfig(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name=os.getenv("QDRANT_COLLECTION_NAME", "taiyi_vectors"),
                vector_size=_env_int("QDRANT_VECTOR_SIZE", 1024),
                distance=os.getenv("QDRANT_DISTANCE", "Cosine"),
            ),
        )