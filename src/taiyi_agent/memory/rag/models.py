"""RAG文档处理模块"""

from typing import Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid5


class SourceDocument(BaseModel):
    """原始文档类"""
    doc_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(BaseModel):
    """文档块类"""
    chunk_id: str       # 使用 sha256
    doc_id: str
    content: str
    index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    chunk: TextChunk
    score: float
    source: str              # dense / sparse / hybrid
    rank: int = 0
    rerank_score: float | None = None


# RAG项目专属namespace，用于将 sha256 确定性的转为 UUID
NAMESPACE_RAGChunk = UUID("e3be227b-3ccc-423f-8728-a76adb86a9f4")

def chunk_hash_to_uuid(sha256_hex: str) -> UUID:
    """
    输入chunk的sha256 hex字符串，稳定生成UUIDv5
    sha256_hex: 64位十六进制字符串（小写/大写都可以）
    """
    return uuid5(NAMESPACE_RAGChunk, sha256_hex)

