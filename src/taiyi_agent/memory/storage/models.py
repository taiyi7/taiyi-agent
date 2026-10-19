from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StoredRecord(BaseModel):
    """
    StoredRecord 适合保存：
        - RAG Document
        - RAG Chunk
        - 用户长期记忆
        - Agent 消息
        - 工具调用结果
        - 其他业务数据
    """
    record_id: str    # str(uuid)
    data_type: str
    namespace: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class VectorRecord(BaseModel):
    """
    VectorRecord 适合保存：
        - RAG Chunk 向量
        - Memory 向量
        - 对话消息向量
        - 其他需要语义检索的数据
    """
    record_id: str     # str(uuid)
    vector: list[float]
    data_type: str
    namespace: str
    payload: dict[str, Any] = Field(default_factory=dict)


class VectorHit(BaseModel):
    record_id: str
    score: float
    data_type: str
    namespace: str
    payload: dict[str, Any] = Field(default_factory=dict)