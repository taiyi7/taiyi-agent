'''记忆的数据结构'''
from typing import Optional, Any
import uuid
from pydantic import BaseModel, Field
from datetime import datetime, timezone

class MemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="记忆唯一ID")
    user_id:str
    content:str
    tags: list[str] = Field(default_factory=list, description="标签，用于筛选")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)     # 每个实例创建时都会获得独立的元数据
        # 可用于排序、过滤和衰减
    importance: float = Field(default=0.5, ge=0, le=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

