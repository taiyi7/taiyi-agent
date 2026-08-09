'''记忆的数据结构'''
from typing import Optional, Dict, Any, 
import uuid
from pydantic import BaseModel, Field
import datetime

class MemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="记忆唯一ID")
    content:str
    tags: list[str] = Field(default_factory=list, description="标签，用于筛选")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)     # 每个实例创建时都会获得独立的元数据
