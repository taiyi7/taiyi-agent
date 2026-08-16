"""工具调用数据结构"""

from typing import Any
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """大模型原生工具接口返回的工具调用数据"""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """包含工具调用信息的大模型返回结果（非流式输出）"""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
