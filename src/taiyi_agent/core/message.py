'''消息系统'''
from typing import Literal, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone

MessageRole = Literal[
    "user",
    "assistant",
    "system",
    "tool",
    "developer",
]

# Message 继承 pydantic.BaseModel，主要是为了让消息对象具备自动校验、类型转换和序列化能力。
class Message(BaseModel):
    '''
    消息类
    TaiyiAgent中用于消息数据结构类

    参数：
        role: 消息来源的角色
        content: 消息具体内容
        timestamp: 消息时间戳
        metadata: 它表示消息的附加元数据，可以存放不属于核心消息协议，但对 Agent 系统有用的信息。常见用途包括：
            记录消息来自哪个工具
            保存搜索来源和网页 URL
            保存请求 ID，方便日志追踪
            记录 token、耗时、模型名
            区分普通回答、工具结果、重试结果
            保存 RAG 文档 ID、相似度分数
            保存消息创建时间、调用链信息
            做审计、计费和调试
    '''

    role: MessageRole                   # 无默认值，实例初始化必传值 （属于模型字段，不是类变量）
    content: str | None = None

    # 原生工具调用消息
    tool_calls: list[dict[str,Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)     # 每个实例创建时都会获得独立时间
    )     
    metadata: dict[str, Any] = Field(default_factory=dict)     # 每个实例创建时都会获得独立的元数据

    def to_openai_dict(self) -> dict[str, Any]:
        '''将消息内容转化为字典格式，符合OpenAI的API'''
        message: dict[str, Any] = {
            "role": self.role,
        }

        if self.content is not None:
            message["content"] = self.content

        if self.tool_calls is not None:
            message["tool_calls"] = self.tool_calls

        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id

        if self.name is not None:
            message["name"] = self.name

        return message

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "Message":
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )
