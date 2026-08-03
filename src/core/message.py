'''消息系统'''
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

MessageRole = Literal["user", "assistant", "system", "tool", "developer"]

# Message 继承 pydantic.BaseModel，主要是为了让消息对象具备自动校验、类型转换和序列化能力。
class Message(BaseModel):
    '''
    消息类
    TaiyiAgent中用于消息系统的基础类

    参数：
        content: 消息具体内容
        role: 消息来源的角色
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

    content: str                        # 无默认值，实例初始化必传值 （属于模型字段，不是类变量）
    role: MessageRole                   # 无默认值，实例初始化必传值 （属于模型字段，不是类变量）
    timestamp: datetime = Field(default_factory=datetime.now)     # 每个实例创建时都会获得独立时间
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)     # 每个实例创建时都会获得独立的元数据

    # BaseModel的子类执行 super().__init__ 支持传入当前 Model 全部字段名 + Pydantic 内置配置参数。
    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get('timestamp', datetime.now()),
            metadata=kwargs.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        '''将消息内容转化为字典格式，符合OpenAI的API'''
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"

