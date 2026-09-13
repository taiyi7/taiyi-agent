'''上下文数据结构'''
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Any

ConextItemType = Literal[
    "system",
    "history",
    "memory",
    "rag",
    "tool",
    "note",
]

ContextSectionType = Literal[
    "role_policies",
    "task",
    "state",
    "evidence",
    "history",
    "context",
    "output",
]

class ContextItem(BaseModel):
    '''上下文条目数据结构

    参数：
        model_config： 模型实例化后数值修改需要校验
        content： 上下文具体内容
        item_tpye： 上下文类型，见ConextItemType
        source： 内容来源类型或来源组件，例如 "memory_store"、"rag"、"history"、"note_tool"
        source_id： 来源内容的唯一标识， 例如： memory_1234
        priority： 人工指定的优先级，用于预算不足时决定保留顺序, 数值越大优先级越高
        relevance_score： 内容与当前用户问题的相关性分数，范围 0.0 到 1.0。0 表示暂未计算或不相关，1 表示高度相关
        token_count： 内容占用的 token 数量，用于 token 预算控制
        created_at： 该上下文条目的创建时间，用于新近性排序、时间衰减和审计
        metadata： 不参与核心上下文协议的扩展信息，例如用户 ID、标签、文档页码、检索距离、工具参数、置信度等
    
    '''
    model_config = ConfigDict(validate_assignment=True)

    content: str = Field(min_length=1)
    item_tpye: ConextItemType

    source: str | None = None
    source_id: str | None = None

    priority: int = Field(default=0, ge=0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)

    token_count: int = Field(default=0, ge=0)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextSection(BaseModel):
    """压缩前的结构化上下文区块。

    items 始终保存完整语义条目。对于 history 区块，一个 item 对应一条
    Message，即使消息正文包含换行，也不会在压缩时被拆成多条消息。
    """

    model_config = ConfigDict(validate_assignment=True)

    section_type: ContextSectionType
    title: str = Field(min_length=1)
    items: list[str] = Field(default_factory=list)
    priority: int = Field(default=0, ge=0)
    required: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextConfig(BaseModel):
    '''上下文约束配置参数

    参数:
        model_config： 模型实例化后数值修改需要校验
        max_tokens： Agent最大上下文窗口大小
        reverse_tokens： Agent预留留上下文窗口大小
        max_history_messages： 上下文最多保留的历史Message个数
        memory_limit： 一次从长期记忆中最多召回的记忆个数
        history_keep_recent： 压缩历史时优先原样保留的最近消息数
        history_summary_max_tokens： 旧消息摘要最多使用的 token 数
        min_relevance： 候选记忆或检索结果进入上下文前的最低相关性分数。低于该值直接淘汰。
        relevance_weight： 上下文综合排序中“与当前问题相关性”的权重
        recency_weight： 上下文综合排序中“信息新旧程度”的权重
        enable_compression： 超出 token 预算时是否尝试压缩上下文。
        enable_memory_retrieval： 构建上下文时是否主动查询长期记忆。
    '''
    model_config = ConfigDict(validate_assignment=True)

    # 上下文tokens限制
    max_tokens: int = Field(default=16384, gt=0)
    reverse_tokens: int = Field(default=2048, ge=0)

    # 历史和记忆限制
    max_history_messages: int = Field(default=50, ge=0)
    memory_limit: int = Field(default=5, ge=0)
    history_keep_recent: int = Field(default=6, ge=0)
    history_summary_max_tokens: int = Field(default=512, gt=0)

    # 上下文综合排序配置
    min_relevance: float = Field(default=0.2, ge=0.0, le=1.0)
    relevance_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # 开关配置
    enable_compression: bool = True
    enable_memory_retrieval: bool = True



# ContextItem(
#     content="项目目前正在迁移 PostgreSQL。",
#     item_type="memory",
#     source="memory_store",
#     source_id="memory-8f31",
#     priority=2,
#     relevance_score=0.86,
#     token_count=12,
#     metadata={
#         "user_id": "user-001",
#         "tags": ["project_state", "database"],
#         "importance": 0.9,
#     },
# )
