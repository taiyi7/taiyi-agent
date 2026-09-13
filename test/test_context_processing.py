"""上下文构建和工具调用过程测试。

测试使用本地 fake LLM，避免依赖外部模型服务，同时保留真实的
ContextBuilder、StandardAgent 和 ToolRegistry 调用链。
"""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from taiyi_agent.agents.standard_agent import StandardAgent
from taiyi_agent.context.context_builder import ContextBuilder
from taiyi_agent.context.context_data import ContextConfig, ContextItem, ContextSection
from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.core.message import Message
from taiyi_agent.core.tool_call import LLMResponse, ToolCall
from taiyi_agent.tool.tool_base import ToolParameter
from taiyi_agent.tool.tool_registry import ToolRegistry


class RecordingTaiyiAgentLLM(TaiyiAgentLLM):
    """TaiyiAgentLLM 的本地测试实现，不创建真实 OpenAI 客户端。"""

    llm_model_id = "test-model"

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        # 测试只验证 TaiyiAgentLLM.invoke 接口，不需要 API key 或网络客户端。
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append({
            "messages": deepcopy(messages),
            "temperature": temperature,
            "tools": deepcopy(tools),
        })
        if not self.responses:
            raise AssertionError("fake LLM 没有预设返回值")
        return self.responses.pop(0)


def run_build(builder: ContextBuilder, **kwargs: Any) -> str:
    return asyncio.run(builder.build(**kwargs))


def test_context_builder_initialization_and_first_context() -> None:
    config = ContextConfig(max_tokens=256, reverse_tokens=0)
    llm = RecordingTaiyiAgentLLM()
    builder = ContextBuilder(config=config, llm=llm)

    rendered = run_build(
        builder,
        user_query="如何发布服务？",
        system_instructions="你是发布助手。",
    )

    assert builder.config is config
    assert "[Role & Policies]\n你是发布助手。" in rendered
    assert "[Task]\n如何发布服务？" in rendered
    assert "[Output]" in rendered
    assert llm.calls == []


def test_tool_result_is_added_to_context() -> None:
    builder = ContextBuilder(
        config=ContextConfig(max_tokens=256, reverse_tokens=0),
        llm=RecordingTaiyiAgentLLM(),
    )
    tool_result = "广州当前 28 摄氏度，晴。"
    item = ContextItem(
        content=tool_result,
        item_tpye="tool",
        relevance_score=1.0,
        token_count=builder._count_token(tool_result),
        timestamp=datetime.now(timezone.utc),
    )

    rendered = run_build(
        builder,
        user_query="广州天气如何？",
        additional_items=[item],
    )

    assert "[Context]" in rendered
    assert tool_result in rendered

# 测试较长上下文的压缩
def test_long_context_is_compressed_without_mutating_sections() -> None:
    builder = ContextBuilder(
        config=ContextConfig(max_tokens=256, reverse_tokens=0),
        llm=RecordingTaiyiAgentLLM(),
    )
    sections = [
        ContextSection(
            section_type="task",
            title="[Task]",
            items=["部署服务时请严格检查配置和日志。"],
            priority=90,
            required=True,
        ),
        ContextSection(
            section_type="context",
            title="[Context]",
            items=["详细背景 " + "说明 " * 200],
            priority=40,
        ),
    ]
    original_items = list(sections[1].items)

    rendered = builder._compress(sections, max_token=30)

    assert builder._count_token(rendered) <= 30
    assert "[Task]" in rendered
    assert sections[1].items == original_items


# 测试： 压缩时优先删除低优先级的section
def test_compress_removes_low_priority_optional_section_first() -> None:
    builder = ContextBuilder(llm=RecordingTaiyiAgentLLM())
    required = ContextSection(
        section_type="task",
        title="[Task]",
        items=["回答部署问题"],
        priority=90,
        required=True,
    )
    evidence = ContextSection(
        section_type="evidence",
        title="[Evidence]",
        items=["重要证据：部署必须先检查配置。"],
        priority=60,
    )
    optional_output = ContextSection(
        section_type="output",
        title="[Output]",
        items=["可选输出 " * 100],
        priority=20,
    )
    budget = builder._count_token(
        builder._format_all_sections([required, evidence])
    )

    rendered = builder._compress(
        [required, evidence, optional_output],
        max_token=budget,
    )

    assert builder._count_token(rendered) <= budget
    assert "重要证据：部署必须先检查配置。" in rendered
    assert "[Output]" not in rendered

# 测试：使用llm进行历史对话压缩
def test_history_compression_uses_llm_summary_and_keeps_recent_message() -> None:
    llm = RecordingTaiyiAgentLLM([
        LLMResponse(content="旧消息摘要：已确认发布流程。"),
    ])
    builder = ContextBuilder(
        config=ContextConfig(
            max_tokens=256,
            reverse_tokens=0,
            history_keep_recent=1,
            history_summary_max_tokens=8,
        ),
        llm=llm,
    )
    history = ContextSection(
        section_type="history",
        title="[Recent Conversation]",
        items=[
            "旧消息一：讨论服务配置和发布前检查。" * 5,
            "旧消息二：确认需要保留回滚方案。" * 5,
            "最新消息：开始执行发布。",
        ],
        priority=50,
    )

    rendered = builder._compress([history], max_token=30)

    assert "[Earlier Conversation Summary]" in rendered
    assert "旧消息摘要" in rendered
    assert "最新消息：开始执行发布。" in rendered
    assert len(llm.calls) == 1
    assert llm.calls[0]["temperature"] == 0.0

# 测试：llm压缩失败后的处理
def test_history_compression_falls_back_to_recent_messages_when_llm_fails() -> None:
    llm = RecordingTaiyiAgentLLM()
    builder = ContextBuilder(
        config=ContextConfig(
            max_tokens=256,
            reverse_tokens=0,
            history_keep_recent=1,
        ),
        llm=llm,
    )
    history = ContextSection(
        section_type="history",
        title="[Recent Conversation]",
        items=[
            "较早消息：讨论需求。" * 8,
            "中间消息：确认实现方案。" * 8,
            "最新消息：请开始执行。",
        ],
        priority=50,
    )

    rendered = builder._compress([history], max_token=24)

    assert builder._count_token(rendered) <= 24
    assert "最新消息：请开始执行。" in rendered
    assert "[Earlier Conversation Summary]" in rendered
    assert len(llm.calls) == 1

# 测试：压缩保留头和尾内容
def test_task_compression_preserves_head_and_tail() -> None:
    builder = ContextBuilder(llm=RecordingTaiyiAgentLLM())
    task = ContextSection(
        section_type="task",
        title="[Task]",
        items=["START " + "middle " * 200 + " END"],
        priority=90,
        required=True,
    )

    builder._compress_section(task, max_token=30)
    compressed = task.items[0]

    assert builder._count_token(compressed) <= 30
    assert "START" in compressed
    assert "END" in compressed
    assert "...[middle omitted]..." in compressed

# 测试：极限场景下，保留requied sections的标题
def test_required_sections_are_fitted_when_optional_sections_are_gone() -> None:
    builder = ContextBuilder(llm=RecordingTaiyiAgentLLM())
    sections = [
        ContextSection(
            section_type="role_policies",
            title="[Role & Policies]",
            items=["系统规则 " * 100],
            priority=100,
            required=True,
        ),
        ContextSection(
            section_type="task",
            title="[Task]",
            items=["用户任务 " * 100],
            priority=90,
            required=True,
        ),
    ]
    headers = [section.model_copy(update={"items": []}) for section in sections]
    header_budget = builder._count_token(builder._format_all_sections(headers)) + 4

    rendered = builder._compress(sections, max_token=header_budget)

    assert builder._count_token(rendered) <= header_budget
    assert "[Role & Policies]" in rendered
    assert "[Task]" in rendered

# 测试：多次build的时候，上下文内容历史更新
def test_context_contains_history_on_each_build() -> None:
    builder = ContextBuilder(
        config=ContextConfig(max_tokens=256, reverse_tokens=0),
        llm=RecordingTaiyiAgentLLM(),
    )
    first_history = [Message(role="user", content="我负责发布服务。")]
    second_history = first_history + [
        Message(role="assistant", content="收到，我会记住这个背景。")
    ]

    first = run_build(
        builder,
        user_query="第一步是什么？",
        conversation_history=first_history,
    )
    second = run_build(
        builder,
        user_query="那现在怎么做？",
        conversation_history=second_history,
    )

    assert "[user] 我负责发布服务。" in first
    assert "[assistant] 收到，我会记住这个背景。" not in first
    assert "[user] 我负责发布服务。" in second
    assert "[assistant] 收到，我会记住这个背景。" in second


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(
        name="get_weather",
        description="查询指定城市天气",
        func=lambda city: f"{city}是晴天",
        parameters=[
            ToolParameter(
                name="city",
                type="string",
                description="城市名称",
            )
        ],
    )
    return registry


def test_each_llm_call_receives_tool_call_context() -> None:
    llm = RecordingTaiyiAgentLLM([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="get_weather",
                    arguments={"city": "广州"},
                )
            ],
        ),
        LLMResponse(content="广州是晴天。"),
    ])
    agent = StandardAgent(
        name="context-test",
        llm=llm,
        context=ContextBuilder(
            config=ContextConfig(max_tokens=256, reverse_tokens=0),
            llm=llm,
        ),
        system_prompt="你是天气助手。",
        tool_registry=build_registry(),
    )

    result = agent.run("广州天气如何？")

    assert result == "广州是晴天。"
    assert len(llm.calls) == 2

    first_call = llm.calls[0]
    assert first_call["messages"] == [
        {"role": "system", "content": "你是天气助手。"},
        {"role": "user", "content": "广州天气如何？"},
    ]
    assert first_call["tools"][0]["function"]["name"] == "get_weather"

    second_messages = llm.calls[1]["messages"]
    assert second_messages[0] == first_call["messages"][0]
    assert second_messages[1] == first_call["messages"][1]
    assert second_messages[2]["role"] == "assistant"
    assert second_messages[2]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert second_messages[3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "广州是晴天",
    }


test_context_builder_initialization_and_first_context()
test_tool_result_is_added_to_context()
test_long_context_is_compressed_without_mutating_sections()
test_compress_removes_low_priority_optional_section_first()
test_history_compression_uses_llm_summary_and_keeps_recent_message()
test_history_compression_falls_back_to_recent_messages_when_llm_fails()
test_task_compression_preserves_head_and_tail()
test_required_sections_are_fitted_when_optional_sections_are_gone()
test_context_contains_history_on_each_build()
test_each_llm_call_receives_tool_call_context()