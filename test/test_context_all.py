"""完整上下文流程测试。

这些用例使用本地 LLM 和上下文替身，不访问外部模型服务。测试同时覆盖：

* ContextBuilder -> ContextManager 的消息构建；
* StandardAgent.run 对上下文生命周期的调用；
* StandardAgent 工具循环中的上下文工具结果同步；
* 一轮完成后用户消息和助手消息写回历史。
"""

import asyncio
from copy import deepcopy
from typing import Any

from taiyi_agent.agents.standard_agent import StandardAgent
from taiyi_agent.context.context_builder import ContextBuilder
from taiyi_agent.context.context_data import ContextConfig, ContextItem
from taiyi_agent.context.context_manager import ContextManager
from taiyi_agent.core.tool_call import LLMResponse, ToolCall
from taiyi_agent.tool.tool_base import ToolParameter
from taiyi_agent.tool.tool_registry import ToolRegistry


class SequenceLLM:
    """符合 TaiyiAgentLLM.invoke 接口的本地响应序列。"""

    llm_model_id = "test-model"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
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
            raise AssertionError("测试 LLM 没有预设响应")
        return self.responses.pop(0)


class RecordingContext:
    """记录 StandardAgent 对上下文的生命周期调用。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.messages = [
            {"role": "system", "content": "context-system"},
            {"role": "user", "content": "context-user"},
        ]

    def begin_turn(self) -> None:
        self.events.append(("begin_turn", None))

    async def build_messages(
        self,
        user_input: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        self.events.append(("build_messages", {
            "user_input": user_input,
            "system_prompt": system_prompt,
        }))
        return deepcopy(self.messages)

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
        tool_name: str | None = None,
    ) -> None:
        self.events.append(("add_tool_result", {
            "tool_call_id": tool_call_id,
            "content": content,
            "tool_name": tool_name,
        }))

    def complete_turn(self, user_input: str, assistant_output: str) -> None:
        self.events.append(("complete_turn", {
            "user_input": user_input,
            "assistant_output": assistant_output,
        }))


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(
        name="lookup_status",
        description="查询服务状态",
        func=lambda service: f"{service}: running",
        parameters=[
            ToolParameter(
                name="service",
                type="string",
                description="服务名称",
            )
        ],
    )
    return registry

# 测试 ContextBuilder 和 ContextManager 构造 messages 流程
def test_context_builder_and_manager_build_complete_messages() -> None:
    builder = ContextBuilder(
        config=ContextConfig(max_tokens=256, reverse_tokens=0),
    )
    context = ContextManager(builder=builder)
    context.begin_turn()
    context.add_context_item(ContextItem(
        content="服务运行在 production 环境。",
        item_type="note",
        relevance_score=1.0,
    ))

    messages = asyncio.run(context.build_messages(
        user_input="检查服务状态",
        system_prompt="你是运维助手。",
    ))

    assert messages[0] == {
        "role": "system",
        "content": "你是运维助手。",
    }
    assert messages[-1] == {
        "role": "user",
        "content": "检查服务状态",
    }
    assert any(
        message["role"] == "system"
        and "production" in message["content"]
        for message in messages
    )
    assert context.last_token_count > 0
    assert context.turn_id == 1

# 测试 standard agent 同步运行上下文构建流程
def test_standard_agent_run_builds_context_before_llm_and_completes_turn() -> None:
    context = RecordingContext()
    llm = SequenceLLM([LLMResponse(content="已完成检查")])
    agent = StandardAgent(
        name="context-agent",
        llm=llm,
        context=context,
        system_prompt="你是运维助手。",
        enable_tool_calling=False,
    )

    result = agent.run("检查服务")

    assert result == "已完成检查"
    assert [event[0] for event in context.events] == [
        "begin_turn",
        "build_messages",
        "complete_turn",
    ]
    assert context.events[1][1] == {
        "user_input": "检查服务",
        "system_prompt": "你是运维助手。",
    }
    assert llm.calls[0]["messages"] == context.messages

# 测试 standard agent 异步运行上下文构建流程
def test_standard_agent_async_run_builds_context_before_llm() -> None:
    context = RecordingContext()
    llm = SequenceLLM([LLMResponse(content="异步检查完成")])
    agent = StandardAgent(
        name="async-context-agent",
        llm=llm,
        context=context,
        system_prompt="你是异步运维助手。",
        enable_tool_calling=False,
    )

    result = asyncio.run(agent.async_run("异步检查服务"))

    assert result == "异步检查完成"
    assert [event[0] for event in context.events] == [
        "begin_turn",
        "build_messages",
        "complete_turn",
    ]
    assert context.events[1][1] == {
        "user_input": "异步检查服务",
        "system_prompt": "你是异步运维助手。",
    }

# 测试 standard agent 工具循环添加工具结果到上下文
def test_standard_agent_tool_loop_adds_result_to_context_and_next_llm_call() -> None:
    context = RecordingContext()
    llm = SequenceLLM([
        LLMResponse(
            tool_calls=[ToolCall(
                id="call-1",
                name="lookup_status",
                arguments={"service": "api"},
            )]
        ),
        LLMResponse(content="api 服务正常。"),
    ])
    agent = StandardAgent(
        name="tool-context-agent",
        llm=llm,
        context=context,
        tool_registry=make_registry(),
    )

    result = agent._run_with_tools(
        messages=deepcopy(context.messages),
        user_input="检查 api 服务",
        max_tool_iterations=2,
    )

    assert result == "api 服务正常。"
    assert len(llm.calls) == 2
    assert llm.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "api: running",
    }
    assert ("add_tool_result", {
        "tool_call_id": "call-1",
        "content": "api: running",
        "tool_name": "lookup_status",
    }) in context.events
    assert context.events[-1] == (
        "complete_turn",
        {
            "user_input": "检查 api 服务",
            "assistant_output": "api 服务正常。",
        },
    )

# 测试 ContextManager 一个turn完成后写入结果到 历史消息 中
def test_context_manager_writes_completed_turn_to_history() -> None:
    builder = ContextBuilder(
        config=ContextConfig(max_tokens=256, reverse_tokens=0),
    )
    context = ContextManager(builder=builder)
    context.begin_turn()

    asyncio.run(context.build_messages(
        user_input="第一轮问题",
        system_prompt="你是助手。",
    ))
    context.complete_turn(
        user_input="第一轮问题",
        assistant_output="第一轮答案",
    )

    history = context.history.get_history()
    assert [(message.role, message.content) for message in history] == [
        ("user", "第一轮问题"),
        ("assistant", "第一轮答案"),
    ]
    assert context._turn_items == []
    assert context._turn_messages == []
