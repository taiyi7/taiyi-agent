"""基于真实 TaiyiAgentLLM 的完整上下文流程集成测试。

这些测试需要 ``LLM_API_KEY``、``LLM_BASE_URL`` 和 ``LLM_MODEL_ID`` 配置。
配置缺失时跳过，而不是构造假的 LLM 或假的 ContextManager。
"""

import asyncio
import os

from taiyi_agent.agents.standard_agent import StandardAgent
from taiyi_agent.context.context_builder import ContextBuilder
from taiyi_agent.context.context_data import ContextConfig, ContextItem
from taiyi_agent.context.context_manager import ContextManager
from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.tool.tool_base import ToolParameter
from taiyi_agent.tool.tool_calling import PromptToolCallingStrategy
from taiyi_agent.tool.tool_registry import ToolRegistry


def _has_llm_configuration() -> bool:
    return all(
        os.getenv(name)
        for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_ID")
    )

# import pytest
# @pytest.fixture(scope="module")
# def llm() -> TaiyiAgentLLM:
#     if not _has_llm_configuration():
#         pytest.skip(
#             "缺少 LLM_API_KEY、LLM_BASE_URL 或 LLM_MODEL_ID，跳过真实 LLM 集成测试"
#         )
#     return TaiyiAgentLLM()


# @pytest.fixture(scope="module")
# def builder(llm: TaiyiAgentLLM) -> ContextBuilder:
#     return ContextBuilder(
#         config=ContextConfig(
#             max_tokens=4096,
#             reverse_tokens=0,
#             max_history_messages=20,
#         ),
#         llm=llm,
#     )


# @pytest.fixture
# def context(builder: ContextBuilder) -> ContextManager:
#     return ContextManager(builder=builder)

llm = TaiyiAgentLLM()
builder = ContextBuilder(
    config=ContextConfig(
        max_tokens=4096,
        reverse_tokens=0,
        max_history_messages=20,
    ),
    llm=llm,
)

context = ContextManager(builder=builder)

def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(
        name="lookup_deployment_status",
        description="查询指定服务的部署状态。需要调用工具时必须使用此工具。",
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


def test_context_builder_and_manager_build_real_messages(
    context: ContextManager,
) -> None:
    """真实 Builder 和 Manager 应生成完整的 system/context/user 消息。"""
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
    assert context.turn_id == 1
    assert context.last_token_count > 0


def test_standard_agent_async_run_uses_real_context_and_llm(
    llm: TaiyiAgentLLM,
    context: ContextManager,
) -> None:
    """异步 Agent 应通过真实 ContextManager 构建消息并完成一轮。"""
    agent = StandardAgent(
        name="real-context-async-agent",
        llm=llm,
        context=context,
        system_prompt="你是简洁的运维助手。只用一句话回答。",
        enable_tool_calling=False,
    )

    result = asyncio.run(agent.async_run("请说明服务健康检查的目的。"))

    assert isinstance(result, str)
    assert result.strip()
    history = context.history.get_history()
    assert [(message.role, message.content) for message in history[-2:]] == [
        ("user", "请说明服务健康检查的目的。"),
        ("assistant", result),
    ]
    assert context._turn_items == []
    assert context._turn_messages == []


def test_standard_agent_sync_run_uses_real_context_and_llm(
    llm: TaiyiAgentLLM,
    builder: ContextBuilder,
) -> None:
    """同步 run 应通过 asyncio 适配器调用真实 async_run。"""
    context = ContextManager(builder=builder)
    agent = StandardAgent(
        name="real-context-sync-agent",
        llm=llm,
        context=context,
        system_prompt="你是简洁的运维助手。只用一句话回答。",
        enable_tool_calling=False,
    )

    result = agent.run("什么是上下文压缩？")

    assert isinstance(result, str)
    assert result.strip()
    history = context.history.get_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "什么是上下文压缩？"
    assert history[1].role == "assistant"
    assert history[1].content == result


def test_standard_agent_tool_call_updates_real_context(
    llm: TaiyiAgentLLM,
    builder: ContextBuilder,
) -> None:
    """工具循环应使用真实 LLM、工具注册表和 ContextManager。"""
    context = ContextManager(builder=builder)
    agent = StandardAgent(
        name="real-tool-context-agent",
        llm=llm,
        context=context,
        system_prompt=(
            "你是服务运维助手。用户要求查询部署状态时，必须调用工具；"
            "拿到工具结果后，用一句话回答。"
        ),
        tool_registry=_make_registry(),
        tool_calling_strategy=PromptToolCallingStrategy(),
    )

    result = agent.run("请调用工具查询 api 服务的部署状态。")

    assert isinstance(result, str)
    assert result.strip()
    history = context.history.get_history()
    assert len(history) == 2
    assert history[0].content == "请调用工具查询 api 服务的部署状态。"
    assert history[1].content == result
    assert context._turn_items == []
    assert context._turn_messages == []

print("\ntest_context_builder_and_manager_build_real_messages:")
test_context_builder_and_manager_build_real_messages(context=context)
print("\ntest_standard_agent_async_run_uses_real_context_and_llm:")
test_standard_agent_async_run_uses_real_context_and_llm(llm=llm, context=context)
print("\ntest_standard_agent_sync_run_uses_real_context_and_llm:")
test_standard_agent_sync_run_uses_real_context_and_llm(llm=llm, builder=builder)
print("\ntest_standard_agent_tool_call_updates_real_context:")
test_standard_agent_tool_call_updates_real_context(llm=llm, builder=builder)
# 设计测试用例，连续调用agnet，可以调用工具，可以查看历史记录