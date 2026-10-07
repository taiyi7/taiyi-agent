"""Taiyi Agent 项目异步测试套件。

运行：
    pytest pytest/test_agent_project.py -q

测试使用 pytest-asyncio 和 unittest.mock，不访问真实 LLM API。
"""

from collections.abc import Sequence
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from taiyi_agent.agents.agent_step import AgentStepExecutor, AgentStepResult
from taiyi_agent.agents.react_agent import CheckpointCallback, ReactAgent
from taiyi_agent.agents.simple_agent import SimpleAgent
from taiyi_agent.context.context_assembler import ContextAssembler
from taiyi_agent.context.context_builder import ContextBuilder
from taiyi_agent.context.context_data import ContextConfig, ContextItem, ContextSection
from taiyi_agent.context.token_counter import TokenCounter
from taiyi_agent.context.token_window import TokenWindow
from taiyi_agent.core.config import Config
from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.core.message import Message
from taiyi_agent.core.tool_call import LLMResponse, ToolCall, ToolExecution
from taiyi_agent.history.history import History
from taiyi_agent.memory.in_memory_store import InMemoryStore
from taiyi_agent.memory.memory_record import MemoryRecord
from taiyi_agent.sessions.session import SessionState
from taiyi_agent.sessions.session_store import InMemorySessionStore
from taiyi_agent.tool.tool_base import ToolParameter
from taiyi_agent.tool.tool_calling import PromptToolCallingStrategy
from taiyi_agent.tool.tool_registry import ToolRegistry


class FixedEncoding:
    """测试用的确定性编码器，避免 ContextBuilder 访问网络加载 tokenizer。"""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))

    def decode(self, tokens: Sequence[int]) -> str:
        return "x" * len(tokens)


@pytest.fixture
def token_counter() -> TokenCounter:
    """提供稳定的 token 计数实现。"""
    return TokenCounter(encoding=FixedEncoding())


@pytest.fixture
def mock_llm() -> Mock:
    """提供不访问网络的异步 LLM mock。"""
    llm = Mock(spec=TaiyiAgentLLM)
    llm.llm_model_id = "test-model"
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def mock_tool() -> ToolRegistry:
    """注册一个可成功执行的本地工具。"""
    registry = ToolRegistry()
    registry.register_function(
        name="lookup_status",
        description="查询服务运行状态",
        func=lambda service: f"{service}: running",
        parameters=[
            ToolParameter(
                name="service",
                type="string",
                description="服务名",
            )
        ],
    )
    return registry


@pytest.fixture
def session_store() -> InMemorySessionStore:
    """每个测试独立使用的内存 SessionStore。"""
    return InMemorySessionStore()


@pytest.fixture
def empty_session() -> SessionState:
    """没有历史和活动 turn 的空 Session。"""
    return SessionState(session_id=uuid4(), max_steps=3)


@pytest.fixture
def context_builder(
    token_counter: TokenCounter,
    mock_llm: Mock,
) -> ContextBuilder:
    return ContextBuilder(
        config=ContextConfig(
            max_tokens=256,
            reserved_output_tokens=0,
            max_history_messages=3,
        ),
        llm=mock_llm,
        counter=token_counter,
    )


# ============================== Core 模块 ==============================


@pytest.mark.asyncio
async def test_core_session_state_transitions() -> None:
    """SessionState 应完成 begin -> step -> complete 的状态流转。"""
    session = SessionState(session_id=uuid4())

    session.begin_turn("你好")
    session.set_step_index(1)
    result = AgentStepResult(response=LLMResponse(content="你好，我是助手。"))
    session.record_step(result)
    session.complete_turn(result.content)

    assert session.active_turn is None
    assert [(item.role, item.content) for item in session.history] == [
        ("user", "你好"),
        ("assistant", "你好，我是助手。"),
    ]
    assert session.revision == 1


@pytest.mark.asyncio
async def test_core_agent_step_result_exposes_finish_and_tool_execution_state() -> None:
    """AgentStepResult 应正确区分最终回答和工具调用步骤。"""
    finished = AgentStepResult(response=LLMResponse(content="完成"))
    pending = AgentStepResult(
        response=LLMResponse(
            tool_calls=[ToolCall(id="t1", name="lookup_status", arguments={})]
        ),
        tool_executions=[
            ToolExecution(
                call=ToolCall(id="t1", name="lookup_status", arguments={}),
                result="running",
            )
        ],
    )

    assert finished.finished is True
    assert finished.content == "完成"
    assert pending.finished is False
    assert pending.tool_executions[0].result == "running"


@pytest.mark.asyncio
async def test_core_message_serializes_tool_metadata() -> None:
    """Core Message 应保留原生工具调用所需的元数据。"""
    message = Message(
        role="tool",
        content="api: running",
        tool_call_id="call-1",
        name="lookup_status",
    )

    assert message.to_openai_dict() == {
        "role": "tool",
        "content": "api: running",
        "tool_call_id": "call-1",
        "name": "lookup_status",
    }


# ============================== Agent 模块 ==============================


@pytest.mark.asyncio
async def test_agent_simple_run_uses_sync_llm_and_updates_history(
    mock_llm: Mock,
) -> None:
    """SimpleAgent.run 应调用同步 LLM，并保存问答历史。"""
    mock_llm.invoke.return_value = LLMResponse(content="同步回答")
    agent = SimpleAgent(
        name="simple",
        llm=mock_llm,
        system_prompt="你是助手。",
    )

    result = agent.run("同步问题")

    assert result == "同步回答"
    mock_llm.invoke.assert_called_once()
    assert [message.content for message in agent.history.get_history()] == [
        "同步问题",
        "同步回答",
    ]


@pytest.mark.asyncio
async def test_agent_simple_async_run_uses_async_llm_and_updates_history(
    mock_llm: Mock,
) -> None:
    """SimpleAgent.async_run 应调用异步 LLM，并保存问答历史。"""
    mock_llm.ainvoke.return_value = LLMResponse(content="异步回答")
    agent = SimpleAgent(name="simple", llm=mock_llm)

    result = await agent.async_run("异步问题")

    assert result == "异步回答"
    mock_llm.ainvoke.assert_awaited_once()
    assert [message.content for message in agent.history.get_history()] == [
        "异步问题",
        "异步回答",
    ]


@pytest.mark.asyncio
async def test_agent_react_runs_multiple_tool_rounds_then_finishes(
    mock_llm: Mock,
    mock_tool: ToolRegistry,
    empty_session: SessionState,
) -> None:
    """ReAct 应支持多轮 LLM -> tool -> LLM，并在最终回答时停止。"""
    mock_llm.ainvoke.side_effect = [
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="lookup_status",
                    arguments={"service": "api"},
                )
            ]
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="call-2",
                    name="lookup_status",
                    arguments={"service": "worker"},
                )
            ]
        ),
        LLMResponse(content="api 和 worker 都正常。"),
    ]
    executor = AgentStepExecutor(mock_llm, mock_tool)
    agent = ReactAgent(executor, agent_system_prompt="你是运维助手。", max_steps=5)

    result = await agent.run(empty_session, "检查服务", checkpoint=None)

    assert result == "api 和 worker 都正常。"
    assert mock_llm.ainvoke.await_count == 3
    assert empty_session.active_turn is None
    assert empty_session.history[-1].content == result


@pytest.mark.asyncio
async def test_agent_react_stops_at_step_limit_without_infinite_loop(
    mock_llm: Mock,
    mock_tool: ToolRegistry,
    empty_session: SessionState,
) -> None:
    """模型持续请求工具时，ReAct 必须在 max_steps 到达后终止。"""
    mock_llm.ainvoke.return_value = LLMResponse(
        tool_calls=[
            ToolCall(
                id="loop",
                name="lookup_status",
                arguments={"service": "api"},
            )
        ]
    )
    agent = ReactAgent(
        AgentStepExecutor(mock_llm, mock_tool),
        max_steps=2,
    )

    result = await agent.run(empty_session, "持续检查", max_steps=2)

    assert "超过限制" in result
    assert mock_llm.ainvoke.await_count == 2
    assert empty_session.active_turn is None
    assert empty_session.history[-1].content == result


@pytest.mark.asyncio
async def test_agent_react_aborts_and_reraises_step_exception(
    mock_llm: Mock,
    mock_tool: ToolRegistry,
    empty_session: SessionState,
) -> None:
    """单步异常应结束活动 turn，保留异常给调用方处理。"""
    mock_llm.ainvoke.side_effect = RuntimeError("llm unavailable")
    agent = ReactAgent(AgentStepExecutor(mock_llm, mock_tool))

    with pytest.raises(RuntimeError, match="llm unavailable"):
        await agent.run(empty_session, "失败请求")

    assert empty_session.active_turn is None
    assert empty_session.history == []


# ============================== History 模块 ==============================


@pytest.mark.asyncio
async def test_history_append_and_clear() -> None:
    """History 应追加消息，并在 clear_history 后为空。"""
    history = History()
    history.add_history(Message(role="user", content="问题"))
    history.add_history(Message(role="assistant", content="答案"))

    assert [message.content for message in history.get_history()] == [
        "问题",
        "答案",
    ]
    history.clear_history()
    assert history.get_history() == []


@pytest.mark.asyncio
async def test_history_window_limits_messages_during_context_gather(
    context_builder: ContextBuilder,
) -> None:
    """History 本身保留消息，ContextBuilder 应按 max_history_messages 截断输入。"""
    history = [
        Message(role="user", content=f"消息 {index}")
        for index in range(5)
    ]

    gathered = context_builder._gather(
        conversation_history=history,
        system_instructions=None,
        additional_items=[],
    )

    assert len(gathered) == 3
    assert [item.content for item in gathered] == [
        "[user] 消息 2",
        "[user] 消息 3",
        "[user] 消息 4",
    ]


# ============================== Context 模块 ==============================


@pytest.mark.asyncio
async def test_context_gssc_builds_system_context_and_user_messages(
    context_builder: ContextBuilder,
) -> None:
    """GSSC 流程应汇聚、筛选、结构化并输出 OpenAI 消息。"""
    built = await context_builder.build_messages(
        user_query="部署步骤是什么？",
        conversation_history=[
            Message(role="user", content="项目使用 PostgreSQL。")
        ],
        system_instructions="你是项目助手。",
        additional_items=[
            ContextItem(
                content="部署前必须运行迁移。",
                item_type="rag",
                relevance_score=1.0,
                token_count=12,
            )
        ],
    )

    assert built.messages[0] == {
        "role": "system",
        "content": "你是项目助手。",
    }
    assert built.messages[-1] == {
        "role": "user",
        "content": "部署步骤是什么？",
    }
    assert "部署前必须运行迁移。" in built.messages[1]["content"]
    assert built.metadata["turn_context"] is True
    assert built.token_count > 0


@pytest.mark.asyncio
async def test_context_token_budget_compresses_optional_sections(
    context_builder: ContextBuilder,
) -> None:
    """预算不足时，ContextBuilder 应限制最终上下文长度并保留任务。"""
    sections = [
        ContextSection(
            section_type="task",
            title="[Task]",
            items=["保留这个任务 " * 30],
            priority=90,
            required=True,
        ),
        ContextSection(
            section_type="context",
            title="[Context]",
            items=["长背景 " * 100],
            priority=40,
        ),
    ]

    compressed = await context_builder._compress(
        sections,
        max_token=60,
    )

    assert context_builder.counter.count_text(compressed) <= 60
    assert "[Task]" in compressed


@pytest.mark.asyncio
async def test_context_assembler_requires_active_turn(
    context_builder: ContextBuilder,
    token_counter: TokenCounter,
    empty_session: SessionState,
) -> None:
    """ContextAssembler 没有活动 turn 时应拒绝构建消息。"""
    assembler = ContextAssembler(
        builder=context_builder,
        token_window=TokenWindow(max_tokens=100, counter=token_counter),
    )

    with pytest.raises(RuntimeError, match="活动 turn"):
        await assembler.build_messages(empty_session, "系统")


# ============================== Tool 模块 ==============================


@pytest.mark.asyncio
async def test_tool_call_parser_accepts_json_and_rejects_invalid_input() -> None:
    """Prompt 工具协议应解析合法 JSON，并拒绝非法结构。"""
    strategy = PromptToolCallingStrategy()

    parsed = strategy._parse_tool_call(
        '{"type":"tool_call","name":"lookup_status",'
        '"arguments":{"service":"api"}}'
    )
    invalid = strategy._parse_tool_call("不是 JSON")
    missing_type = strategy._parse_tool_call(
        '{"name":"lookup_status","arguments":{}}'
    )

    assert parsed is not None
    assert parsed.name == "lookup_status"
    assert parsed.arguments == {"service": "api"}
    assert invalid is None
    assert missing_type is None


@pytest.mark.asyncio
async def test_tool_registry_captures_tool_exception_and_missing_tool() -> None:
    """工具异常和未知工具应转换为可传回模型的错误字符串。"""
    registry = ToolRegistry()
    registry.register_function(
        name="explode",
        description="总是失败",
        func=lambda: (_ for _ in ()).throw(ValueError("bad input")),
        parameters=[],
    )

    failed = await registry.aexecute_tool("explode", {})
    missing = await registry.aexecute_tool("missing", {})

    assert "执行工具explode时报错" in failed
    assert "未找到名字为'missing'的工具" in missing


@pytest.mark.asyncio
async def test_tool_registry_reports_missing_required_parameter() -> None:
    """缺少必填参数时，工具执行应返回可供模型处理的错误。"""
    registry = ToolRegistry()
    registry.register_function(
        name="lookup",
        description="查询服务",
        func=lambda service: service,
        parameters=[
            ToolParameter(
                name="service",
                type="string",
                description="服务名",
                required=True,
            )
        ],
    )

    result = await registry.aexecute_tool("lookup", {})

    assert "执行工具lookup时报错" in result


@pytest.mark.asyncio
async def test_tool_execution_appends_result_to_working_context(
    mock_llm: Mock,
    mock_tool: ToolRegistry,
) -> None:
    """AgentStepExecutor 应执行工具并把工具结果注入工作消息。"""
    mock_llm.ainvoke.return_value = LLMResponse(
        tool_calls=[
            ToolCall(
                id="call-1",
                name="lookup_status",
                arguments={"service": "api"},
            )
        ]
    )
    executor = AgentStepExecutor(mock_llm, mock_tool)
    messages = [{"role": "user", "content": "检查 api"}]

    result = await executor.execute(messages)

    assert result.finished is False
    assert result.tool_executions[0].result == "api: running"
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "api: running",
    }
    assert result.appended_messages[-1] == messages[-1]


# ============================== Memory 模块 ==============================


@pytest.mark.asyncio
async def test_memory_store_writes_and_retrieves_by_user() -> None:
    """内存记忆应保存记录，并按用户和查询关键词检索。"""
    store = InMemoryStore()
    record = MemoryRecord(
        user_id="user-1",
        content="项目使用 PostgreSQL 数据库",
        importance=0.9,
    )

    saved = await store.add(record)
    matches = await store.search("user-1", "PostgreSQL", limit=5)

    assert saved.memory_id == record.memory_id
    assert matches[0].content == record.content


@pytest.mark.asyncio
async def test_memory_store_empty_search_is_empty_and_clear_is_scoped() -> None:
    """空记忆检索应返回空列表，按用户清理不影响其他用户。"""
    store = InMemoryStore()
    await store.add(MemoryRecord(user_id="u1", content="alpha"))
    await store.add(MemoryRecord(user_id="u2", content="beta"))

    assert await store.search("missing", "anything") == []
    await store.clear("u1")

    assert await store.search("u1", "alpha") == []
    assert len(await store.search("u2", "beta")) == 1


# ============================== Session 模块 ==============================


@pytest.mark.asyncio
async def test_session_store_save_load_delete_and_empty_session(
    session_store: InMemorySessionStore,
    empty_session: SessionState,
) -> None:
    """内存 SessionStore 应支持创建后的保存、读取和删除。"""
    assert await session_store.get(str(empty_session.session_id)) is None

    await session_store.save(empty_session)
    loaded = await session_store.get(empty_session.session_id)
    assert loaded is not None
    assert loaded.session_id == empty_session.session_id

    await session_store.delete(empty_session.session_id)
    assert await session_store.get(empty_session.session_id) is None


@pytest.mark.asyncio
async def test_session_checkpoint_callback_receives_running_and_completed_states(
    mock_llm: Mock,
    mock_tool: ToolRegistry,
    empty_session: SessionState,
) -> None:
    """ReAct checkpoint 回调应在步骤中和完成后收到 Session 快照。"""
    mock_llm.ainvoke.return_value = LLMResponse(content="完成")
    checkpoints: list[tuple[str, str | None]] = []

    async def checkpoint(session: SessionState) -> None:
        if session.active_turn is None:
            checkpoints.append(("completed", None))
            return
        checkpoints.append((
            session.active_turn.status,
            session.active_turn.final_output,
        ))

    checkpoint_callback: CheckpointCallback = checkpoint

    agent = ReactAgent(
        AgentStepExecutor(mock_llm, mock_tool, enable_tool_calling=False),
        max_steps=2,
    )
    result = await agent.run(
        empty_session,
        "执行",
        checkpoint=checkpoint_callback,
    )

    assert result == "完成"
    assert checkpoints == [("running", None), ("completed", None)]


@pytest.mark.asyncio
async def test_config_from_env_and_dict_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Core Config 应读取环境变量，并提供可序列化字典。"""
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TEMPERATURE", "0.2")

    config = Config.from_env()

    assert config.debug is True
    assert config.log_level == "DEBUG"
    assert config.temperature == 0.2
    assert config.to_dict()["temperature"] == 0.2

