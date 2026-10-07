"""Offline regression tests for pre-step compression and ReAct continuation."""

import asyncio
from copy import deepcopy

import pytest

from taiyi_agent.agents.agent_step import AgentStepExecutor
from taiyi_agent.agents.react_agent import ReactAgent
from taiyi_agent.context.context_assembler import ContextAssembler
from taiyi_agent.context.context_builder import ContextBuilder
from taiyi_agent.context.context_data import ContextConfig, ContextItem
from taiyi_agent.context.token_counter import TokenCounter
from taiyi_agent.context.token_window import TokenWindow
from taiyi_agent.core.message import Message
from taiyi_agent.core.tool_call import LLMResponse, ToolCall
from taiyi_agent.sessions.session import SessionState
from taiyi_agent.tool.tool_calling import PromptToolCallingStrategy
from taiyi_agent.tool.tool_registry import ToolRegistry


class CharacterEncoding:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class SummaryLLM:
    def __init__(self, content="Earlier progress: evidence verified."):
        self.content = content
        self.calls = []

    async def ainvoke(self, messages, **kwargs):
        self.calls.append(deepcopy(messages))
        return LLMResponse(content=self.content)


def make_assembler(llm=None, *, max_tokens=10000, reserved_output_tokens=2000):
    llm = SummaryLLM() if llm is None else llm
    config = ContextConfig(
        max_tokens=max_tokens, reserved_output_tokens=reserved_output_tokens,
    )
    counter = TokenCounter(encoding=CharacterEncoding())
    builder = ContextBuilder(config=config, llm=llm, counter=counter)
    return ContextAssembler(builder, TokenWindow(config, counter))


def make_session():
    session = SessionState(system_prompt="Follow the policy.")
    session.begin_turn("task")
    return session


def base_messages():
    return [
        {"role": "system", "content": "Follow the policy."},
        {"role": "system", "name": "context_summary", "content": ""},
        {"role": "user", "content": "task"},
    ]


def pad_to_count(messages, counter, target):
    padding = target - counter.count_messages(messages)
    assert padding >= 0
    messages[1]["content"] += "x" * padding
    assert counter.count_messages(messages) == target
    return messages


def tool_batch(*call_ids, result="ok"):
    return [
        {
            "role": "assistant", "content": "Using the evidence.",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {
                    "name": "lookup", "arguments": '{"key":"value"}',
                }}
                for call_id in call_ids
            ],
        },
        *[{
            "role": "tool", "tool_call_id": call_id, "content": result,
        } for call_id in call_ids],
    ]


def prepare(assembler, session, messages):
    return asyncio.run(assembler.prepare_step_messages(session, messages))


@pytest.mark.parametrize("used", [5599, 5600])
def test_at_or_below_trigger_returns_original_messages_without_summary(used):
    llm = SummaryLLM()
    assembler = make_assembler(llm)
    messages = pad_to_count(base_messages(), assembler.token_window.counter, used)
    assert prepare(assembler, make_session(), messages) is messages
    assert llm.calls == []


@pytest.mark.parametrize("used", [5601, 12000])
def test_exceeding_trigger_compresses_to_target_using_input_limit(used):
    llm = SummaryLLM()
    assembler = make_assembler(llm)
    messages = pad_to_count(base_messages(), assembler.token_window.counter, used)
    original = deepcopy(messages)
    session = make_session()
    result = prepare(assembler, session, messages)
    assert assembler.token_window.counter.count_messages(result) <= 4000
    assert len(llm.calls) == 1
    assert assembler.token_window.counter.count_messages(llm.calls[0]) <= 8000
    assert result[0] == original[0]
    assert result[-1] == original[-1]
    assert result[1]["name"] == "context_summary"
    assert result[1]["role"] == "assistant"
    assert messages == original
    assert session.active_turn.working_messages == []
    assert prepare(assembler, session, result) is result
    assert len(llm.calls) == 1


def test_all_tool_steps_are_summarized_and_policy_prompts_are_preserved():
    llm = SummaryLLM()
    assembler = make_assembler(llm)
    old = tool_batch("old", result="old evidence")
    latest = tool_batch("a", "b")
    messages = base_messages() + old + latest
    messages.insert(0, {"role": "system", "content": "Tool protocol: JSON only."})
    messages.insert(1, {"role": "developer", "content": "Do not invent evidence."})
    messages[3]["content"] = "rag evidence " + "x" * 6000
    result = prepare(assembler, make_session(), messages)
    assert result[:3] == messages[:3]
    assert len(result) == 5
    assert result[-1] == {"role": "user", "content": "task"}
    assert not any(message.get("role") == "tool" for message in result)
    assert not any(message.get("tool_calls") for message in result)
    assert "old evidence" in llm.calls[0][1]["content"]
    assert "Using the evidence." in llm.calls[0][1]["content"]
    assert "rag evidence" in llm.calls[0][1]["content"]
    assert '"id": "a"' in llm.calls[0][1]["content"]
    assert '"id": "b"' in llm.calls[0][1]["content"]
    assert assembler.token_window.counter.count_messages(result) <= 4000


def test_oversized_latest_batch_is_summarized_as_a_whole():
    llm = SummaryLLM()
    assembler = make_assembler(llm)
    messages = base_messages() + tool_batch("large", result="evidence " + "x" * 6000)
    result = prepare(assembler, make_session(), messages)
    assert not any(message.get("role") == "tool" for message in result)
    assert not any(message.get("tool_calls") for message in result)
    assert "large" in llm.calls[0][1]["content"]
    assert "evidence" in llm.calls[0][1]["content"]
    assert assembler.token_window.counter.count_messages(result) <= 4000


def test_repeated_compaction_includes_previous_summary_and_new_progress():
    llm = SummaryLLM(content="Previously confirmed result.")
    assembler = make_assembler(llm)
    session = make_session()
    messages = pad_to_count(base_messages(), assembler.token_window.counter, 5601)
    first = prepare(assembler, session, messages)
    second = prepare(assembler, session, first + tool_batch("new", result="x" * 6000))
    assert "Previously confirmed result." in llm.calls[1][1]["content"]
    assert "new" in llm.calls[1][1]["content"]
    assert assembler.token_window.counter.count_messages(second) <= 4000


def test_prompt_tool_action_and_observation_are_summarized_together():
    llm = SummaryLLM()
    assembler = make_assembler(llm)
    action = {"role": "assistant", "content": '{"type":"tool_call","name":"lookup"}'}
    observation = {"role": "user", "content": "External tool result: ok. Continue."}
    messages = pad_to_count(base_messages(), assembler.token_window.counter, 5601)
    result = prepare(assembler, make_session(), messages + [action, observation])
    assert action not in result
    assert observation not in result
    assert result[-1] == {"role": "user", "content": "task"}
    assert "tool_call" in llm.calls[0][1]["content"]
    assert observation["content"] in llm.calls[0][1]["content"]


def test_repeated_user_input_preserves_the_current_message():
    llm = SummaryLLM()
    assembler = make_assembler(llm)
    messages = pad_to_count(base_messages(), assembler.token_window.counter, 5601)
    old_user = {"role": "user", "content": "task", "name": "earlier_user"}
    messages.insert(1, old_user)
    result = prepare(assembler, make_session(), messages)
    assert result[-1] == messages[-1]
    assert "earlier_user" in llm.calls[0][1]["content"]


def test_fixed_instructions_over_target_fail_explicitly():
    assembler = make_assembler(SummaryLLM())
    messages = base_messages()
    messages[0]["content"] = "policy " * 1000
    with pytest.raises(RuntimeError, match="压缩目标"):
        prepare(assembler, make_session(), messages)


def test_serialized_summary_envelope_is_included_in_budget():
    llm = SummaryLLM(content='"\\\n' * 1000)
    assembler = make_assembler(llm, max_tokens=1800, reserved_output_tokens=0)
    messages = pad_to_count(base_messages(), assembler.token_window.counter, 1261)
    result = prepare(assembler, make_session(), messages)
    assert assembler.token_window.counter.count_messages(result) <= 900


def test_summary_failure_uses_local_fallback_with_valid_protocol():
    class FailingLLM:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("offline")

    assembler = make_assembler(FailingLLM())
    result = prepare(
        assembler, make_session(),
        base_messages() + tool_batch("huge", result="x" * 6000),
    )
    assert assembler.token_window.counter.count_messages(result) <= 4000
    assert not any(message.get("tool_calls") for message in result)
    assert not any(message.get("role") == "tool" for message in result)


def test_initial_gssc_uses_session_history_and_context_without_mutation():
    assembler = make_assembler(SummaryLLM())
    session = make_session()
    session.history = [Message(role="assistant", content="Historic decision.")]
    session.add_turn_context(ContextItem(
        item_type="rag", content="Verified source.", relevance_score=1.0,
    ))
    before = session.model_dump()
    initial = asyncio.run(assembler.build_initial_messages(session))
    assert "Historic decision." in initial[1]["content"]
    assert "Verified source." in initial[1]["content"]
    assert initial[1]["name"] == "context_summary"
    assert session.model_dump() == before
    rendered = asyncio.run(assembler.builder.build("task", system_instructions="policy"))
    assert "[Task]" in rendered


@pytest.mark.parametrize("prompt_mode", [False, True])
def test_react_continues_multiple_tool_steps_and_a_second_user_turn(prompt_mode):
    class LoopLLM(SummaryLLM):
        def __init__(self):
            super().__init__("Evidence checked; next retrieve the final value.")
            self.agent_calls = []

        async def ainvoke(self, messages, **kwargs):
            if kwargs.get("temperature") == 0.0:
                return await super().ainvoke(messages, **kwargs)
            call_ids = {
                call["id"]
                for message in messages
                for call in message.get("tool_calls", [])
            }
            result_ids = {
                message["tool_call_id"]
                for message in messages if message.get("role") == "tool"
            }
            assert call_ids == result_ids
            assert assembler.token_window.counter.count_messages(messages) < 5600
            self.agent_calls.append(deepcopy(messages))
            step = len(self.agent_calls)
            if step > 2:
                return LLMResponse(content="The final answer.")
            if prompt_mode:
                return LLMResponse(content='{"type":"tool_call","name":"lookup","arguments":{}}')
            return LLMResponse(tool_calls=[ToolCall(id=f"call-{step}", name="lookup")])

    llm = LoopLLM()
    assembler = make_assembler(llm)
    registry = ToolRegistry()
    results = iter(["large evidence " + "x" * 6000, "final value: 42"])
    registry.register_function("lookup", "Retrieve evidence.", lambda: next(results), [])
    executor = AgentStepExecutor(
        llm, registry,
        tool_calling_strategy=PromptToolCallingStrategy() if prompt_mode else None,
    )
    agent = ReactAgent(executor, assembler, agent_system_prompt="Default policy.")
    session = SessionState()
    snapshots = []

    async def checkpoint(state):
        if state.active_turn is not None:
            snapshots.append(deepcopy(state.active_turn.working_messages))

    assert asyncio.run(agent.run(session, "task", checkpoint=checkpoint)) == "The final answer."
    assert len(llm.agent_calls) == 3
    assert len(llm.calls) == 1
    assert session.active_turn is None
    assert len(session.history) == 2
    second_call = llm.agent_calls[1]
    assert second_call[0]["role"] == "system"
    assert any(message.get("content") == "Default policy." for message in second_call)
    assert any(message.get("name") == "context_summary" for message in second_call)
    if prompt_mode:
        assert "lookup" in second_call[0]["content"]
    else:
        assert llm.agent_calls[2][-1]["tool_call_id"] == "call-2"
    assert snapshots
    assert asyncio.run(agent.run(session, "another task")) == "The final answer."
    assert len(session.history) == 4
    assert any(
        "The final answer." in (message.get("content") or "")
        for message in llm.agent_calls[-1]
    )


def test_config_ratios_are_validated_and_shared_dependencies_are_required():
    with pytest.raises(ValueError):
        ContextConfig(compression_target_ratio=0.8)
    assembler = make_assembler()
    with pytest.raises(ValueError, match="ContextConfig"):
        ContextAssembler(assembler.builder, make_assembler().token_window)


def test_trigger_and_target_follow_shared_configuration():
    llm = SummaryLLM()
    assembler = make_assembler(llm)
    config = assembler.token_window.config
    config.compression_trigger_ratio = 0.8
    config.compression_target_ratio = 0.6
    counter = assembler.token_window.counter
    session = make_session()
    below = pad_to_count(base_messages(), counter, 6400)
    assert prepare(assembler, session, below) is below
    reached = pad_to_count(base_messages(), counter, 6401)
    assert counter.count_messages(prepare(assembler, session, reached)) <= 4800
    assert len(llm.calls) == 1


def test_missing_active_turn_and_current_task_fail_explicitly():
    assembler = make_assembler()
    with pytest.raises(RuntimeError):
        asyncio.run(assembler.build_initial_messages(SessionState()))
    with pytest.raises(RuntimeError):
        prepare(assembler, SessionState(), base_messages())
    with pytest.raises(RuntimeError):
        prepare(assembler, make_session(), [{"role": "system", "content": "policy"}])


@pytest.mark.parametrize("from_end", [False, True])
def test_token_truncation_reuses_counter_and_keeps_requested_end(from_end):
    counter = TokenCounter(encoding=CharacterEncoding())
    assert counter.truncate_text("abcdef", 3, from_end=from_end) == (
        "def" if from_end else "abc"
    )
    counter._encoding = None
    result = counter.truncate_text("abcdef你好", 2, from_end=from_end)
    assert counter.count_text(result) <= 2
