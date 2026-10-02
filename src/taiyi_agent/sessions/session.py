from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

from taiyi_agent.core.message import Message
from taiyi_agent.agents.agent_step import AgentStepResult
from taiyi_agent.context.context_data import ContextItem


SessionStatus = Literal[
    "idle",
    "running",
    "completed",
    "failed",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TurnState(BaseModel):
    """当前用户请求的临时状态。"""

    turn_id: UUID = Field(default_factory=uuid4)
    turn_index: int
    user_input: str
    status: SessionStatus = "running"

    step_index: int = 0

    # 存放额外的上下文：例如 RAG、文件、临时业务上下文等
    context_items:list[ContextItem] = Field(default_factory=list)

    # 当前 ReAct 工作消息，包括：
    # system、history、user、assistant tool_calls、tool。
    working_messages: list[dict[str, Any]] = Field(default_factory=list)

    # 结构化的步骤审计信息。
    step_records: list[dict[str, Any]] = Field(default_factory=list)

    final_output: str | None = None
    error: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None


class SessionState(BaseModel):
    """
    会话状态对象。

    Agent 可以读写当前 turn 状态，
    SessionManager 负责加载和保存它。
    """

    session_id: UUID = Field(default_factory=uuid4)
    agent_type: str = "react"
    system_prompt: str | None = None
    max_steps: int = 6
    max_input_tokens: int = 8192
    persist_trace: bool = False    # turn完成后是否持久化

    history: list[Message] = Field(default_factory=list)
    active_turn: TurnState | None = None

    revision: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def begin_turn(self, user_input: str) -> None:
        if self.active_turn is not None:
            raise RuntimeError(
                f"Session {self.session_id} 已有活动 turn"
            )

        self.active_turn = TurnState(
            turn_index=self.revision + 1,
            user_input=user_input,
        )
        self.revision += 1
        self.touch()

    def replace_working_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        if self.active_turn is None:
            raise RuntimeError("当前没有活动 turn")

        self.active_turn.working_messages = deepcopy(messages)
        self.touch()

    def add_turn_context(
        self,
        item: ContextItem,
    ) -> None:
        if self.active_turn is None:
            raise RuntimeError("当前没有活动 turn")

        self.active_turn.context_items.append(item)
        self.touch()

    def set_step_index(self, step_index: int) -> None:
        if self.active_turn is None:
            raise RuntimeError("当前没有活动 turn")

        self.active_turn.step_index = step_index
        self.touch()

    def record_step(self, result: AgentStepResult) -> None:
        if self.active_turn is None:
            raise RuntimeError("当前没有活动 turn")

        self.active_turn.step_records.append({
            "content": result.response.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in result.response.tool_calls
            ],
            "tool_results": [
                {
                    "tool_call_id": execution.call.id,
                    "tool_name": execution.call.name,
                    "content": execution.result,
                }
                for execution in result.tool_executions
            ],
        })
        self.touch()

    def complete_turn(self, assistant_output: str) -> None:
        if self.active_turn is None:
            raise RuntimeError("当前没有活动 turn")

        turn = self.active_turn
        turn.status = "completed"
        turn.final_output = assistant_output
        turn.finished_at = utc_now()

        # 长期历史默认只保存用户输入和最终回答。
        # ReAct 中间消息保存在 checkpoint/step_records 中。
        self.history.append(
            Message(
                role="user",
                content=turn.user_input,
            )
        )
        self.history.append(
            Message(
                role="assistant",
                content=assistant_output,
            )
        )

        if self.persist_trace:
            self._persist_turn_trace(turn)

        self.active_turn = None
        self.touch()

    def abort_turn(self, error: str) -> None:
        if self.active_turn is None:
            return

        self.active_turn.status = "failed"
        self.active_turn.error = error
        self.active_turn.finished_at = utc_now()

        # 失败时不把半成品回答写入长期对话历史。
        self.active_turn = None
        self.touch()

    def _persist_turn_trace(self, turn: TurnState) -> None:
        self.history.extend(
            Message.from_dict(message)
            for message in turn.working_messages
            if message.get("role") in {
                "assistant",
                "tool",
            }
        )

    def touch(self) -> None:
        self.updated_at = utc_now()