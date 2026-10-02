from __future__ import annotations

from typing import Any, Awaitable, Callable

from taiyi_agent.agents.agent_step import AgentStepExecutor
from taiyi_agent.sessions.session import SessionState
from taiyi_agent.context.context_assembler import ContextAssembler


CheckpointCallback = Callable[
    [SessionState],
    Awaitable[None],
]


class ReactAgent:
    """
    ReAct 循环 Agent。

    负责：
        1、一次用户请求的生命周期；
        2、ReAct 步骤循环；
        3、最大步数；
        4、最终答案判断；
        5、Session 的开始、更新和结束。
    """

    agent_type = "react"

    def __init__(
        self,
        step_executor: AgentStepExecutor,
        context_assembler: ContextAssembler,
        agent_system_prompt: str | None = None,
        max_steps: int = 6,
    ) -> None:
        self.step_executor = step_executor
        self.context_assembler = context_assembler
        self.agent_system_prompt = agent_system_prompt
        self.max_steps = max_steps

    async def run(
        self,
        session: SessionState,
        user_input: str,
        *,
        max_steps: int | None = None,
        checkpoint: CheckpointCallback | None = None,
    ) -> str:
        """
        执行一次完整用户请求。

        一个 run 可以包含多个：
        LLM -> tool call -> tool result -> LLM
        步骤。
        """
        if session.active_turn:
            raise RuntimeError(f"Session {session.session_id} 已经存在运行中的 turn")

        step_limit = max_steps or self.max_steps

        session.begin_turn(user_input)

        try:
            working_messages = await self.context_assembler.build_initial_messages(
                session=session,
                system_prompt_override=self.context_assembler.resolve_system_prompt(
                    session=session, 
                    agent_default=self.agent_system_prompt
                ),
            )

            working_messages = (
                self.step_executor.prepare_messages(working_messages)
            )

            session.replace_working_messages(working_messages)

            for step_index in range(step_limit):
                session.set_step_index(step_index)

                # 每次llm调用前，需要检查输入消息, 如果超限，执行压缩
                working_messages = await self.context_assembler.prepare_step_messages(
                    session=session,
                    working_messages=working_messages
                )
                result = await self.step_executor.execute(working_messages)

                session.replace_working_messages(working_messages)
                session.record_step(result)

                if checkpoint is not None:
                    await checkpoint(session)

                if result.finished:
                    final_content = result.content

                    session.complete_turn(assistant_output=final_content,)

                    if checkpoint is not None:
                        await checkpoint(session)

                    return final_content

            error_message = (
                f"Error: ReAct 迭代次数超过限制 "
                f"({step_limit})"
            )

            session.complete_turn(assistant_output=error_message,)

            if checkpoint is not None:
                await checkpoint(session)

            return error_message

        except Exception as exc:
            session.abort_turn(error=str(exc))

            if checkpoint is not None:
                await checkpoint(session)

            raise
