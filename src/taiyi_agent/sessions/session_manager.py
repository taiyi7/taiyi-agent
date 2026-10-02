from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any
from uuid import UUID

from taiyi_agent.agents.agent_factory import AgentFactory
from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.sessions.session import SessionState
from taiyi_agent.sessions.session_store import SessionStore
from taiyi_agent.tool.tool_calling import ToolCallingStrategy
from taiyi_agent.tool.tool_registry import ToolRegistry
from taiyi_agent.context.context_assembler import ContextAssembler


class SessionManager:
    """
    管理 Session 生命周期和并发访问。

    SessionManager 负责：
    1. 加载 Session；
    2. 创建 Session；
    3. 获取 Agent；
    4. 防止同一个 Session 并发执行多个 turn；
    5. 保存最终状态和检查点。

    SessionManager 不负责 ReAct 循环。
    """

    def __init__(
        self,
        store: SessionStore,
        agent_factory: AgentFactory,
        llm: TaiyiAgentLLM,
        context_assembler: ContextAssembler,
        tool_registry: ToolRegistry | None = None,
        tool_calling_strategy: ToolCallingStrategy | None = None,
    ) -> None:
        self.store = store
        self.agent_factory = agent_factory
        self.llm = llm
        self.context_assembler = context_assembler
        self.tool_registry = tool_registry
        self.tool_calling_strategy = tool_calling_strategy

        self._session_locks: defaultdict[
            str,
            asyncio.Lock,
        ] = defaultdict(asyncio.Lock)

    async def get_or_create(
        self,
        session_id: UUID,
        *,
        agent_type: str = "react",
        system_prompt: str | None = None,
        max_steps: int = 6,
    ) -> SessionState:
        session = await self.store.get(session_id)

        if session is not None:
            return session

        return SessionState(
            session_id=session_id,
            agent_type=agent_type,
            system_prompt=system_prompt,
            max_steps=max_steps,
        )

    async def send(
        self,
        session_id: UUID,
        user_input: str,
        *,
        agent_type: str = "react",
        system_prompt: str | None = None,
        max_steps: int = 6,
    ) -> str:
        """
        处理一次用户请求。

        一个 send 对应一个完整用户 turn。
        ReAct 内部可能执行多个 Agent step。
        """
        lock = self._session_locks[session_id]

        async with lock:
            session = await self.get_or_create(
                session_id,
                agent_type=agent_type,
                system_prompt=system_prompt,
                max_steps=max_steps,
            )

            agent = self.agent_factory.create(
                session,
                llm=self.llm,
                context_assembler=self.context_assembler,
                tool_registry=self.tool_registry,
                tool_calling_strategy=self.tool_calling_strategy
            )

            async def checkpoint(
                current_session: SessionState,
            ) -> None:
                await self.store.save(current_session)

            try:
                result = await agent.run(
                    session=session,
                    user_input=user_input,
                    max_steps=max_steps,
                    checkpoint=checkpoint,
                )
            except Exception:
                # ReactAgent 已经将当前 turn 标记为 failed。
                await self.store.save(session)
                raise

            await self.store.save(session)
            return result

    async def load(
        self,
        session_id: UUID,
    ) -> SessionState | None:
        return await self.store.get(session_id)

    async def delete(
        self,
        session_id: UUID,
    ) -> None:
        lock = self._session_locks[session_id]

        async with lock:
            await self.store.delete(session_id)