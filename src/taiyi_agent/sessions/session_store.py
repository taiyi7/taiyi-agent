from __future__ import annotations

import asyncio
from uuid import UUID
from copy import deepcopy
from typing import Protocol


from taiyi_agent.sessions.session import SessionState


# Protocol 不是抽象基类 ABC，不需要显式继承，只要你的类实现了这 3 个 async 方法，就自动满足 `SessionStore` 类型
class SessionStore(Protocol):
    async def get(
        self,
        session_id: UUID,
    ) -> SessionState | None:
        ...

    async def save(
        self,
        session: SessionState,
    ) -> None:
        ...

    async def delete(
        self,
        session_id: UUID,
    ) -> None:
        ...


class InMemorySessionStore:
    """开发和测试使用的内存存储。"""

    def __init__(self) -> None:
        self._data: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def get(
        self,
        session_id: UUID,
    ) -> SessionState | None:
        async with self._lock:
            value = self._data.get(session_id)
            return deepcopy(value)

    async def save(
        self,
        session: SessionState,
    ) -> None:
        async with self._lock:
            self._data[session.session_id] = deepcopy(session)

    async def delete(
        self,
        session_id: UUID,
    ) -> None:
        async with self._lock:
            self._data.pop(session_id, None)