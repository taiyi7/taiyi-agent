'''Memory存储基本能力'''
from abc import ABC, abstractmethod
from taiyi_agent.memory.memory_record import MemoryRecord

class MemoryStore(ABC):
    @ abstractmethod
    async def add(self, memory: MemoryRecord) -> MemoryRecord:
        ...

    @ abstractmethod
    async def get(self, memory: MemoryRecord) -> MemoryRecord | None:
        ...

    @ abstractmethod
    async def search(self, user_id: str, query: str, limit: int=5) -> list[MemoryRecord]:
        ...

    @ abstractmethod
    async def update(self, memory: MemoryRecord) -> None:
        ...

    @ abstractmethod
    async def delete(self, memory_id: str) -> None:
        ...

    @ abstractmethod
    async def clear(self, user_id: str | None = None) -> None:
        ...