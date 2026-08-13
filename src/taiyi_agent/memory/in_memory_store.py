'''Memory存储基本能力'''
from datetime import datetime, timezone
from taiyi_agent.memory.memory_store import MemoryStore
from taiyi_agent.memory.memory_record import MemoryRecord

class InMemoryStore(MemoryStore):
    '''
    采用内存来存储记忆
    '''
    def __init__(self):
        self._data: dict[str, MemoryRecord] = {}

    async def add(self, memory: MemoryRecord) -> MemoryRecord:
        self._data[memory.memory_id] = memory
        return memory

    async def get(self, memory_id: str) -> MemoryRecord | None:
        return self._data[memory_id]

    async def get_all(self) -> dict[MemoryRecord] | None:
        return self._data

    async def search(self, user_id: str, query: str, limit: int=5) -> list[MemoryRecord]:
        keywords = set(query.lower().split())
        candidates = [
            item for item in self._data.values()
            if item.user_id == user_id
        ]

        candidates.sort(
            key=lambda item:(
                len(keywords & set(item.content.lower().split())),
                item.importance,
                item.updated_at
            ),
            reverse=True
        )

        return candidates[:limit]
    
    async def update(self, memory: MemoryRecord) -> None:
        memory.updated_at = datetime.now(timezone.utc)
        self._data[memory.memory_id] = memory

    async def delete(self, memory_id: str) -> None:
        self._data.pop(memory_id, None)

    async def clear(self, user_id: str | None = None) -> None:
        if user_id == None:
            self._data.clear()
            return

        memory_ids = [
            memory_id for memory_id, memory in self._data.items()
            if memory.user_id == user_id
        ]
        for memory_id in memory_ids:
            del self._data[memory_id]