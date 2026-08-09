from taiyi_agent.memory.memory_record import MemoryRecord

class MemoryStore:
    async def update(self, memory: MemoryRecord) -> None:
        pass
    async def clear(self) -> None:
        pass