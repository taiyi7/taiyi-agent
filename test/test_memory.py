import asyncio
from taiyi_agent.memory.memory_record import MemoryRecord
from taiyi_agent.memory.in_memory_store import InMemoryStore

async def main():
    memory_store = InMemoryStore()
    memory_record_1 = MemoryRecord(user_id="700", content="你好，我现在自己 开发 一个 Agent 叫 Taiyi Agent")
    memory_1 = await memory_store.add(memory_record_1)
    print(await memory_store.get(memory_1.memory_id))
    print('============================================')
    memory_record_2 = MemoryRecord(user_id="700", content="你好，我现在正在 开发 Agent的 记忆 模块")
    memory_2 = await memory_store.add(memory_record_2)
    print(await memory_store.get(memory_2.memory_id))
    print('============================================')
    print("获取全部测试：",await memory_store.get_all())
    print('============================================')
    print("查询测试：", await memory_store.search(user_id="700", query="我在 开发 记忆 模块？", limit=1))
    print('============================================')
    await memory_store.delete(memory_id=memory_1.memory_id)
    print("删除测试：", await memory_store.get_all())
    print('============================================')
    memory_record_2.content = "现在该休息了"
    await memory_store.update(memory_record_2)
    print("更新测试：", await memory_store.get_all())
    print('============================================')
    await memory_store.clear(user_id="700")
    print("清空测试：", await memory_store.get_all())

if __name__ == "__main__":
    asyncio.run(main())