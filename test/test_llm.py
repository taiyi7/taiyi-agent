from taiyi_agent.core.llm import TaiyiAgentLLM
import asyncio

llm = TaiyiAgentLLM()
messages = [{"role": "user", "content": "请简短的介绍你自己"}]

# 同步流式输出
def test_stream():
    response = llm.stream(messages)
    result = ""
    print("\n test_stream:")
    for chunk in response:
        print(chunk, end="", flush=True)
        result += chunk

# 同步非流式输出
def test_invoke():
    print("\n test_invoke:")
    response = llm.invoke(messages)
    print(response)

async def main():

    # 异步流式输出测试
    async def test_astream():
        await asyncio.sleep(5)
        print("\n test_astream:")
        response2 = llm.astream(messages)
        result = ""
        async for chunk in response2:
            print(chunk, end="", flush=True)
            result += chunk

    # 异步非流式输出测试
    async def test_ainvoke():
        print("\n test_ainvoke:")
        response2 = await llm.ainvoke(messages)
        print(response2)

    try:
        await asyncio.gather(test_astream(), test_ainvoke())
    finally:
        await llm.async_client.close()
        llm.client.close()

if __name__ == "__main__":
    test_stream()
    test_invoke()
    asyncio.run(main())