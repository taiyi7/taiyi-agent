


from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.agents.agent_factory import AgentFactory
from taiyi_agent.sessions.session_store import InMemorySessionStore
from taiyi_agent.sessions.session_manager import SessionManager
from taiyi_agent.sessions.session import SessionState
from taiyi_agent.tool.tool_registry import ToolRegistry
from taiyi_agent.tool.tool_base import ToolParameter
from taiyi_agent.context.context_data import ContextConfig
from taiyi_agent.context.context_assembler import ContextAssembler
from taiyi_agent.context.context_builder import ContextBuilder
from taiyi_agent.context.token_window import TokenWindow, TokenCounter

from fastapi import FastAPI
import uvicorn
import asyncio

llm = TaiyiAgentLLM()
store = InMemorySessionStore()
factory = AgentFactory()
tool_registry = ToolRegistry()
context_cfg = ContextConfig()
counter = TokenCounter()
context_builder = ContextBuilder(config=context_cfg, llm=llm, counter=counter)
token_window = TokenWindow(config=context_cfg, counter=counter)
context_assembler = ContextAssembler(builder=context_builder, token_window=token_window)

def get_weather(date: str, city: str):
    return f"{date}{city}的天气为晴天"

tool_registry.register_function(
    name="get_weather",
    description="查询指定城市的天气",
    func=get_weather,
    parameters=[
        ToolParameter(
            name="date",
            type="string",
            description="日期",
            required=True,
        ),
        ToolParameter(
            name="city",
            type="string",
            description="城市名称",
            required=True,
        )
    ],
)


manager = SessionManager(
    store=store,
    agent_factory=factory,
    llm=llm,
    context_assembler=context_assembler,
    tool_registry=tool_registry,
)
async def test_session():
    session = SessionState()
    await store.save(session)
    answer1 = await manager.send(
        session_id=session.session_id,
        user_input="帮我查询6月16日的天气",
    )
    print("answer1", answer1)
    answer2 = await manager.send(
        session_id=session.session_id,
        user_input="城市为广州",
    )

    print()
    print("answer2", answer2)
    answer3 = await manager.send(
        session_id=session.session_id,
        user_input="那6月17日呢？",
    )
    print("answer3", answer3)
    
    print()
    store_session = await store.get(session_id=session.session_id)
    print("store_session.history", store_session.history)

if __name__ == "__main__":
    asyncio.run(test_session())