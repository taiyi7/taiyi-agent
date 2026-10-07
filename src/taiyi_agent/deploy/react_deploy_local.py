


from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.agents.agent_factory import AgentFactory
from taiyi_agent.sessions.session_store import InMemorySessionStore
from taiyi_agent.sessions.session_manager import SessionManager
from taiyi_agent.sessions.session import SessionState
from taiyi_agent.tool.tool_registry import ToolRegistry
from taiyi_agent.context.context_data import ContextConfig
from taiyi_agent.context.context_assembler import ContextAssembler
from taiyi_agent.context.context_builder import ContextBuilder
from taiyi_agent.context.token_window import TokenWindow, TokenCounter
from taiyi_agent.tool.datetime_tool.get_current_time_tool import GetCurrentTimeTool
from taiyi_agent.tool.search_tool.tavily_search import TavilySearchTool
from taiyi_agent.tool.search_tool.web_search_tool import SerpSearchTool
from taiyi_agent.tool.search_tool.weather_tool import WeatherTool

import asyncio

llm = TaiyiAgentLLM()
store = InMemorySessionStore()
factory = AgentFactory()
context_cfg = ContextConfig()
counter = TokenCounter()
context_builder = ContextBuilder(config=context_cfg, llm=llm, counter=counter)
token_window = TokenWindow(config=context_cfg, counter=counter)
context_assembler = ContextAssembler(builder=context_builder, token_window=token_window)


tool_registry = ToolRegistry()
tool_list = [
    GetCurrentTimeTool(),
    TavilySearchTool(),
    SerpSearchTool(),
    WeatherTool()
]
for tool in tool_list:
    tool_registry.register_tool(tool)


manager = SessionManager(
    store=store,
    agent_factory=factory,
    llm=llm,
    context_assembler=context_assembler,
    tool_registry=tool_registry,
)

async def conversation_loop():
    session = await create_conversation()
    print("请输入内容(q退出):")

    while True:
        user_input= input("请输入:")
        if user_input.lower() == "q":
            print("输入结束")
            return

        answer = await manager.send(session.session_id, user_input)
        print("Agent答复:", answer)

async def create_conversation() -> SessionState:
    session = SessionState()
    await store.save(session)
    return session

if __name__ == "__main__":
    asyncio.run(conversation_loop())