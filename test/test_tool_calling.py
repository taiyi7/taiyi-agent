
from taiyi_agent.agents.react_agent import ReactAgent
from taiyi_agent.agents.agent_step import AgentStepExecutor
from taiyi_agent.tool.tool_base import ToolParameter
from taiyi_agent.tool.tool_calling import PromptToolCallingStrategy, ToolCallingStrategy, NativeToolCallingStrategy
from taiyi_agent.tool.tool_registry import ToolRegistry
from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.sessions.session import SessionState

import asyncio

llm = TaiyiAgentLLM()
session = SessionState()

def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(
        name="get_weather",
        description="查询指定城市天气",
        func=lambda city: f"{city}是晴天",
        parameters=[
            ToolParameter(
                name="city",
                type="string",
                description="城市名称",
            )
        ],
    )
    return registry


def build_react_agent(
    calling_strategy : ToolCallingStrategy | None = None
) -> ReactAgent:
    agent_step = AgentStepExecutor(
        llm=llm,
        tool_registry= build_registry(),
        tool_calling_strategy=calling_strategy
    )

    return ReactAgent(
        step_executor=agent_step,
        agent_system_prompt="你是天气助手。"
    )

def test_native_tool_calling_executes_registered_tool() -> None:
    agent = build_react_agent()

    result = asyncio.run(agent.run(session=session, user_input="广州今天天气如何"))
    print("result1:", result)

def test_prompt_tool_calling_executes_registered_tool() -> None:
    agent = build_react_agent(PromptToolCallingStrategy())

    result = asyncio.run(agent.run(session=session, user_input="广州今天天气如何"))
    print("result2:", result)

test_native_tool_calling_executes_registered_tool()
print()
test_prompt_tool_calling_executes_registered_tool()
print()
print("history:")
print(session.history)
