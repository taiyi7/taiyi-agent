from typing import Any

from taiyi_agent.agents.standard_agent import StandardAgent
from taiyi_agent.core.tool_call import LLMResponse, ToolCall
from taiyi_agent.tool.tool_base import ToolParameter
from taiyi_agent.tool.tool_calling import PromptToolCallingStrategy
from taiyi_agent.tool.tool_registry import ToolRegistry
from taiyi_agent.core.llm import TaiyiAgentLLM


llm = TaiyiAgentLLM()

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


def test_native_tool_calling_executes_registered_tool() -> None:
    agent = StandardAgent(
        name="test",
        llm=llm,
        context=object(),
        tool_registry=build_registry(),
    )
    print(agent.run("广州今天天气如何"))

def test_prompt_tool_calling_executes_registered_tool() -> None:
    agent = StandardAgent(
        name="test",
        llm=llm,
        context=object(),
        tool_registry=build_registry(),
        tool_calling_strategy=PromptToolCallingStrategy(),
    )

    print(agent.run("广州今天天气如何"))

test_native_tool_calling_executes_registered_tool()
test_prompt_tool_calling_executes_registered_tool()
