'''将函数类方法包装成BaseTool对象'''
from typing import Callable, Any
from taiyi_agent.tool.tool_base import BaseTool, ToolParameter

class FunctionTool(BaseTool):
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., str],
        parameters: list[ToolParameter]
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters

    def get_parameters(self) -> list[ToolParameter]:
        return self.parameters

    def run(self, parameters: dict[str, Any]) -> str:
        return self.func(**parameters)