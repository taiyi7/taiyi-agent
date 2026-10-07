'''将函数类方法包装成BaseTool对象'''
import asyncio
import inspect
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
        
        self.is_async = (
            inspect.iscoroutinefunction(func)
            or inspect.iscoroutinefunction(getattr(func, "__call__", None))
        )


    def get_parameters(self) -> list[ToolParameter]:
        return self.parameters

    def run(self, parameters: dict[str, Any]) -> str:
        if self.is_async:
            raise RuntimeError(f"异步工具 {self.name} 不能通过 run() 调用，请使用 arun()")
        return self.func(**parameters)

    async def arun(self, parameters: dict[str, Any]) -> str:
        if self.is_async:
            result = self.func(**parameters)

            if inspect.isawaitable(result):
                result = await result

            return str(result)

        result = await asyncio.to_thread(
            self.func,
            **parameters,
        )
        return str(result)