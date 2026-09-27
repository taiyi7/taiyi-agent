from taiyi_agent.tool.tool_base import BaseTool, ToolParameter
from typing import Any
import httpx

class WebSearchTool(BaseTool):
    name="网络搜索工具"
    description="网络搜索工具，整和多个搜索源，当问题大模型无法回答时(如实时问题)，使用搜索工具进行搜索"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="question",
                type="string",
                description="需要进行网络查询的问题内容",
                required=True
            )
        ]
    def run(self, quesiton: str) -> str:
        pass

    async def arun(self, parameters: dict[str, Any]) -> str:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(self.endpoint, json=parameters)
            response.raise_for_status()
            return response.text