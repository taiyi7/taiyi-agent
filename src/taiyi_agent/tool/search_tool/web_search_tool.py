from taiyi_agent.tool.tool_base import BaseTool, ToolParameter

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