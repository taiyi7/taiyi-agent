'''Tavily搜索引擎'''
import os
from typing import Any
from dotenv import load_dotenv
from tavily import AsyncTavilyClient

load_dotenv()
from taiyi_agent.tool.tool_base import BaseTool, ToolParameter

class TavilySearchTool(BaseTool):
    name="Tavily网络搜索API"
    description="当遇到大模型无法答复的问题，需要网络搜索的时候，优先使用TavilyAPI进行网络搜索"

    def __init__(self):
        self.tavily_client = AsyncTavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="需要使用TavilyAPI进行网络搜索的问题",
                required=True,
            )
        ]

    async def arun(self, parameters: dict[str, Any]) -> str:
        """
        Tavily实时联网检索工具，专为LLM/RAG/AI Agent设计，获取互联网最新公开信息
        适用场景：实时新闻、最新技术文档、赛事数据、政策、产品信息、行业调研等时效性内容查询
        不适用：纯静态常识、数学计算、内置知识库可直接解答的内容，无需调用本工具
        """

        # 提取入参
        query = parameters.get("query")
        if not query:
            return "搜索问题为空，跳过网络搜索"
        print(f"🔍 正在执行 [TavilyApi] 网页搜索: {query}")

        try:
            # 调用API，include_answer=True会返回一个综合性的回答
            response = await self.tavily_client.search(
                query=query,
                search_depth="basic",
                include_answer=True
            )
            
            # Tavily返回的结果已经非常干净，可以直接使用
            # response['answer'] 是一个基于所有搜索结果的总结性回答
            if response.get("answer"):
                return response["answer"]
            
            # 如果没有综合性回答，则格式化原始结果
            formatted_results = []
            for result in response.get("results", []):
                formatted_results.append(f"- {result['title']}: {result['content']}")
            
            if not formatted_results:
                return "抱歉，没有找到相关的问题答案"

            return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

        except Exception as exc:
            return f"错误:执行Tavily搜索时出现问题 - {exc}"

# import asyncio
# tavily = TavilySearchTool()
# ret = asyncio.run(tavily.arun({"query": "今天的油价"}))
# print(ret)