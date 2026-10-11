'''Serp搜索引擎'''
import os
import httpx
from typing import Any
from dotenv import load_dotenv
from tavily import AsyncTavilyClient

load_dotenv()
from taiyi_agent.tool.tool_base import BaseTool, ToolParameter


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "查询实时网络信息。系统会优先使用 Tavily；"
        "只有 Tavily 无结果或调用失败时才使用 SerpAPI。"
    )

    def __init__(self):
        self.tavily = TavilySearchTool()
        self.serp = SerpSearchTool()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="需要查询的实时问题",
                required=True,
            )
        ]

    async def arun(self, parameters: dict[str, Any]) -> str:
        tavily_result = await self.tavily.arun(parameters)

        if tavily_result.startswith("[WEB_SEARCH_OK]"):
            return tavily_result

        serp_result = await self.serp.arun(parameters)

        if serp_result.startswith("[WEB_SEARCH_OK]"):
            return serp_result

        return (
            "[WEB_SEARCH_ERROR]\n"
            f"Tavily 和 SerpAPI 均未能返回有效结果。\n"
            f"Tavily:{tavily_result}\n"
            f"SerpAPI:{serp_result}"
        )


class TavilySearchTool(BaseTool):
    name="Tavily网络搜索API"
    description="当遇到大模型无法答复的问题，需要网络搜索的时候，优先使用TavilyAPI进行网络搜索, 如果tavily无法搜索到结果，再使用serpAPI"

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
                print(f"[TavilySearch]-output: {response["answer"]}")
                return f"[WEB_SEARCH_OK]\n{response['answer']}"

            # 如果没有综合性回答，则格式化原始结果
            formatted_results = []
            for result in response.get("results", []):
                formatted_results.append(f"- {result['title']}: {result['content']}")

            if not formatted_results:
                return "[WEB_SEARCH_EMPTY]\n抱歉，没有找到相关的问题答案"

            str_ret = "\n".join(formatted_results)
            print(f"[TavilySearch]-output: {str_ret}")
            return "[WEB_SEARCH_OK]\n根据搜索，为您找到以下信息:\n" + str_ret

        except Exception as exc:
            return f"[WEB_SEARCH_ERROR]\n错误:执行Tavily搜索时出现问题 - {exc}"


class SerpSearchTool(BaseTool):
    name="Serp网络搜索API"
    description="网络搜索API, 默认不使用，只有当通过TavilyAPI无法获取到答案的时候，才使用SerpAPI进行网络搜索"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="需要使用SerpPI进行网络搜索的问题",
                required=True,
            )
        ]

    async def arun(self, parameters: dict[str, Any]) -> str:
        """
        一个基于SerpApi的实战网页搜索引擎工具。
        它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
        """
        query = parameters.get("query")
        if not query:
            return "搜索问题为空，跳过网络搜索"
        print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")

        try:
            api_key = os.getenv("SERPAPI_API_KEY")
            if not api_key:
                return "[WEB_SEARCH_ERROR]\n错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

            params = {
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "gl": "cn",  # 国家代码
                "hl": "zh-cn", # 语言代码
            }

            # 智能解析:优先寻找最直接的答案
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get("https://serpapi.com/search.json", params=params)
                data = resp.json()
                if "error" in data:
                    return f"[WEB_SEARCH_ERROR]\n{data["error"]}"


            # 智能解析:优先寻找最直接的答案
            if "answer_box_list" in data:
                return "[WEB_SEARCH_OK]\n" + "\n".join(data["answer_box_list"])
            if "answer_box" in data and "answer" in data["answer_box"]:
                return "[WEB_SEARCH_OK]\n" + data["answer_box"]["answer"]
            if "knowledge_graph" in data and "description" in data["knowledge_graph"]:
                return "[WEB_SEARCH_OK]\n" + data["knowledge_graph"]["description"]
            if "organic_results" in data and data["organic_results"]:
                # 如果没有直接答案，则返回前三个结果的摘要
                snippets = [
                    f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                    for i, res in enumerate(data["organic_results"][:3])
                ]
                str_ret = "\n".join(snippets)
                print(f"[SerpSearchTool]: {str_ret}")
                return f"[WEB_SEARCH_OK]\n{str_ret}"
            
            return f"[WEB_SEARCH_EMPTY]\n对不起，没有找到关于 '{query}' 的信息。"

        except Exception as exc:
            return f"[WEB_SEARCH_ERROR]\n搜索时发生错误: {exc}"

# serp= SerpSearchTool()
# import asyncio
# ret = asyncio.run(serp.arun({"query":"广东省的美食有哪些？"}))
# print(ret)

# tavily = TavilySearchTool()
# ret = asyncio.run(tavily.arun({"query": "今天的油价"}))
# print(ret)