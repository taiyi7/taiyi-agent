'''Serp搜索引擎'''
import os
import httpx
from typing import Any
from dotenv import load_dotenv

load_dotenv()
from taiyi_agent.tool.tool_base import BaseTool, ToolParameter

class SerpSearchTool(BaseTool):
    name="Serp网络搜索API"
    description="当通过TavilyAPI无法获取到答案的时候，才使用SerpAPI进行网络搜索"

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
                return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

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
                    return {
                        "success": False,
                        "query": query,
                        "msg": data["error"],
                        "results": []
                    }

            # 智能解析:优先寻找最直接的答案
            if "answer_box_list" in data:
                return "\n".join(data["answer_box_list"])
            if "answer_box" in data and "answer" in data["answer_box"]:
                return data["answer_box"]["answer"]
            if "knowledge_graph" in data and "description" in data["knowledge_graph"]:
                return data["knowledge_graph"]["description"]
            if "organic_results" in data and data["organic_results"]:
                # 如果没有直接答案，则返回前三个结果的摘要
                snippets = [
                    f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                    for i, res in enumerate(data["organic_results"][:3])
                ]
                return "\n\n".join(snippets)
            
            return f"对不起，没有找到关于 '{query}' 的信息。"

        except Exception as exc:
            return f"搜索时发生错误: {exc}"

# serp= SerpSearchTool()
# import asyncio
# ret = asyncio.run(serp.arun({"query":"广东省的美食有哪些？"}))
# print(ret)