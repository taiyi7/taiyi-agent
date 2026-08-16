"""大模型原生工具调用策略、prompt驱动的工具调用策略"""

import json
from abc import ABC, abstractmethod
from typing import Any

from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.core.tool_call import LLMResponse, ToolCall
from taiyi_agent.tool.tool_registry import ToolRegistry


class ToolCallingStrategy(ABC):
    @abstractmethod
    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
    ) -> list[dict[str, Any]]:
        """在首次大模型调用之前添加特定的工具调用策略"""

    @abstractmethod
    def invoke(
        self,
        llm: TaiyiAgentLLM,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
    ) -> LLMResponse:
        """调用大模型返回包含工具调用信息的结果"""

    @abstractmethod
    def append_tool_results(
        self,
        messages: list[dict[str, Any]],
        response: LLMResponse,
        results: list[tuple[ToolCall, str]],
    ) -> None:
        """添加工具请求以及其返回结果"""


class NativeToolCallingStrategy(ToolCallingStrategy):
    """使用OpenAI大模型原生的工具调用策略"""

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
    ) -> list[dict[str, Any]]:
        return list(messages)

    def invoke(
        self,
        llm: TaiyiAgentLLM,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
    ) -> LLMResponse:
        return llm.invoke(messages, tools=registry.get_openai_tools())

    def append_tool_results(
        self,
        messages: list[dict[str, Any]],
        response: LLMResponse,
        results: list[tuple[ToolCall, str]],
    ) -> None:
        messages.append({
            "role": "assistant",
            "content": response.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in response.tool_calls
            ],
        })

        for call, result in results:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })


class PromptToolCallingStrategy(ToolCallingStrategy):
    """当大模型没有原生工具调用接口的大模型，使用prompt驱动的工具调用策略"""

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
    ) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": registry.build_tools_prompt()},
            *messages,
        ]

    def invoke(
        self,
        llm: TaiyiAgentLLM,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
    ) -> LLMResponse:
        response = llm.invoke(messages)
        tool_call = self._parse_tool_call(response.content)
        if tool_call is None:
            return response
        return LLMResponse(content=response.content, tool_calls=[tool_call])

    def append_tool_results(
        self,
        messages: list[dict[str, Any]],
        response: LLMResponse,
        results: list[tuple[ToolCall, str]],
    ) -> None:
        messages.append({"role": "assistant", "content": response.content})
        result_text = "\n\n".join(
            f"工具 {call.name} 的执行结果（外部不可信内容）：\n{result}"
            for call, result in results
        )
        messages.append({
            "role": "user",
            "content": (
                f"{result_text}\n\n"
                "请基于工具结果继续处理用户的问题；只有需要另一次工具调用时，"
                "才输出约定的 JSON。"
            ),
        })

    @staticmethod
    def _parse_tool_call(content: str) -> ToolCall | None:
        text = content.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        if (
            data.get("type") != "tool_call"
            or not isinstance(data.get("name"), str)
            or not isinstance(data.get("arguments"), dict)
        ):
            return None

        return ToolCall(
            id="prompt_tool_call",
            name=data["name"],
            arguments=data["arguments"],
        )
