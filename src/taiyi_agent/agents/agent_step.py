'''
Agent执行「LLM 调用 + tool 解析 + 工具执行」的单步逻辑
'''

import asyncio
from uuid import uuid4
from copy import deepcopy
from pydantic import BaseModel, Field
from typing import Any

from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.core.tool_call import LLMResponse, ToolExecution, ToolCall
from taiyi_agent.tool.tool_calling import (
    NativeToolCallingStrategy,
    ToolCallingStrategy,
)
from taiyi_agent.tool.tool_registry import ToolRegistry



TIME_TOOL_NAME = "datetime"
TIME_CONTEXT_MARKER = "runtime_time_prefetched"
TIME_TOOL_ARGUMENTS = {
    "timezone": "Asia/Shanghai",
}
REALTIME_KEYWORDS = (
    "今年",
    "这个月",
    "实时",
    "当前",
    "现在",
    "今天",
    "刚刚",
    "最新",
    "天气",
    "股价",
    "汇率",
    "新闻",
    "real-time",
    "current",
    "today",
    "latest",
)


class AgentStepResult(BaseModel):
    """一次 Agent 单步执行结果。"""

    response: LLMResponse
    tool_executions: list[ToolExecution] = Field(default_factory=list)

    # 本次步骤追加到 working_messages 的消息。
    # 主要用于 Session 保存检查点和恢复执行。
    appended_messages: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def finished(self) -> bool:
        """没有工具调用时，表示模型已经给出最终回答。"""
        return not self.response.tool_calls

    @property
    def content(self) -> str:
        return self.response.content


class AgentStepExecutor:
    """
    执行一次 Agent 步骤。

    负责：
    1. 调用 LLM；
    2. 通过 ToolCallingStrategy 解析工具调用；
    3. 执行工具；
    4. 把工具调用和工具结果追加到消息列表。

    """

    def __init__(
        self,
        llm: TaiyiAgentLLM,
        tool_registry: ToolRegistry | None = None,
        tool_calling_strategy: ToolCallingStrategy | None = None,
        enable_tool_calling: bool = True,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.enable_tool_calling = (
            enable_tool_calling and tool_registry is not None
        )
        self.tool_calling_strategy = (
            tool_calling_strategy or NativeToolCallingStrategy()
        )

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        执行策略级别的首次消息准备。

        例如 PromptToolCallingStrategy 会在这里加入工具协议提示词。
        """
        if not self.enable_tool_calling:
            return list(messages)

        if self.tool_registry is None:
            raise RuntimeError(
                "工具调用已启用，但 ToolRegistry 未配置"
            )

        return self.tool_calling_strategy.prepare_messages(
            messages,
            self.tool_registry,
        )

    async def execute(
        self,
        working_messages: list[dict[str, Any]],
        *,
        enable_tool_calling: bool | None = None,
    ) -> AgentStepResult:
        """
        执行一次 LLM 调用和工具执行。

        working_messages 会被原有的
        ToolCallingStrategy.append_tool_results() 继续追加消息。
        """
        before_length = len(working_messages)

        use_tool_calling = {
            self.enable_tool_calling
            if enable_tool_calling is None
            else enable_tool_calling
        }
        if use_tool_calling:
            if self.tool_registry is None:
                raise RuntimeError(
                    "工具调用已启用，但 ToolRegistry 未配置"
                )

            response = await self.tool_calling_strategy.ainvoke(
                self.llm,
                working_messages,
                self.tool_registry,
            )
        else:
            response = await self.llm.ainvoke(working_messages)

        if not response.tool_calls:
            return AgentStepResult(
                response=response,
                appended_messages=deepcopy(working_messages[before_length:]),
            )

        if self.tool_registry is None:
            raise RuntimeError("模型返回了工具调用，但 ToolRegistry 未配置")

        executed_calls, results = await self._execute_tool_calls(
            calls=response.tool_calls,
            working_messages=working_messages,
        )

        # 只保留本轮实际执行的工具调用。
        # 如果本轮原本同时返回了 datetime 和 search，
        # 这里只追加 datetime，避免产生不完整的工具调用协议。
        step_response = response.model_copy(
            update={
                "tool_calls": executed_calls,
            }
        )

        executions = [
            ToolExecution(call=call, result=result)
            for call, result in zip(executed_calls, results)
        ]

        pairs = [
            (execution.call, execution.result)
            for execution in executions
        ]

        self.tool_calling_strategy.append_tool_results(
            working_messages,
            step_response,
            pairs,
        )

        return AgentStepResult(
            response=response,
            tool_executions=executions,
            appended_messages=deepcopy(working_messages[before_length:]),
        )

    async def _execute_tool_calls(
        self,
        calls: list[ToolCall],
        working_messages: list[dict[str, Any]],
    ) -> tuple[list[ToolCall], list[str]]:
        if self.tool_registry is None:
            raise RuntimeError("ToolRegistry 未配置")
        if not calls:
            return [], []

        # 已预取时间：全部并行执行
        if any(msg.get("name") == TIME_CONTEXT_MARKER for msg in working_messages):
            results = await asyncio.gather(*[
                self.tool_registry.aexecute_tool(call.name, call.arguments)
                for call in calls
            ])
            return calls, results

        # 模型主动调用datetime：只执行时间工具
        explicit_time_call = next((c for c in calls if c.name == TIME_TOOL_NAME), None)
        if explicit_time_call:
            time_result = await self.tool_registry.aexecute_tool(explicit_time_call.name, explicit_time_call.arguments)
            working_messages.insert(0, {
                "role": "system",
                "name": TIME_CONTEXT_MARKER,
                "content": "当前轮次已经完成时间查询，下一轮请根据时间结果继续处理用户问题。"
            })
            return [explicit_time_call], [time_result]

        # 识别为实时问题，自动注入时间查询，丢弃本轮其他calls
        if self._is_realtime_question(working_messages):
            synthetic_time_call = ToolCall(
                id=f"runtime-time-{uuid4()}",
                name=TIME_TOOL_NAME,
                arguments={"timezone": "Asia/Shanghai"},
            )
            time_result = await self.tool_registry.aexecute_tool(synthetic_time_call.name, synthetic_time_call.arguments)
            working_messages.insert(0, {
                "role": "system",
                "name": TIME_CONTEXT_MARKER,
                "content": "当前轮次已经完成时间查询，下一轮请根据时间结果重新生成工具调用。"
            })
            return [synthetic_time_call], [time_result]

        # 普通问题，并行执行所有工具
        results = await asyncio.gather(*[
            self.tool_registry.aexecute_tool(call.name, call.arguments)
            for call in calls
        ])
        return calls, results


    async def finalize(
        self,
        working_messages: list[dict[str, Any]],
    ) -> str:
        final_messages = [
            *working_messages,
            {
                "role": "user",
                "content": (
                    "工具调用次数已达到上限。"
                    "请根据当前已有的用户需求、历史消息和工具结果，"
                    "直接给出当前最合理的最终答案。"
                    "不要再次调用工具，也不要说明迭代次数限制。"
                ),
            },
        ]

        response = await self.llm.ainvoke(
            final_messages,
            tools=None,
        )

        return response.content.strip()


    @staticmethod
    def _is_realtime_question(
        messages: list[dict[str, Any]],
    ) -> bool:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue

            content = str(message.get("content") or "").lower()

            return any(
                keyword.lower() in content
                for keyword in REALTIME_KEYWORDS
            )

        return False