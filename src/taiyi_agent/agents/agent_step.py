'''
Agent执行「LLM 调用 + tool 解析 + 工具执行」的单步逻辑
'''

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import Any

from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.core.tool_call import LLMResponse, ToolExecution
from taiyi_agent.tool.tool_calling import (
    NativeToolCallingStrategy,
    ToolCallingStrategy,
)
from taiyi_agent.tool.tool_registry import ToolRegistry



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
    ) -> AgentStepResult:
        """
        执行一次 LLM 调用和工具执行。

        working_messages 会被原有的
        ToolCallingStrategy.append_tool_results() 继续追加消息。
        """
        before_length = len(working_messages)

        if self.enable_tool_calling:
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

        results = await asyncio.gather(*[
            self.tool_registry.aexecute_tool(call.name,call.arguments)
            for call in response.tool_calls
        ])

        executions = [
            ToolExecution(call=call, result=result)
            for call, result in zip(response.tool_calls, results)
        ]

        pairs = [
            (execution.call, execution.result)
            for execution in executions
        ]


        # 将响应和工具调用及工具结果追加到 working_messages
        self.tool_calling_strategy.append_tool_results(
            working_messages,
            response,
            pairs,
        )

        return AgentStepResult(
            response=response,
            tool_executions=executions,
            appended_messages=deepcopy(working_messages[before_length:]),
        )