# complex_agent.py
from typing import Optional, List, Any, Dict
from taiyi_agent.core.agent import BaseAgent
from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.core.config import Config
from taiyi_agent.core.message import Message
from taiyi_agent.history.history import History
from taiyi_agent.context.context_manager import ContextManager
from taiyi_agent.context.context_builder import ContextBuilder
from taiyi_agent.tool.tool_registry import ToolRegistry
from taiyi_agent.tool.tool_calling import (
    NativeToolCallingStrategy,
    ToolCallingStrategy,
)

import asyncio

class StandardAgent(BaseAgent):
    '''
    包含工具管理、上下文管理、记忆管理等能力的标准Agent
    '''
    def __init__(
            self,
            name: str,
            llm: TaiyiAgentLLM,
            context: ContextManager,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            history: Optional[History] = None,
            tool_registry: Optional[ToolRegistry] = None,
            enable_tool_calling: bool = True,
            tool_calling_strategy: Optional[ToolCallingStrategy] = None,
    ):
        super().__init__(name=name, llm=llm, system_prompt=system_prompt, config=config, history=history)
        self.context = context
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.tool_calling_strategy = tool_calling_strategy or NativeToolCallingStrategy()

    async def async_run(
        self,
        user_input: str,
        max_tool_iterations: int=3,
        **kwargs: Any,
    ) -> str:
        '''StandardAgent的run方法,
        异步执行，实现工具调用，上下文管理等功能
        '''
        self.context.begin_turn()

        messages = await self.context.build_messages(
            user_input=user_input,
            system_prompt=self._build_system_prompt(),
        )

        if not self.enable_tool_calling:
            response = self.llm.invoke(messages).content
            self.context.complete_turn(
                user_input=user_input,
                assistant_output=response
            )
            return response

        return self._run_with_tools(
            messages=messages,
            user_input=user_input,
            max_tool_iterations=max_tool_iterations,
            **kwargs,
        )
        
    def run(
        self,
        user_input: str,
        max_tool_iterations: int=3,
        **kwargs: Any,
    ) -> str:
        """
        将run()方法包装为同步的run()
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.async_run(
                    user_input=user_input, 
                    max_tool_iterations=max_tool_iterations,
                    **kwargs,
                )
            )

        raise RuntimeError("当前时间循环已运行，请使用 await agent.async_run")


    def _run_with_tools(
        self,
        messages: List[Dict[str, Any]],
        user_input: str,
        max_tool_iterations: int,
        **kwargs: Any,
    ) -> str:
        if self.tool_registry is None:
            raise RuntimeError("工具调用已启用，但 ToolRegistry 未配置")

        working_messages = self.tool_calling_strategy.prepare_messages(
            messages,
            self.tool_registry,
        )

        for _ in range(max_tool_iterations):
            response = self.tool_calling_strategy.invoke(
                self.llm,
                working_messages,
                self.tool_registry,
            )

            # 如果某一轮的大模型答复中没有返回tool_calls, 说明已经得到了答案，循环结束返回结果
            if not response.tool_calls:
                self.context.complete_turn(
                    user_input=user_input,
                    assistant_output=response.content
                )
                return response.content

            results = [
                (
                    tool_call,
                    self.tool_registry.execute_tool(
                        tool_call.name,
                        tool_call.arguments,
                    ),
                )
                for tool_call in response.tool_calls
            ]

            # 更新工具调用结果到 working_messages 中
            self.tool_calling_strategy.append_tool_results(
                working_messages,
                response,
                results,
            )

            # 同步到ContextManager的当前turn的状态
            for tool_call, result in results:
                self.context.add_tool_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    content=result,
                )

        error_message = "Error: 工具调用次数超过限制"
        self.context.complete_turn(
            user_input=user_input,
            assistant_output=error_message,
        )
        return error_message


    def _build_messages(self, input: str) -> List[Dict[str, Any]]:
        # 创建初始消息
        messages = []

        # 首先添加系统消息
        system_prompt = self._build_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # 添加历史消息
        for msg in  self.history.get_history():
            messages.append({"role": msg.role, "content": msg.content})

        # 添加用户输入内容
        messages.append({"role": "user", "content": input})

        return messages

    def _build_system_prompt(self) -> str:
        '''构建基础系统提示词；工具协议由策略单独注入。'''
        return self.get_system_prompt() or "你是一个AI助手。"
