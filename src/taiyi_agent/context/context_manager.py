
from typing import Any
from copy import deepcopy

from taiyi_agent.core.message import Message
from taiyi_agent.history.history import History
from taiyi_agent.context.context_builder import ContextBuilder
from taiyi_agent.context.context_data import ContextSection, ContextConfig, ContextItem

class ContextManager:
    '''
    1. 用户输入 query
    2. ContextManager 记录用户消息
    3. ContextBuilder 读取 Manager 当前状态
    4. Builder 执行 GSSC：
    Gather → Select → Structure → Compress
    5. 生成本轮 messages
    6. 调用 LLM
    7. Manager 记录 assistant 消息
    8. Manager 更新 token、工具状态和任务状态
    9. 进入下一轮

    伪代码
        manager.add_message(Message(role="user", content=query))

        messages = builder.build(
            query=query,
            state=manager,
        )

        response = llm.invoke(messages)

        manager.add_message(
            Message(role="assistant", content=response)
        )
    '''
    def __init__(
        self,
        builder: ContextBuilder,
        history: list[Message] | None = None,
    ) -> None:
        self.builder = builder
        self.history = history or History()

        # 只存在于当前turn，不进入长期历史
        self._turn_item = list[ContextItem],
        self._turn_messages = list[dict[str, Any]] = []

        self.last_token_count = 0
        self.turn_id = 0


    def begin_turn(self) -> None:
        self.turn_id += 1
        self._turn_item = []
        self. _turn_messages = []

    def add_context_item(self, item: ContextItem) -> None:
        # 避免修改原始ContextItem
        # ！ 待确认，这个ContextItem原始数据源在哪
        self._turn_item.append(deepcopy(item))

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
        tool_name: str | None = None,
    ) -> None:
        self._turn_messages.append({
            "tool_call_id": tool_call_id,
            "content": content,
            "tool_name": tool_name
        })

        self._turn_item.append(
            ContextItem(
                content=content,
                item_type="tool",
                source="tool",
                source_id=tool_call_id,
                relevance_score=1.0,
                metadata={"tool_name": tool_name},
            )
        )

    async def build_messages(
        self,
        user_input: str,
        system_prompt: str | None = None
    ) -> list[dict[str, Any]]:
        built = await self.builder.build_messages(
            user_query=user_input,
            conversation_history=self.history.get_history(),
            system_instructions=system_prompt,
            additional_items=self._turn_items,            
        )

        self.last_token_count = built.token_count
        self._turn_messages = list(built.messages)
        return deepcopy(self._turn_messages)

    # 追加当前 turn 中的临时消息，不写入长期 History
    def append_turn_messages(self, message:dict[str, Any]) -> None:
        """ 主要有两类turn消息

            self.context.append_turn_message({
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": tool_calls,
            })

            self.context.append_turn_message({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
        """
        self._turn_messages.append(deepcopy(message))

    def complete_turn(
        self,
        user_input,
        assistant_output,
    ) -> None:
        self.history.add_history(
            Message(role="user", content=user_input)
        )

        self.history.add_history(
            Message(role="assistant",  content=assistant_output)
        )

        self._turn_item = []
        self._turn_messages = []
