from typing import Any

from taiyi_agent.context.token_counter import TokenCounter
from taiyi_agent.context.context_data import ContextConfig


class TokenWindow:
    """
    裁剪历史消息。

    保留：
    1. 所有 system 消息；
    2. 尽可能多的最近消息；
    3. 当前 user 消息。

    工具调用消息必须成组裁剪，
    不能只保留 tool 消息而删除对应 assistant tool_calls。
    """

    def __init__(
        self,
        config: ContextConfig,
        counter: TokenCounter,
    ) -> None:
        self.config = config
        self.counter = counter

    @property
    def max_tokens(self) -> int:
        return self.config.input_limit
    
    def clip(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        system_messages = [
            message
            for message in messages
            if message.get("role") == "system"
        ]

        non_system_messages = [
            message
            for message in messages
            if message.get("role") != "system"
        ]

        selected: list[dict[str, Any]] = []

        for message in reversed(non_system_messages):
            candidate = [*system_messages, message, *selected]

            if self.counter.count_messages(candidate) <= self.max_tokens:
                selected.insert(0, message)
            else:
                break

        result = [*system_messages, *selected]

        # 确保当前用户输入不会被完全裁掉。
        last_user_message = next(
            (
                message
                for message in reversed(non_system_messages)
                if message.get("role") == "user"
            ),
            None,
        )

        if (last_user_message is not None) and (last_user_message not in result):
            result.append(last_user_message)

        return result

    def fit(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        fit 作为更明确的别名。
        """
        return self.clip(messages)