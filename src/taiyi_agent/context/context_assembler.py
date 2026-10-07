from copy import deepcopy
from typing import Any

from taiyi_agent.context.context_builder import ContextBuilder
from taiyi_agent.context.token_window import TokenWindow
from taiyi_agent.sessions.session import SessionState, TurnState


MessageDict = dict[str, Any]


class ContextAssembler:
    '''构造初始上下文，并在每次 LLM 调用前压缩工作消息。'''

    SUMMARY_NAME = "context_summary"
    DEFAULT_SYSTEM_PROMPT = "你是一个 AI 助手。"

    def __init__(
        self,
        builder: ContextBuilder,
        token_window: TokenWindow,
    ) -> None:
        if builder.config is not token_window.config:
            raise ValueError("Builder 和 TokenWindow 必须共享 ContextConfig")
        if builder.counter is not token_window.counter:
            raise ValueError("Builder 和 TokenWindow 必须共享 TokenCounter")

        self.builder = builder
        self.token_window = token_window


    async def build_initial_messages(
        self,
        session: SessionState,
        *,
        system_prompt_override: str | None = None,
    ) -> list[MessageDict]:
        '''使用 ContextBuilder 构造当前 turn 的初始消息。'''
        turn = self._require_turn(session)
        built = await self.builder.build_messages(
            user_query=turn.user_input,
            conversation_history=session.history,
            system_instructions=self.resolve_system_prompt(
                session,
                override=system_prompt_override,
            ),
            additional_items=deepcopy(turn.context_items),
        )
        return built.messages


    async def prepare_step_messages(
        self,
        session: SessionState,
        working_messages: list[MessageDict],
    ) -> list[MessageDict]:
        '''
        Agent 每次调用 LLM 前需要进行消息准备
        超过输入预算的 70% 时，将轨迹摘要压缩到 50% 以内。
        '''
        turn = self._require_turn(session)
        current_user = self._find_current_user(working_messages, turn.user_input)

        if current_user is None:
            raise RuntimeError("工作消息缺少当前用户输入")

        limit = self.token_window.max_tokens
        counter = self.token_window.counter
        config = self.token_window.config
        trigger = limit * config.compression_trigger_ratio

        if counter.count_messages(working_messages) <= trigger:
            return working_messages

        policies, trace = self._separate_policies(
            working_messages,
            current_user=current_user,
        )
        target = int(limit * config.compression_target_ratio)
        fixed = [*policies, current_user]     # 固定不动消息
        remaining = target - counter.count_messages(fixed)

        if remaining < 0:
            raise RuntimeError("系统指令和当前用户输入已超过压缩目标")
        if not trace:
            return deepcopy(fixed)

        # 全部轨迹一起摘要，不再拆分工具批次或保留原始 tool 消息。
        summary = {
            "role": "assistant",
            "name": self.SUMMARY_NAME,
            "content": "",
        }
        budget = remaining - counter.count_message(summary)

        if budget <= 0:
            raise RuntimeError("没有空间保留上下文摘要，请增加输入预算")

        # 压缩和调整 trace 消息
        summary["content"] = await self.builder.compress_trace(
            trace,
            max_tokens=budget
        )
        summary["content"] = self._fit_summary(summary, remaining)

        if not summary["content"].strip():
            raise RuntimeError("上下文压缩未产生有效摘要")
        return deepcopy([*policies, summary, current_user])

    def _separate_policies(
        self,
        messages: list[MessageDict],
        *,
        current_user: MessageDict,
    ) -> tuple[
            list[MessageDict],
            list[MessageDict]
        ]:
        '''将系统消息指令提取出来'''
        policies: list[MessageDict] = []
        trace: list[MessageDict] = []

        for message in messages:
            if message is current_user:
                continue
            is_policy = (
                message.get("role") in {"system", "developer"}
                and message.get("name") != self.SUMMARY_NAME
            )
            if is_policy:
                policies.append(message)
            else:
                trace.append(message)
        return policies, trace

    def _fit_summary(self, summary: MessageDict, budget: int) -> str:
        '''修正摘要消息自身的序列化开销，确保整体仍在预算内。'''
        counter = self.token_window.counter
        text = str(summary.get("content") or "")
        if counter.count_message(summary) <= budget:
            return text
        low, high = 0, counter.count_text(text)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = counter.truncate_text(text, middle)
            if counter.count_message({**summary, "content": candidate}) <= budget:
                low = middle
            else:
                high = middle - 1
        return counter.truncate_text(text, low)

    def _require_turn(self, session: SessionState) -> TurnState:
        if session.active_turn is None:
            raise RuntimeError("当前没有活动 turn")
        return session.active_turn

    @staticmethod
    def _find_current_user(
        messages: list[MessageDict],
        user_input: str,
    ) -> MessageDict | None:
        return next(
            (
                message for message in reversed(messages)
                if message.get("role") == "user"
                and message.get("content") == user_input
            ),
            None,
        )

    def resolve_system_prompt(
        self,
        session: SessionState,
        agent_default: str | None = None,
        override: str | None = None,
    ) -> str:
        for prompt in (override, session.system_prompt, agent_default):
            if prompt is not None:
                return prompt
        return self.DEFAULT_SYSTEM_PROMPT
