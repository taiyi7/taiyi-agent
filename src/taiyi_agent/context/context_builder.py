import asyncio
import json
import math
import os
from copy import deepcopy
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Any

from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.context.context_data import ContextItem, ContextConfig, ContextSection, BuiltContext
from taiyi_agent.core.message import Message
from taiyi_agent.context.token_counter import TokenCounter

load_dotenv()

class ContextBuilder:
    '''
    按照GSSC范式来构造上下文
    '''
    def __init__(
        self,
        # memory_tool: MemoryTool | None = None,
        # rag_tool: RAGTool | None = None,
        config: ContextConfig,
        llm: TaiyiAgentLLM,
        counter: TokenCounter,
    ):
        # self.memory_tool = memory_tool
        # self.rag_tool = rag_tool
        self.config = config
        self.llm = llm
        self.counter = counter

    async def build(
        self,
        user_query: str,
        conversation_history: list[Message] | None = None,
        system_instructions: str | None = None,
        additional_items: list[ContextItem] | None = None,
    ) -> str:
        """
        执行 Gather-Select-Structure-Compress 流水线。

        输出结构为 str
        """
        available_tokens = self.config.input_limit
            
        items = self._gather(
            conversation_history or [],
            system_instructions,
            additional_items or [],
        )
        selected = await self._select(items, user_query, available_tokens)
        sections = self._structure(selected, user_query)
        if not self.config.enable_compression:
            return self._format_all_sections(sections)
        return await self._compress(sections, available_tokens)


    async def build_messages(
        self,
        user_query: str,
        conversation_history: list[Message] | None = None,
        system_instructions: str | None = None,
        additional_items: list[ContextItem] | None = None,
    ) -> BuiltContext:
        """
        执行 Gather-Select-Structure-Compress 流水线。
        
        输出数据结构为 BuiltContext
        """
        
        available_tokens = self.config.input_limit

        items =self._gather(
            conversation_history=conversation_history,
            system_instructions=system_instructions,
            additional_items=additional_items or [],
        )

        selected = await self._select(
            items,
            user_query,
            available_tokens,
        )

        sections = self._structure(selected, user_query)

        formated = (
            self._format_all_sections(sections)
            if not self.config.enable_compression
            else await self._compress(sections, available_tokens)
        )

        messages: list[dict[str, Any]] = []

        # 添加系统提示词
        if system_instructions:
            messages.append({
                "role": "system",
                "content": system_instructions,
            })

        # 当前过渡方案
        # 将压缩后的evidence、 memory、 历史摘要作为额外的 system context
        if formated:
            messages.append({
                "role": "system",
                "name": "context_summary",
                "content": (
                    "以下是经过本轮筛选和压缩后的上下文信息：\n\n"
                    + formated
                ),
            })

        # 添加当前用户查询的问题
        messages.append({
            "role": "user",
            "content": user_query, 
        })

        return BuiltContext(
            messages=messages,
            token_count=self.counter.count_messages(messages),
            selected_items=selected,
            metadata={
                "section_count": len(sections),
                "turn_context": True,
            },
        )


    async def compress_trace(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> str:
        """
        将已经完成的 Agent 消息压缩成文本摘要。
        
        主要用于对话过程中 token 超限压缩
        """
        if not messages or max_tokens <= 0:
            return ""

        formatted = [
            json.dumps(message, ensure_ascii=False, default=str)
            for message in messages
        ]

        summary = await self.summarize_messages(
            messages=formatted,
            max_tokens=max_tokens
        )

        return summary


    def _gather(
        self,
        conversation_history: list[Message],
        system_instructions: str | None,
        additional_items: list[ContextItem]
    ) -> list[ContextItem]:

        """汇集所有候选信息

        Args:
            conversation_history: 对话历史
            system_instructions: 系统指令
            additional_items: 自定义信息包

        Returns:
            List[ContextPacket]: 候选信息列表
        """
        items = []

        # 0、 系统指令
        if system_instructions:
            items.append(ContextItem(
                content=system_instructions,
                item_type="system",
                relevance_score=1.0,   # 系统指令相关性为1.0
                token_count=self.counter.count_text(system_instructions),
            ))

        # 1、 从记忆中获取任务状态与关键结论
        # if self.memory_tool:

        # 2、 从RAG中获取事实证据
        # if self.rag_tool:

        # 3、 收集对话历史
        if conversation_history:
            history_limit = self.config.max_history_messages
            recent_history = (
                conversation_history[-history_limit:]
                if history_limit > 0
                else []
            )
            for msg in recent_history:
                rendered_message = f"[{msg.role}] {msg.content}"
                items.append(ContextItem(
                    content=rendered_message,
                    item_type="history",
                    timestamp=msg.timestamp,
                    relevance_score=0.6,   # 历史对话相关性为0.6
                    token_count=self.counter.count_text(rendered_message),
                ))

        # 4、 添加额外上下文
        if additional_items:
            items.extend(additional_items)

        print(f"[ContextBuilder] _gather汇聚了{len(items)}个候选上下文条目(ContextItem)")
        return items

    async def _select(
        self,
        items: list[ContextItem],
        user_query: str,
        available_tokens: int,
    ) -> list[ContextItem]:
        """选择最相关的上下文条目

        Args:
            items: 候选上下文清单
            user_query: 用户查询(用于计算相关性)
            available_tokens: 可用的 token 数量

        Returns:
            List[ContextPacket]: 选中的信息包列表
        """
        # 1、系统指令和历史属于基础上下文，统一交给 Compress 阶段处理。
        system_items = [i for i in items if i.item_type == "system"]
        history_items = [i for i in items if i.item_type == "history"]
        other_items = [
            i for i in items if i.item_type not in {"system", "history"}
        ]

        # 2、计算基础上下文所占的 token
        base_items = system_items + history_items
        base_tokens = sum(i.token_count for i in base_items)
        remaining_tokens = available_tokens - base_tokens

        if remaining_tokens <= 0:
            print("基础上下文已经达到 token 预算，将交给压缩阶段处理")
            return base_items

        # 3、计算其他item的综合分数
        scored_items = []
        for item in other_items:
            # 计算相关性分数
            if item.relevance_score == 0.0:    # 默认值，需要计算相关性分数
                relevance = await self._calculate_relevance(item.content, user_query)
                item.relevance_score = relevance

            # 计算新鲜度分数
            recency = self._calculate_recency(item.timestamp)

            # 计算综合分数
            combined_score = (
                self.config.relevance_weight * item.relevance_score +
                self.config.recency_weight * recency
            )

            # 过滤低于最小相关性的消息
            if item.relevance_score >= self.config.min_relevance:
                scored_items.append((combined_score, item))

        # 4、按照分数降序排序
        scored_items.sort(key=lambda x:x[0], reverse=True)

        # 5、 按照分数从高到低填充，直到token上限
        selected = base_items.copy()
        current_tokens = base_tokens

        for score, item in scored_items:
            if current_tokens + item.token_count <= available_tokens:
                selected.append(item)
                current_tokens += item.token_count
            # 当前条目过大时继续尝试后续较小条目。

        print(f"[ContextBuilder] _select 选择了 {len(selected)} 个信息包,共 {current_tokens} tokens")
        return selected

    def _structure(
        self,
        selected_items: list[ContextItem],
        user_query: str,
    ) -> list[ContextSection]:
        """将选中的上下文items组织成结构化的上下文模板

        Args:
            selected_items: 选中的上下文items
            user_query: 用户查询

        Returns:
            list[ContextSection]: 尚未序列化的结构化上下文区块
        """
        system_instructions = []
        evidence = []
        history_items = []
        other_context = []

        for item in selected_items:
            if item.item_type == "system":
                system_instructions.append(item.content)
            elif item.item_type == "rag":
                evidence.append(item.content)
            elif item.item_type == "history":
                history_items.append(item)
            else:
                other_context.append(item.content)

        history_items.sort(key=lambda item: item.timestamp)

        sections: list[ContextSection] = []

        # 1、[Role & Policies]
        if system_instructions:
            sections.append(ContextSection(
                section_type="role_policies",
                title="[Role & Policies]",
                items=system_instructions,
                priority=100,
                required=True,
            ))

        # 2、[Task]
        sections.append(ContextSection(
            section_type="task",
            title="[Task]",
            items=[user_query],
            priority=90,
            required=True,
        ))

        # 3、[Evidence]
        if evidence:
            sections.append(ContextSection(
                section_type="evidence",
                title="[Evidence]",
                items=evidence,
                priority=60,
            ))

        # 4、[Recent Conversation] --history
        if history_items:
            sections.append(ContextSection(
                section_type="history",
                title="[Recent Conversation]",
                items=[item.content for item in history_items],
                priority=50,
                metadata={"message_count": len(history_items)},
            ))

        # 5、[Context]
        if other_context:
            sections.append(ContextSection(
                section_type="context",
                title="[Context]",
                items=other_context,
                priority=40,
            ))

        # 6、 [Output]"
        sections.append(ContextSection(
            section_type="output",
            title="[Output]",
            items=["请基于以上信息,提供准确、有据的回答。"],
            priority=20,
        ))

        return sections

    async def _compress(
        self,
        sections: list[ContextSection],
        max_token: int,
    ) -> str:
        """按区块优先级压缩上下文，并保证最终结果不超过预算。"""
        if max_token <= 0:
            return ""

        working = deepcopy(sections)
        formatted, used_token = self._formatted_with_tokens(working)
        if used_token <= max_token:
            return formatted

        print(
            f"[ContextBuilder] _compress 上下文超限current_tokens: {used_token} > "
            f"max_token:{max_token}。进行压缩"
        )

        # 从最低优先级开始压缩可选区块。每个区块最多处理一次，
        optional_sections = sorted(
            (section for section in working if not section.required),
            key=lambda section: section.priority,
        )
        for section in optional_sections:
            if used_token <= max_token:
                break
            if not section.items:
                continue

            before = self._get_section_body_tokens(section)
            target = max(before - (used_token - max_token), 0)
            await self._compress_section(section, target)

            # 截断器可能因 tokenizer 的粒度无法继续缩短，此时直接移除区块。
            if self._get_section_body_tokens(section) >= before:
                section.items = []

            working = [
                item for item in working
                if item.required or item.items
            ]
            formatted, used_token = self._formatted_with_tokens(working)

        if used_token <= max_token:
            return formatted
        return await self._fit_required_sections(working, max_token)

    async def _compress_history_section(
        self,
        section: ContextSection,
        max_token: int,
    ) -> None:
        """按消息压缩历史：保留最近消息，摘要被省略的旧消息。"""
        messages = list(section.items)
        if not messages or self._get_section_body_tokens(section) <= max_token:
            return

        # 保留历史消息个数
        keep_count = min(
            max(self.config.history_keep_recent, 0),
            len(messages),
        )
        summary_header = "[Earlier Conversation Summary]\n"
        can_summarize = (
            len(messages) > 1
            and max_token > self.counter.count_text(summary_header)
        )
        summary_reserve = (
            min(self.config.history_summary_max_tokens, max_token // 3)
            if can_summarize
            else 0
        )
        # 消息拆分成两块：1、完整保留的最近消息。 2、需要压缩的旧消息
        recent_messages, old_messages = self._take_recent_messages(
            messages,
            keep_count,
            max(max_token - summary_reserve, 0),
        )

        summary = ""
        # 如有有旧消息，则进行摘要处理
        if old_messages:
            summary_budget = min(
                self.config.history_summary_max_tokens,
                max(
                    max_token
                    - self.counter.count_text("\n".join(recent_messages))
                    - self.counter.count_text(summary_header),
                    0,
                ),
            )
            if summary_budget > 0:
                summary = await self.summarize_messages(
                    old_messages,
                    summary_budget,
                )

        items: list[str] = []
        if summary:
            items.append(summary_header + summary)
        items.extend(recent_messages)
        section.items = items

        # 拼接边界可能产生少量额外 token；最后只按消息项严格收缩，
        # 不再递归调用历史摘要。
        if self._get_section_body_tokens(section) > max_token:
            section.items = self._fit_recent_messages(
                section.items,
                max_token,
            )

    async def summarize_messages(
        self, 
        messages: list[str],
        max_tokens: int,
        *,
        system_prompt_override: str | None = None
    ) -> str:
        """优先使用 LLM 摘要旧消息，失败时按消息边界降级。"""
        if max_tokens <= 0:
            return ""

        summarize_prompt = (
            "你是对话历史压缩器。只总结给定的旧消息，保留用户目标、"
            "重要约束、关键参数、已确认结论、已执行工具及其关键结果、"
            "文件路径、命令、错误和未完成步骤。"
            "不要编造，不要输出新的分段标题。"
        )
        system_prompt = system_prompt_override or summarize_prompt

        if self.llm is not None:
            try:
                llm_messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"请将以下旧消息压缩到约 {max_tokens} tokens:\n\n"
                            + "\n\n".join(messages)
                        ),
                    }
                ]
                # 超长工具结果也可能撑满摘要请求；摘要输入同样受总预算约束。
                body_budget = self.config.input_limit - self.counter.count_messages([
                    llm_messages[0], {"role": "user", "content": ""},
                ])
                if body_budget <= 0:
                    raise ValueError("输入预算不足以容纳摘要提示词")
                while self.counter.count_messages(llm_messages) > self.config.input_limit:
                    content = llm_messages[1]["content"]
                    llm_messages[1]["content"] = self._head_tail_truncate(content, body_budget)
                    overflow = self.counter.count_messages(llm_messages) - self.config.input_limit
                    body_budget = max(body_budget - max(overflow, 1), 0)
                response = await self.llm.ainvoke(
                    messages=llm_messages,
                    temperature=0.0
                )
                summary = getattr(response, "content", response)
                if isinstance(summary, str) and summary.strip():
                    return self._truncate_text(summary.strip(), max_tokens)
            except Exception as exc:
                print(f"[ContextBuilder] 历史摘要调用失败，使用本地降级策略: {exc}")

        # 降级时仍以消息为单位，优先保留较新的旧消息。
        excerpts = self._fit_recent_messages(messages, max_tokens)
        return "\n".join(excerpts)

    def _take_recent_messages(
        self,
        messages: list[str],
        keep_count: int,
        max_token: int,
    ) -> tuple[list[str], list[str]]:
        """返回：1、预算内的最近消息，2、未纳入窗口的待压缩的旧消息。"""
        if keep_count <= 0:
            return [], list(messages)

        start = max(len(messages) - keep_count, 0)
        candidates = messages[start:]
        recent = self._fit_recent_messages(candidates, max_token)
        omitted_from_window = len(candidates) - len(recent)
        old = messages[:start] + candidates[:omitted_from_window]
        return recent, old

    def _fit_recent_messages(self, messages: list[str], max_token: int) -> list[str]:
        """保留完整的最近消息，只有最新单条过长时才截断它。
        适用于：

            - 最近对话消息筛选；
            - LLM 摘要失败后的本地降级；
            - 最终超限校正。
        """
        if max_token <= 0:
            return []

        selected: list[str] = []
        for message in reversed(messages):
            candidate = [message] + selected
            if self.counter.count_text("\n".join(candidate)) <= max_token:
                selected = candidate
                continue

            if not selected:
                excerpt = self._head_tail_truncate(message, max_token)
                if excerpt:
                    selected = [excerpt]
            break
        return selected

    def _head_tail_truncate(self, text: str, max_token: int) -> str:
        """滑动窗口截断：优先保留开头和结尾，中间内容被省略。"""
        if max_token <= 0:
            return ""
        if self.counter.count_text(text) <= max_token:
            return text

        marker = "\n...[middle omitted]...\n"
        marker_count = self.counter.count_text(marker)
        if max_token <= marker_count + 1:
            return self._truncate_text(text, max_token, from_end=True)

        # 按照头内容预算 1/3，尾内容预算 2/3来截断内容
        content_budget = max_token - marker_count
        head_budget = content_budget // 3
        tail_budget = content_budget - head_budget

        # 优先使用编码器编码处理，只保留头和尾巴内容
        if self.counter._encoding is not None:
            tokens = self.counter._encoding.encode(text)
            marker_tokens = self.counter._encoding.encode(marker)
            combined = (
                tokens[:head_budget]
                + marker_tokens
                + (tokens[-tail_budget:] if tail_budget else [])
            )
            return self.counter._encoding.decode(combined)

        head_chars = head_budget * 4
        tail_chars = tail_budget * 4
        tail = text[-tail_chars:] if tail_chars else ""
        return self._truncate_text(
            text[:head_chars] + marker + tail,
            max_token,
        )

    async def _compress_section(self, section: ContextSection, max_token: int) -> None:
        """按区块类型选择相应的压缩策略。"""
        if max_token <= 0:
            section.items = []
            return

        body = "\n".join(section.items)
        if self.counter.count_text(body) <= max_token:
            return

        if section.section_type == "history":
            await self._compress_history_section(section, max_token)
        elif section.section_type in {"task", "context"}:
            section.items = [self._head_tail_truncate(body, max_token)]
        else:
            section.items = [self._truncate_text(body, max_token)]

        section.items = [item for item in section.items if item]

    async def _fit_required_sections(
        self,
        sections: list[ContextSection],
        max_token: int,
    ) -> str:
        """token 极度紧张的场景，只处理标记为 required 的区块：
        `role_policies`（角色策略）、`task`（任务）
        永远优先保留住标题，裁剪正文内容"""
        role = next(
            (
                section
                for section in sections
                if section.section_type == "role_policies"
                and section.required
            ),
            None,
        )
        task = next(
            (
                section
                for section in sections
                if section.section_type == "task" and section.required
            ),
            None,
        )
        if task is None:
            return ""

        # 1、只保留标题
        required = [section for section in (role, task) if section is not None]
        headers_only = [deepcopy(section) for section in required]
        for section in headers_only:
            section.items = []
        overhead = self.counter.count_text(self._format_all_sections(headers_only))

        # 2、只保留标题的情况下依旧超预算，需要进一步降级处理，只保留task部分
        if overhead > max_token:
            if self.counter.count_text(task.title) > max_token:
                return ""

            task_only = deepcopy(task)
            body_budget = max(
                max_token - self.counter.count_text(task.title + "\n"),
                0,
            )
            await self._compress_section(task_only, body_budget)
            formatted = self._format_single_section(task_only)
            if self.counter.count_text(formatted) <= max_token:
                return formatted
            return task.title

        # 3、标题不超预算时，计算正文预算
        body_budget = max_token - overhead
        task_budget = body_budget
        role_budget = 0
        # 按照role和task优先级分配正文预算
        if role is not None:
            total_priority = max(role.priority, 1) + max(task.priority, 1)
            role_budget = body_budget * max(role.priority, 1) // total_priority
            task_budget = body_budget - role_budget

            role_tokens = self._get_section_body_tokens(role)
            task_tokens = self._get_section_body_tokens(task)

            # Role 正文较短时把预算转给 Task，反正亦然
            if role_tokens < role_budget:
                task_budget += role_budget - role_tokens
                role_budget = role_tokens
            elif task_tokens < task_budget:
                role_budget += task_budget - task_tokens
                task_budget = task_tokens

        # 4、按照预算压缩role和task正文
        fitted = [deepcopy(section) for section in required]
        for section in fitted:
            budget = (
                role_budget
                if section.section_type == "role_policies"
                else task_budget
            )
            await self._compress_section(section, budget)

        # 格式化并统计最终 token
        formatted, used_token = self._formatted_with_tokens(fitted)
        # 格式化后有可能因为换行、分词方式等原因导致最终结果超限，需要做校正
        while used_token > max_token:
            # 优先压缩role和task中正文较长的一个
            victim = max(
                (section for section in fitted if section.items),
                key=lambda section: self._get_section_body_tokens(section),
                default=None,
            )
            if victim is None:
                break
            before = self._get_section_body_tokens(victim)
            overflow = used_token - max_token
            await self._compress_section(
                victim,
                max(before - overflow, 0),
            )
            if self._get_section_body_tokens(victim) >= before:
                victim.items = []
            formatted, used_token = self._formatted_with_tokens(fitted)
        return formatted if used_token <= max_token else ""

    @staticmethod
    def _format_single_section(section: ContextSection) -> str:
        body = "\n".join(section.items)
        return section.title + ("\n" + body if body else "")

    def _format_all_sections(self, sections: list[ContextSection]) -> str:
        return "\n\n".join(self._format_single_section(section) for section in sections)

    def _get_section_body_tokens(self, section: ContextSection) -> int:
        return self.counter.count_text("\n".join(section.items))

    def _formatted_with_tokens(
        self,
        sections: list[ContextSection],
    ) -> tuple[str, int]:
        formatted = self._format_all_sections(sections)
        return formatted, self.counter.count_text(formatted)

    def _truncate_text(
        self,
        text: str,
        max_token: int,
        *,
        from_end: bool = False,
    ) -> str:
        return self.counter.truncate_text(text, max_token, from_end=from_end)

    async def _calculate_relevance(self, context: str, query: str) -> float:
        """计算内容和查询问题的相关性,使用向量相似度计算"""
        emb_context, emb_query = await asyncio.gather(
            self._get_embedding(context),
            self._get_embedding(query),
        )

        dot = sum(left * right for left, right in zip(emb_context, emb_query))
        norm_c = math.sqrt(sum(value * value for value in emb_context))
        norm_q = math.sqrt(sum(value * value for value in emb_query))
        if norm_c == 0 or norm_q == 0:
            return 0.0

        score = float(dot / (norm_c * norm_q))     # [-1, 1]
        return (score + 1.0) / 2.0                 # [0, 1]

    async def _get_embedding(self, text: str) -> list[float]:
        """把文本转化为向量"""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "qwen3.7-text-embedding")
        response = await client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding

    def _calculate_recency(self, timestamp: datetime) -> float:
        """按指数衰减计算记忆新鲜度，返回 (0, 1]"""
        now = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            # 写入侧建议统一存 aware UTC；此处对 naive 输入做兜底解释
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        delta = max(0.0, (now - timestamp).total_seconds())   # 未来时间按 0 处理
        half_life = 7 * 24 * 3600                       # 半衰期 7 天（可配置）
        return 0.5 ** (delta / half_life)
