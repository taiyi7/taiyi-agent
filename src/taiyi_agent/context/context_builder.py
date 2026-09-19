import math
import os
from copy import deepcopy
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Any

from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.context.context_data import ContextItem, ContextConfig, ContextSection, BuiltContext
from taiyi_agent.core.message import Message

try:
    import tiktoken
except ImportError:  # tiktoken 是可选依赖，缺失时使用保守估算。
    tiktoken = None

# 使用标准 ASCII 连字符，且允许调用方通过环境变量覆盖镜像地址。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

load_dotenv()

class ContextBuilder:
    '''
    按照GSSC范式来构造上下文
    '''
    def __init__(
        self,
        # memory_tool: MemoryTool | None = None,
        # rag_tool: RAGTool | None = None,
        config: ContextConfig | None = None,
        llm: TaiyiAgentLLM | None = None,
        tokenizer_model: str | None = None,
    ):
        # self.memory_tool = memory_tool
        # self.rag_tool = rag_tool
        self.config = config or ContextConfig()
        self.llm = llm
        self._encoding = self._load_encoding(
            tokenizer_model
            or os.getenv("QWEN_TOKENIZER_MODEL")
            or getattr(llm, "llm_model_id", None)
        )

    async def build(
        self,
        user_query: str,
        conversation_history: list[Message] | None = None,
        system_instructions: str | None = None,
        additional_items: list[ContextItem] | None = None,
    ) -> str:
        """执行 Gather-Select-Structure-Compress 流水线。"""
        available_tokens = max(
            self.config.max_tokens - self.config.reverse_tokens,
            0,
        )
        items = self._gather(
            user_query,
            conversation_history or [],
            system_instructions,
            additional_items or [],
        )
        selected = self._select(items, user_query, available_tokens)
        sections = self._structure(selected, user_query)
        if not self.config.enable_compression:
            return self._format_all_sections(sections)
        return self._compress(sections, available_tokens)

    async def build_messages(
        self,
        user_query: str,
        conversation_history: list[Message] | None = None,
        system_instructions: str | None = None,
        additional_items: list[ContextItem] | None = None,
    ) -> list[]:
        

    def _gather(
        self,
        user_query,
        conversation_history: list[Message],
        system_instructions: str | None,
        additional_items: list[ContextItem]
    ) -> list[ContextItem]:

        """汇集所有候选信息

        Args:
            user_query: 用户查询
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
                token_count=self._count_token(system_instructions),
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
                    token_count=self._count_token(rendered_message),
                ))

        # 4、 添加额外上下文
        if additional_items:
            items.extend(additional_items)

        print(f"ConextItem汇聚了{len(items)}个候选上下文条目")
        return items

    def _select(
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
                relevance = self._calculate_relevance(item.content, user_query)
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

        print(f"[ContextBuilder] 选择了 {len(selected)} 个信息包,共 {current_tokens} tokens")
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

    def _compress(
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
            f"[ContextBuilder]上下文超限current_tokens: {used_token} > "
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
            self._compress_section(section, target)

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
        return self._fit_required_sections(working, max_token)

    def _compress_history_section(
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
            and max_token > self._count_token(summary_header)
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
                    - self._count_token("\n".join(recent_messages))
                    - self._count_token(summary_header),
                    0,
                ),
            )
            if summary_budget > 0:
                summary = self._summarize_messages(
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

    def _summarize_messages(self, messages: list[str], max_token: int) -> str:
        """优先使用 LLM 摘要旧消息，失败时按消息边界降级。"""
        if max_token <= 0:
            return ""

        if self.llm is not None:
            try:
                response = self.llm.invoke([
                    {
                        "role": "system",
                        "content": (
                            "你是对话历史压缩器。只总结给定的旧消息，保留用户目标、"
                            "重要约束、已确认结论、文件路径、命令、错误和未决事项。"
                            "不要编造，不要输出新的分段标题。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"请将以下旧消息压缩到约 {max_token} tokens：\n\n"
                            + "\n\n".join(messages)
                        ),
                    },
                ], temperature=0.0)
                summary = getattr(response, "content", response)
                if isinstance(summary, str) and summary.strip():
                    return self._truncate_text(summary.strip(), max_token)
            except Exception as exc:
                print(f"[ContextBuilder] 历史摘要调用失败，使用本地降级策略: {exc}")

        # 降级时仍以消息为单位，优先保留较新的旧消息。
        excerpts = self._fit_recent_messages(messages, max_token)
        return "\n".join(excerpts)

    def _take_recent_messages(
        self,
        messages: list[str],
        keep_count: int,
        max_token: int,
    ) -> tuple[list[str], list[str]]:
        """返回预算内的最近消息，以及未纳入窗口的旧消息。"""
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
            if self._count_token("\n".join(candidate)) <= max_token:
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
        if self._count_token(text) <= max_token:
            return text

        marker = "\n...[middle omitted]...\n"
        marker_count = self._count_token(marker)
        if max_token <= marker_count + 1:
            return self._truncate_text(text, max_token, from_end=True)

        # 按照头内容预算 1/3，尾内容预算 2/3来截断内容
        content_budget = max_token - marker_count
        head_budget = content_budget // 3
        tail_budget = content_budget - head_budget

        # 优先使用编码器编码处理，只保留头和尾巴内容
        if self._encoding is not None:
            tokens = self._encoding.encode(text)
            marker_tokens = self._encoding.encode(marker)
            combined = (
                tokens[:head_budget]
                + marker_tokens
                + (tokens[-tail_budget:] if tail_budget else [])
            )
            return self._encoding.decode(combined)

        head_chars = head_budget * 4
        tail_chars = tail_budget * 4
        tail = text[-tail_chars:] if tail_chars else ""
        return self._truncate_text(
            text[:head_chars] + marker + tail,
            max_token,
        )

    def _compress_section(self, section: ContextSection, max_token: int) -> None:
        """按区块类型选择相应的压缩策略。"""
        if max_token <= 0:
            section.items = []
            return

        body = "\n".join(section.items)
        if self._count_token(body) <= max_token:
            return

        if section.section_type == "history":
            self._compress_history_section(section, max_token)
        elif section.section_type in {"task", "context"}:
            section.items = [self._head_tail_truncate(body, max_token)]
        else:
            section.items = [self._truncate_text(body, max_token)]

        section.items = [item for item in section.items if item]

    def _fit_required_sections(
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
        overhead = self._count_token(self._format_all_sections(headers_only))

        # 2、只保留标题的情况下依旧超预算，需要进一步降级处理，只保留task部分
        if overhead > max_token:
            if self._count_token(task.title) > max_token:
                return ""

            task_only = deepcopy(task)
            body_budget = max(
                max_token - self._count_token(task.title + "\n"),
                0,
            )
            self._compress_section(task_only, body_budget)
            formatted = self._format_single_section(task_only)
            if self._count_token(formatted) <= max_token:
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
            self._compress_section(section, budget)

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
            self._compress_section(
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
        return self._count_token("\n".join(section.items))

    def _formatted_with_tokens(
        self,
        sections: list[ContextSection],
    ) -> tuple[str, int]:
        formatted = self._format_all_sections(sections)
        return formatted, self._count_token(formatted)

    def _truncate_text(
        self,
        text: str,
        max_token: int,
        *,
        from_end: bool = False,
    ) -> str:
        """以 token 为单位截断文本，避免字符截断破坏编码。"""
        if max_token <= 0 or not text:
            return ""

        # 优先使用实例自带编码器
        if self._encoding is not None:
            tokens = self._encoding.encode(text)
            selected = tokens[-max_token:] if from_end else tokens[:max_token]
            return self._encoding.decode(selected)

        # 当实例编码器为空时，使用二分法查找截断
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = text[-middle:] if from_end else text[:middle]
            if self._count_token(candidate) <= max_token:
                low = middle
            else:
                high = middle - 1
        if from_end:
            return text[-low:] if low else ""
        return text[:low]

    def _calculate_relevance(self, context: str, query: str) -> float:
        """计算内容和查询问题的相关性,使用向量相似度计算"""
        emb_context = self._get_embedding(context)
        emb_query = self._get_embedding(query)

        dot = sum(left * right for left, right in zip(emb_context, emb_query))
        norm_c = math.sqrt(sum(value * value for value in emb_context))
        norm_q = math.sqrt(sum(value * value for value in emb_query))
        if norm_c == 0 or norm_q == 0:
            return 0.0

        score = float(dot / (norm_c * norm_q))     # [-1, 1]
        return (score + 1.0) / 2.0                 # [0, 1]

    def _get_embedding(self, text: str) -> list[float]:
        """把文本转化为向量"""
        from openai import OpenAI
        client = OpenAI()
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "qwen3.7-text-embedding")
        response = client.embeddings.create(
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

    def _count_token(self, text: str):
        """计算文本token数（使用tiktoken）"""
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        # 无 tokenizer 时使用 UTF-8 字节保守估算，并保证短文本至少为 1 token。
        return max(1, math.ceil(len(text.encode("utf-8")) / 4))

    @staticmethod
    def _load_encoding(model: str | None):
        """加载与模型匹配的 tokenizer。

        Qwen 模型使用 Transformers/ModelScope 提供的自定义 tokenizer，
        其他模型继续优先使用 tiktoken。Qwen tokenizer 加载失败时返回
        ``None``，让 ``_count_token`` 使用已有的保守估算，而不是误用
        不匹配的 tiktoken 词表。
        """
        # 配置文件或复制粘贴内容可能带有非 ASCII 连字符，ModelScope
        # 会将其作为模型名的一部分，最终请求不存在的仓库。
        dash_map = {
            ord(char): "-"
            for char in "‐‑‒–—―−"
        }
        model_name = (model or "").strip().translate(dash_map)
        model_key = model_name.lower()
        is_qwen = model_key.startswith("qwen") or model_key.startswith("qwen/")

        if is_qwen:
            # API 模型名不一定是 Hub 路径；允许通过环境变量指定实际的
            # tokenizer，也为常见的 Qwen embedding 别名提供默认映射。
            tokenizer_name = os.getenv("QWEN_TOKENIZER_MODEL")
            if tokenizer_name:
                tokenizer_name = tokenizer_name.strip().translate(dash_map)
            if not tokenizer_name:
                tokenizer_name = (
                    "Qwen/Qwen3-Embedding-0.6B"
                    if "embedding" in model_key
                    else model_name
                )
                if "/" not in tokenizer_name:
                    tokenizer_name = f"Qwen/{tokenizer_name}"

            tokenizer = None
            try:
                from modelscope import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception:
                tokenizer = None

            if tokenizer is None:
                try:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_name,
                        trust_remote_code=True,
                    )
                except Exception:
                    return None

            class QwenTokenWrapper:
                def encode(self, text: str):
                    try:
                        return tokenizer.encode(
                            text,
                            add_special_tokens=False,
                        )
                    except TypeError:
                        # 某些自定义 tokenizer 不接受该可选参数。
                        return tokenizer.encode(text)

                def decode(self, ids):
                    return tokenizer.decode(ids)

            return QwenTokenWrapper()

        if tiktoken is None:
            return None
        if model_name:
            try:
                return tiktoken.encoding_for_model(model_name)
            except Exception:
                pass
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
