import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

os.environ.setdefault(
    "HF_ENDPOINT",
    "https://hf-mirror.com",
)


class EncodingProtocol(Protocol):
    def encode(
        self,
        text: str,
    ) -> list[int]:
        ...

    def decode(
        self,
        tokens: Sequence[int],
    ) -> str:
        ...


class TokenCounter:
    """
    统一 token 计算和文本截断能力。

    支持：
    1. Qwen tokenizer；
    2. tiktoken；
    3. UTF-8 字节数保守估算。
    """

    def __init__(
        self,
        *,
        tokenizer_model: str | None = None,
        encoding: EncodingProtocol | None = None,
    ) -> None:
        self.tokenizer_model = (
            tokenizer_model
            or os.getenv("QWEN_TOKENIZER_MODEL")
        )

        self._encoding = (
            encoding
            if encoding is not None
            else self._load_encoding(tokenizer_model=self.tokenizer_model,)
        )

    def count_text(self, text: str | None) -> int:
        """
        计算普通文本的 token 数量。
        """
        if not text:
            return 0

        if self._encoding is not None:
            return len(self._encoding.encode(text))

        # 没有 tokenizer 时的保守估算。
        return max(1, math.ceil(len(text.encode("utf-8")) / 4))

    def count_message(
        self,
        message: Mapping[str, Any],
    ) -> int:
        """
        计算一条 OpenAI-compatible 消息的 token 数量。

        将 role、content、tool_calls、tool_call_id、
        name 一并计入，避免工具消息被低估。
        """
        normalized = {
            "role": message.get("role"),
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "tool_call_id": message.get("tool_call_id"),
            "name": message.get("name"),
        }

        serialized = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

        return self.count_text(serialized)

    def count_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        """
        计算消息列表的 token 数量。
        """
        return sum(
            self.count_message(message)
            for message in messages
        )

    def truncate_text(
        self, text: str, max_tokens: int, *, from_end: bool = False,
    ) -> str:
        """按 token 预算截断文本；没有编码器时使用计数器查找边界。"""
        if max_tokens <= 0 or not text:
            return ""
        if self._encoding is not None:
            tokens = self._encoding.encode(text)
            selected = tokens[-max_tokens:] if from_end else tokens[:max_tokens]
            text = self._encoding.decode(selected)

        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = text[-middle:] if from_end else text[:middle]
            if self.count_text(candidate) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        return (text[-low:] if low else "") if from_end else text[:low]

    @staticmethod
    def _normalize_model_name(
        model_name: str | None,
    ) -> str:
        dash_map = {
            ord(char): "-"
            for char in "‐‑‒–—―−"
        }

        return ((model_name or "").strip().translate(dash_map))

    @classmethod
    def _load_encoding(
        cls,
        tokenizer_model: str | None,
    ) -> EncodingProtocol | None:
        """
        根据模型加载对应 tokenizer。

        Qwen 模型优先使用 ModelScope 或 Transformers；
        其他模型优先使用 tiktoken。
        """
        tokenizer_name = cls._normalize_model_name(tokenizer_model)

        if not tokenizer_name:
            tokenizer_name = "Qwen/Qwen3-Embedding-0.6B"

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

        class QwenEncoding:
            def encode(self, text: str) -> list[int]:
                try:
                    return tokenizer.encode(text, add_special_tokens=False)
                except TypeError:
                    return tokenizer.encode(text)

            def decode(self, tokens: Sequence[int]) -> str:
                return tokenizer.decode(tokens)

        return QwenEncoding()
