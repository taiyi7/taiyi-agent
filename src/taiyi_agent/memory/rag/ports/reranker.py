from typing import  Protocol

class RerankModel(Protocol):
    """底层rerank模型契约：只负责文本对打分，不感知业务对象"""
    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """
        输入：query + 候选文本列表
        返回：和texts一一对应的分数列表
        """
        ...


class RerankerPort(Protocol):
    def rerank(
        self,
        query: str,
        hits: list,
        *,
        top_k: int,
    ) -> list:
        ...