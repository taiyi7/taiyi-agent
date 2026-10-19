import os
from dotenv import load_dotenv
from taiyi_agent.memory.rag.models import RetrievalHit
import requests
from taiyi_agent.memory.rag.ports.reranker import RerankModel
from taiyi_agent.memory.rag.config import RerankerConfig

load_dotenv()

class AliRerankModel(RerankModel):
    def __init__(self, config: RerankerConfig):
        if not config.base_url:
            raise ValueError("Reranker URL 未配置")
        if not config.api_key:
            raise ValueError("Reranker API key 未配置")
        if not config.model:
            raise ValueError("Reranker model 未配置")

        self.url = config.base_url
        self.api_key = config.api_key
        self.model_name = config.model
        self.timeout = config.timeout
        
    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type":"application/json"}
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": texts
        }
        resp = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        scores = [item["relevance_score"] for item in data["output"]["results"]]
        return scores


class RAGReranker:
    def __init__(
        self,
        model: RerankModel,
        config: RerankerConfig,
    ):
        self.model = model
        self.config = config

    def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        *,
        top_k: int | None = None
    ) -> list[RetrievalHit]:
        top_k = top_k or self.config.top_k
        if top_k <= 0 or not hits:
            return []

        texts = [hit.chunk.content for hit in hits]
        scores = self.model.score_pairs(query, texts)

        ranked = sorted(
            zip(hits, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        result = []

        for rank, (hit, score) in enumerate(
            ranked[:top_k],
            start=1,
        ):
            hit.rerank_score = float(score)
            hit.rank = rank
            result.append(hit)

        return result