from taiyi_agent.memory.rag.ports.reranker import RerankerPort
from taiyi_agent.memory.rag.retrieval.retriever import HybridRetriever
from taiyi_agent.memory.rag.config import RAGConfig

class QueryService:
    def __init__(
        self,
        retriever: HybridRetriever, 
        reranker: RerankerPort,
        config: RAGConfig
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.config = config

    def query(
        self,
        query_text: str,
        *,
        namespace: str | None = None,
        candidate_k: int | None = None,
        answer_k: int | None = None,
        filters: dict | None = None,
    ):
        # 参数校验
        query_text = query_text.strip()
        if not query_text:
            return {"query": query_text, "hits": [], "contexts": [], "citations": []}

        namespace = namespace or self.config.namespace
        candidate_k = (
            candidate_k
            if candidate_k is not None
            else self.config.retrieval.candidate_k
        )
        answer_k = (
            answer_k
            if answer_k is not None
            else self.config.retrieval.answer_k
        )
        candidate_k = max(1, min(candidate_k, self.config.retrieval.max_candidate_k))
        answer_k = max(1, min(answer_k, candidate_k))

        candidates = self.retriever.search(
            query_text,
            namespace=namespace,
            limit=candidate_k,
            filters=filters,
        )

        hits = self.reranker.rerank(
            query_text,
            candidates,
            top_k=answer_k,
        )

        # context 建议使用 TokenCounter.truncate_text() 控制最终上下文总 token 数。
        return {
            "query": query_text,
            "hits": hits,
            "contexts": [
                hit.chunk.content
                for hit in hits
            ],
            "citations": [
                {
                    "doc_id": hit.chunk.doc_id,
                    "chunk_id": hit.chunk.chunk_id,
                    "source_uri": hit.chunk.metadata.get(
                        "source_uri"
                    ),
                    "page_number": hit.chunk.metadata.get(
                        "page_number"
                    ),
                    "score": hit.score,
                    "rerank_score": hit.rerank_score,
                }
                for hit in hits
            ],
        }