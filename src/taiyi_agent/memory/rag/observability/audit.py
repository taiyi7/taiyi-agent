class AuditLogger:
    def log_query(self, query_id, query, hits, latency_ms):
        # 生产环境写入 SQLite/PostgreSQL 或日志系统
        record = {
            "query_id": query_id,
            "query": query,
            "latency_ms": latency_ms,
            "hit_count": len(hits),
            "citations": [
                {
                    "doc_id": hit.chunk.doc_id,
                    "chunk_id": hit.chunk.chunk_id,
                    "score": hit.score,
                    "rerank_score": hit.rerank_score,
                }
                for hit in hits
            ],
        }
        return record