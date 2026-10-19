import json
import re
from collections.abc import Iterable
from typing import Any

from taiyi_agent.memory.rag.models import SourceDocument, TextChunk
from taiyi_agent.memory.rag.ports.storage import (
    RAG_CHUNK_TYPE,
    RAG_DOCUMENT_TYPE,
)
from taiyi_agent.memory.storage.models import StoredRecord
from taiyi_agent.memory.storage.sqlite_store import SQLiteStore


class SQLiteRAGStore:
    """使用 SQLiteStore 实现 RecordStorePort"""

    def __init__(self, store: SQLiteStore):
        self.store = store
        self.connection = store.connection
        self._create_rag_fts()

    # FTS = Full Text Search，全文检索
    def _create_rag_fts(self) -> None:
        self.connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts
            USING fts5(
                chunk_id UNINDEXED,
                namespace UNINDEXED,
                content
            )
            """
        )
        self.connection.commit()

    def save_document(
        self,
        document: SourceDocument,
        *,
        namespace: str,
    ) -> None:
        self.store.put(
            StoredRecord(
                record_id=document.doc_id,
                data_type=RAG_DOCUMENT_TYPE,
                namespace=namespace,
                content=document.content,
                metadata=document.metadata,
            )
        )

    def save_chunks(
        self,
        chunks: Iterable[TextChunk],
        *,
        namespace: str,
    ) -> None:
        chunks = list(chunks)
        if not chunks:
            return

        records = [
            StoredRecord(
                record_id=chunk.chunk_id,
                data_type=RAG_CHUNK_TYPE,
                namespace=namespace,
                content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                },
            )
            for chunk in chunks
        ]

        # 完整 Chunk 保存到通用 records 表
        self.store.put_many(records)

        # 重建当前 Chunk 的 FTS 内容
        self.connection.executemany(
            """
            DELETE FROM rag_chunks_fts
            WHERE chunk_id = ? AND namespace = ?
            """,
            [
                (chunk.chunk_id, namespace)
                for chunk in chunks
            ],
        )

        self.connection.executemany(
            """
            INSERT INTO rag_chunks_fts(
                chunk_id,
                namespace,
                content
            )
            VALUES (?, ?, ?)
            """,
            [
                (chunk.chunk_id, namespace, chunk.content)
                for chunk in chunks
            ],
        )

        self.connection.commit()

    def get_chunks(
        self,
        chunk_ids: Iterable[str],
        *,
        namespace: str,
    ) -> list[TextChunk]:
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return []

        placeholders = ",".join("?" for _ in ids)

        rows = self.connection.execute(
            f"""
            SELECT record_id, content, metadata_json
            FROM records
            WHERE namespace = ?
              AND data_type = ?
              AND record_id IN ({placeholders})
            """,
            [
                namespace,
                RAG_CHUNK_TYPE,
                *ids,
            ],
        ).fetchall()

        records = {
            row["record_id"]: TextChunk(
                chunk_id=row["record_id"],
                doc_id=json.loads(row["metadata_json"]).get("doc_id", ""),
                content=row["content"],
                index=json.loads(row["metadata_json"]).get("chunk_index", 0),
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        }

        # 按检索结果顺序返回，避免影响融合排序
        return [
            records[chunk_id]
            for chunk_id in ids
            if chunk_id in records
        ]

    def sparse_search(
        self,
        query: str,
        *,
        namespace: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:

        # 把问题的标点去掉，提取token，包上双引号，然后用 AND 拼接
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", query)
        sql_query =" AND ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"'
            for token in tokens
        )
    
        rows = self.connection.execute(
            """
            SELECT chunk_id, bm25(rag_chunks_fts) AS score
            FROM rag_chunks_fts
            WHERE rag_chunks_fts MATCH ?
              AND namespace = ?
            ORDER BY score
            LIMIT ?
            """,
            (sql_query, namespace, limit),
        ).fetchall()

        return [
            (row["chunk_id"], float(row["score"]))
            for row in rows
        ]