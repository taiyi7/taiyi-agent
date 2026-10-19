import os
from dotenv import load_dotenv
from uuid import uuid4
import hashlib

from taiyi_agent.memory.storage.models import StoredRecord, VectorRecord
from taiyi_agent.memory.storage.sqlite_store import SQLiteStore
from taiyi_agent.memory.storage.qdrant_store import QdrantStore
from taiyi_agent.memory.rag.models import TextChunk, chunk_hash_to_uuid

load_dotenv()

sqlite_store = SQLiteStore(r"C:\code\agent\taiyi-agent\src\taiyi_agent\memory\data\rag_data\rag_test.db")

# 测试删除 collection

# delete_store = QdrantStore(
#     url=os.getenv("QDRANT_URL"),
#     collection_name="taiyi_vectors",
#     vector_size=1536,
#     api_key=os.getenv("QDRANT_API_KEY")
# )
# delete_store.delete_collection("taiyi_vectors")

qdrant_store = QdrantStore(
    url=os.getenv("QDRANT_URL"),
    collection_name="test_memory_storage",
    vector_size=3,
    api_key=os.getenv("QDRANT_API_KEY")
)

CHUNK_ID = hashlib.sha256(f"测试memory_storage".encode()).hexdigest()
print("CHUNK_ID:", CHUNK_ID)
print("record_id(uuid):", str(chunk_hash_to_uuid(CHUNK_ID)))


chunk = TextChunk(
    chunk_id=CHUNK_ID,
    doc_id="2345",
    content="你好，我现在测试memory_storage的数据库能力",
    index= 0,
    metadata={
        "source_url": "https://xxx"
    }
)

# 1. 保存 RAG Chunk 到 SQLite
sqlite_store.put(
    StoredRecord(
        record_id=str(chunk_hash_to_uuid(chunk.chunk_id)),
        data_type="test_rag_chunk",
        namespace="test_namespace",
        content=chunk.content,
        metadata=chunk.metadata,
    )
)

# 2. 保存 Chunk 向量到 Qdrant
qdrant_store.upsert(
    VectorRecord(
        record_id=str(chunk_hash_to_uuid(chunk.chunk_id)),
        vector=[1.0, 2.0, 3.0],
        data_type="test_rag_chunk",
        namespace="test_namespace",
        payload={
            "doc_id": chunk.doc_id,
            "source_uri": chunk.metadata.get("source_uri"),
            "page_number": chunk.metadata.get("page_number"),
        },
    )
)

# 3. 在线查询 Qdrant
hits = qdrant_store.search(
    [2.0, 3.0, 4.0],
    data_type="test_rag_chunk",
    namespace="test_namespace",
    limit=20,
)

# 4. 根据向量命中结果回 SQLite 读取完整 Chunk
chunks = [
    sqlite_store.get(
        hit.record_id,
        data_type="test_rag_chunk",
        namespace="test_namespace",
    )
    for hit in hits
]

print(hits)
print()
print(chunks)