from taiyi_agent.memory.storage.qdrant_store import QdrantStore
from taiyi_agent.memory.storage.sqlite_store import SQLiteStore
from taiyi_agent.memory.rag.config import RAGConfig
from dotenv import load_dotenv

load_dotenv()

rag_config = RAGConfig.from_env()

delete_store = QdrantStore(
    url=rag_config.qdrant.url,
    collection_name="test_rag_ingestion",
    vector_size=rag_config.qdrant.vector_size,
    api_key=rag_config.qdrant.api_key
)
delete_store.delete_collection("test_rag_ingestion")