from taiyi_agent.memory.rag.ingestion.pipeline import IngestionPipeline
from taiyi_agent.memory.rag.ingestion.chunker import DocumentChunker
from taiyi_agent.memory.rag.ingestion.indexer import Indexer
from taiyi_agent.memory.storage.sqlite_store import SQLiteStore
from taiyi_agent.memory.storage.qdrant_store import QdrantStore
from taiyi_agent.memory.rag.infra.embedder import RAGEmbedder
from taiyi_agent.memory.storage.adapters.rag.sqlite_rag_stroe import SQLiteRAGStore
from taiyi_agent.memory.storage.adapters.rag.qdrant_rag_index import QdrantRAGIndex
from taiyi_agent.memory.rag.config import RAGConfig

from dotenv import load_dotenv
load_dotenv()

rag_config = RAGConfig.from_env()
sqlite_store = SQLiteStore(rag_config.sqlite.path)
delete_store = QdrantStore(
    url=rag_config.qdrant.url,
    collection_name="test_rag_ingestion",
    vector_size=rag_config.qdrant.vector_size,
    api_key=rag_config.qdrant.api_key
)
delete_store.delete_collection("test_rag_ingestion")

qdrant_store = QdrantStore(
    url=rag_config.qdrant.url,
    collection_name="test_rag_ingestion",
    vector_size=rag_config.qdrant.vector_size,
    api_key=rag_config.qdrant.api_key
)

record_store = SQLiteRAGStore(sqlite_store)
vector_index = QdrantRAGIndex(qdrant_store)
embedder = RAGEmbedder(rag_config.embedding)

chunker = DocumentChunker(rag_config.chunking)
indexer = Indexer(
    record_store=record_store,
    vector_index=vector_index,
    embedder=embedder,
    config=rag_config,
)

pipeline = IngestionPipeline(
    chunker=chunker,
    indexer=indexer,
    config=rag_config,
)

path = r"C:\code\agent\taiyi-agent\test\rag_test\data\documents\凡根.md"
pipeline.ingest_file(path)
