from taiyi_agent.memory.rag.ingestion.chunker import DocumentChunker
from taiyi_agent.memory.storage.sqlite_store import SQLiteStore
from taiyi_agent.memory.storage.qdrant_store import QdrantStore
from taiyi_agent.memory.rag.infra.embedder import RAGEmbedder
from taiyi_agent.memory.storage.adapters.rag.sqlite_rag_stroe import SQLiteRAGStore
from taiyi_agent.memory.storage.adapters.rag.qdrant_rag_index import QdrantRAGIndex
from taiyi_agent.memory.rag.retrieval.retriever import HybridRetriever
from taiyi_agent.memory.rag.retrieval.query_service import QueryService
from taiyi_agent.memory.rag.infra.reranker import RAGReranker
from taiyi_agent.memory.rag.infra.reranker import AliRerankModel
from taiyi_agent.memory.rag.config import RAGConfig
from dotenv import load_dotenv

load_dotenv()

rag_config = RAGConfig.from_env()

sqlite_store = SQLiteStore(rag_config.sqlite.path)
qdrant_store = QdrantStore(
    url=rag_config.qdrant.url,
    collection_name="test_rag_ingestion",
    vector_size=rag_config.qdrant.vector_size,
    api_key=rag_config.qdrant.api_key
)

record_store = SQLiteRAGStore(sqlite_store)
vector_index = QdrantRAGIndex(qdrant_store)
embedder = RAGEmbedder(rag_config.embedding)

retriever = HybridRetriever(
    record_store=record_store,
    vector_index=vector_index,
    embedder=embedder,
    config=rag_config,
)

rerank_model = AliRerankModel(rag_config.reranker)
reranker = RAGReranker(rerank_model, rag_config.reranker)

query_dervice = QueryService(
    retriever=retriever,
    reranker=reranker,
    config=rag_config
)
ret = query_dervice.query("陆沉有灵根吗？")
print(ret)