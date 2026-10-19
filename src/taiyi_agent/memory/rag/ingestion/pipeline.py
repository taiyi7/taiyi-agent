from pathlib import Path

from taiyi_agent.memory.rag.ingestion.chunker import DocumentChunker
from taiyi_agent.memory.rag.ingestion.indexer import Indexer
from taiyi_agent.memory.rag.ingestion.loaders import load_file
from taiyi_agent.memory.rag.config import RAGConfig


class IngestionPipeline:
    def __init__(
        self,
        chunker: DocumentChunker,
        indexer: Indexer,
        config: RAGConfig,
    ) -> None:
        self.chunker = chunker
        self.indexer = indexer
        self.config = config

    def ingest_file(
        self,
        path: str | Path,
        *,
        namespace: str | None = None,
    ) -> None:
        document = load_file(str(path), encoding=self.config.loader.encoding)
        chunks = self.chunker.split_document(document)
        namespace = namespace or self.config.namespace
        self.indexer.index(
            document,
            chunks,
            namespace=namespace,
        )