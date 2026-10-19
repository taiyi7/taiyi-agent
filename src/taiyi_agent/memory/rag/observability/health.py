from taiyi_agent.memory.rag.ports.storage import (
    RecordStorePort,
    VectorIndexPort,
)
from taiyi_agent.memory.rag.ports.embedder import EmbedderPort

class HealthChecker:
    def __init__(
            self,
            record_store: RecordStorePort,
            vector_index: VectorIndexPort,
            embedder: EmbedderPort
        ):
        self.record_store = record_store
        self.vector_index = vector_index
        self.embedder = embedder

    def check(self):
        result = {}

        try:
            result["sqlite"] = "ok" if self.record_store.healthcheck() else "error"
        except Exception:
            result["sqlite"] = "error"

        try:
            result["vector_store"] = "ok" if self.vector_index.healthcheck() else "error"
        except Exception:
            result["vector_store"] = "error"

        try:
            result["embedder"] = "ok" if self.embedder.healthcheck() else "error"
        except Exception:
            result["embedder"] = "error"

        result["status"] = (
            "ok"
            if all(value == "ok" for key, value in result.items() if key != "status")
            else "degraded"
        )

        return result