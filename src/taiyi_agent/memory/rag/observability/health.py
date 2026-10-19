class HealthChecker:
    def __init__(self, db, vector_store, embedder):
        self.db = db
        self.vector_store = vector_store
        self.embedder = embedder

    def check(self):
        result = {}

        try:
            self.db.execute("SELECT 1")
            result["sqlite"] = "ok"
        except Exception:
            result["sqlite"] = "error"

        try:
            result["vector_store"] = (
                "ok" if self.vector_store.healthcheck() else "error"
            )
        except Exception:
            result["vector_store"] = "error"

        try:
            result["embedder"] = (
                "ok" if self.embedder.healthcheck() else "error"
            )
        except Exception:
            result["embedder"] = "error"

        result["status"] = (
            "ok"
            if all(value == "ok" for key, value in result.items() if key != "status")
            else "degraded"
        )

        return result