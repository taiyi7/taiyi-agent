from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from taiyi_agent.memory.storage.models import VectorHit, VectorRecord


class QdrantStore:
    """通用 Qdrant 向量存储。

    一个 collection 使用固定向量维度；
    data_type 和 namespace 通过 payload 隔离。
    """

    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
        *,
        api_key: str | None = None,
        distance: Distance = Distance.COSINE,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)

        self._ensure_collection(
            vector_size=vector_size,
            distance=distance,
        )

    def _ensure_collection(
        self,
        *,
        vector_size: int,
        distance: Distance,
    ) -> None:
        collections = self.client.get_collections().collections
        names = {item.name for item in collections}

        if self.collection_name not in names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance,
                ),
            )

    def upsert(self, record: VectorRecord) -> None:
        self.upsert_many([record])

    def upsert_many(self, records: Iterable[VectorRecord]) -> None:
        points = [
            PointStruct(
                id=record.record_id,
                vector=record.vector,
                payload={
                    **record.payload,
                    "record_id": record.record_id,
                    "data_type": record.data_type,
                    "namespace": record.namespace,
                },
            )
            for record in records
        ]

        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        vector: list[float],
        *,
        data_type: str,
        namespace: str = "default",
        limit: int = 10,
        extra_filter: dict[str, str] | None = None,
    ) -> list[VectorHit]:
        conditions = [
            FieldCondition(
                key="data_type",
                match=MatchValue(value=data_type),
            ),
            FieldCondition(
                key="namespace",
                match=MatchValue(value=namespace),
            ),
        ]

        for key, value in (extra_filter or {}).items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=Filter(must=conditions),
            limit=limit,
            with_payload=True,
        )

        return [
            VectorHit(
                record_id=str(point.id),
                score=float(point.score),
                data_type=point.payload.get("data_type", data_type),
                namespace=point.payload.get("namespace", namespace),
                payload=point.payload,
            )
            for point in result.points
        ]

    def delete(
        self,
        record_id: str,
        *,
        data_type: str,
        namespace: str = "default",
    ) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="record_id",
                        match=MatchValue(value=record_id),
                    ),
                    FieldCondition(
                        key="data_type",
                        match=MatchValue(value=data_type),
                    ),
                    FieldCondition(
                        key="namespace",
                        match=MatchValue(value=namespace),
                    ),
                ]
            ),
            wait=True,
        )

    def delete_filter(
        self,
        *,
        data_type: str,
        namespace: str,
        extra_filter: dict[str, str] | None = None,
    ) -> None:
        conditions = [
            FieldCondition(
                key="data_type",
                match=MatchValue(value=data_type),
            ),
            FieldCondition(
                key="namespace",
                match=MatchValue(value=namespace),
            ),
        ]

        for key, value in (extra_filter or {}).items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=conditions),
            wait=True,
        )

    def delete_collection(self, collection_name: str) -> None:
        """
        删除集合，所有数据永久销毁。
        生产环境慎用，仅限测试/重置场景!!
        """
        self.client.delete_collection(collection_name=collection_name)

    def healthcheck(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False