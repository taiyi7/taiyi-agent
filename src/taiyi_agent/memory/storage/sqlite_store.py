import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from taiyi_agent.memory.storage.models import StoredRecord


class SQLiteStore:
    """通用 SQLite 存储。

    使用 namespace + data_type + record_id 隔离不同业务数据。
    """

    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(
            str(path),
            timeout=30,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self._configure()
        self._create_schema()

    def _configure(self) -> None:
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=30000")

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                namespace TEXT NOT NULL,
                data_type TEXT NOT NULL,
                record_id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (namespace, data_type, record_id)
            );

            CREATE INDEX IF NOT EXISTS idx_records_type
            ON records(namespace, data_type);

            CREATE INDEX IF NOT EXISTS idx_records_updated_at
            ON records(namespace, data_type, updated_at);
            """
        )
        self.connection.commit()

    def put(self, record: StoredRecord) -> None:
        """新增或覆盖一条记录。"""
        self.connection.execute(
            """
            INSERT INTO records (
                namespace,
                data_type,
                record_id,
                content,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, data_type, record_id)
            DO UPDATE SET
                content = excluded.content,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                record.namespace,
                record.data_type,
                record.record_id,
                record.content,
                json.dumps(record.metadata, ensure_ascii=False),
                record.created_at,
                record.updated_at,
            ),
        )
        self.connection.commit()

    def put_many(self, records: Iterable[StoredRecord]) -> None:
        """批量写入，单事务提交。"""
        rows = [
            (
                record.namespace,
                record.data_type,
                record.record_id,
                record.content,
                json.dumps(record.metadata, ensure_ascii=False),
                record.created_at,
                record.updated_at,
            )
            for record in records
        ]

        if not rows:
            return

        self.connection.executemany(
            """
            INSERT INTO records (
                namespace,
                data_type,
                record_id,
                content,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, data_type, record_id)
            DO UPDATE SET
                content = excluded.content,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            rows,
        )
        self.connection.commit()

    def get(
        self,
        record_id: str,
        *,
        data_type: str,
        namespace: str = "default",
    ) -> StoredRecord | None:
        row = self.connection.execute(
            """
            SELECT *
            FROM records
            WHERE namespace = ?
              AND data_type = ?
              AND record_id = ?
            """,
            (namespace, data_type, record_id),
        ).fetchone()

        return self._to_record(row) if row else None

    def list(
        self,
        *,
        data_type: str,
        namespace: str = "default",
        limit: int = 100,
    ) -> list[StoredRecord]:
        rows = self.connection.execute(
            """
            SELECT *
            FROM records
            WHERE namespace = ?
              AND data_type = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (namespace, data_type, limit),
        ).fetchall()

        return [self._to_record(row) for row in rows]

    def delete(
        self,
        record_id: str,
        *,
        data_type: str,
        namespace: str = "default",
    ) -> bool:
        cursor = self.connection.execute(
            """
            DELETE FROM records
            WHERE namespace = ?
              AND data_type = ?
              AND record_id = ?
            """,
            (namespace, data_type, record_id),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def delete_type(
        self,
        *,
        data_type: str,
        namespace: str = "default",
    ) -> int:
        cursor = self.connection.execute(
            """
            DELETE FROM records
            WHERE namespace = ?
              AND data_type = ?
            """,
            (namespace, data_type),
        )
        self.connection.commit()
        return cursor.rowcount

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> StoredRecord:
        return StoredRecord(
            record_id=row["record_id"],
            data_type=row["data_type"],
            namespace=row["namespace"],
            content=row["content"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )