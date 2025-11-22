"""Database integrations for storing extraction results."""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional


class DatabaseBackend(ABC):
    """Abstract base class for database backends."""

    @abstractmethod
    def connect(self) -> None:
        """Establish database connection."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass

    @abstractmethod
    def save_result(
        self,
        document_id: str,
        task_type: str,
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save extraction result."""
        pass

    @abstractmethod
    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get result by ID."""
        pass

    @abstractmethod
    def query_results(
        self,
        task_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query results with filters."""
        pass


class PostgreSQLBackend(DatabaseBackend):
    """PostgreSQL database backend."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "qwen_vl",
        user: str = "postgres",
        password: str = "",
    ):
        """
        Initialize PostgreSQL backend.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._conn = None

    def connect(self) -> None:
        """Connect to PostgreSQL."""
        try:
            import psycopg2
            self._conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            self._ensure_tables()
        except ImportError:
            raise ImportError(
                "psycopg2 is required. Install with: pip install psycopg2-binary"
            )

    def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS extraction_results (
                    id SERIAL PRIMARY KEY,
                    result_id VARCHAR(36) UNIQUE NOT NULL,
                    document_id VARCHAR(255) NOT NULL,
                    task_type VARCHAR(50) NOT NULL,
                    result JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_task_type ON extraction_results(task_type);
                CREATE INDEX IF NOT EXISTS idx_created_at ON extraction_results(created_at);
            """)
            self._conn.commit()

    def save_result(
        self,
        document_id: str,
        task_type: str,
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save extraction result to PostgreSQL."""
        import uuid
        result_id = str(uuid.uuid4())

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO extraction_results
                (result_id, document_id, task_type, result, metadata)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    result_id,
                    document_id,
                    task_type,
                    json.dumps(result),
                    json.dumps(metadata) if metadata else None,
                ),
            )
            self._conn.commit()

        return result_id

    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get result by ID from PostgreSQL."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM extraction_results WHERE result_id = %s",
                (result_id,),
            )
            row = cur.fetchone()

            if row:
                return {
                    "result_id": row[1],
                    "document_id": row[2],
                    "task_type": row[3],
                    "result": row[4],
                    "metadata": row[5],
                    "created_at": row[6].isoformat() if row[6] else None,
                }

        return None

    def query_results(
        self,
        task_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query results from PostgreSQL."""
        query = "SELECT * FROM extraction_results WHERE 1=1"
        params = []

        if task_type:
            query += " AND task_type = %s"
            params.append(task_type)

        if start_date:
            query += " AND created_at >= %s"
            params.append(start_date)

        if end_date:
            query += " AND created_at <= %s"
            params.append(end_date)

        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

            return [
                {
                    "result_id": row[1],
                    "document_id": row[2],
                    "task_type": row[3],
                    "result": row[4],
                    "metadata": row[5],
                    "created_at": row[6].isoformat() if row[6] else None,
                }
                for row in rows
            ]


class MongoDBBackend(DatabaseBackend):
    """MongoDB database backend."""

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        database: str = "qwen_vl",
        collection: str = "extraction_results",
    ):
        """
        Initialize MongoDB backend.

        Args:
            uri: MongoDB connection URI
            database: Database name
            collection: Collection name
        """
        self.uri = uri
        self.database_name = database
        self.collection_name = collection
        self._client = None
        self._db = None
        self._collection = None

    def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            from pymongo import MongoClient
            self._client = MongoClient(self.uri)
            self._db = self._client[self.database_name]
            self._collection = self._db[self.collection_name]

            # Create indexes
            self._collection.create_index("result_id", unique=True)
            self._collection.create_index("task_type")
            self._collection.create_index("created_at")
        except ImportError:
            raise ImportError(
                "pymongo is required. Install with: pip install pymongo"
            )

    def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self._client:
            self._client.close()
            self._client = None

    def save_result(
        self,
        document_id: str,
        task_type: str,
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save extraction result to MongoDB."""
        import uuid
        result_id = str(uuid.uuid4())

        doc = {
            "result_id": result_id,
            "document_id": document_id,
            "task_type": task_type,
            "result": result,
            "metadata": metadata,
            "created_at": datetime.utcnow(),
        }

        self._collection.insert_one(doc)
        return result_id

    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get result by ID from MongoDB."""
        doc = self._collection.find_one({"result_id": result_id})

        if doc:
            return {
                "result_id": doc["result_id"],
                "document_id": doc["document_id"],
                "task_type": doc["task_type"],
                "result": doc["result"],
                "metadata": doc.get("metadata"),
                "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None,
            }

        return None

    def query_results(
        self,
        task_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query results from MongoDB."""
        query = {}

        if task_type:
            query["task_type"] = task_type

        if start_date or end_date:
            query["created_at"] = {}
            if start_date:
                query["created_at"]["$gte"] = start_date
            if end_date:
                query["created_at"]["$lte"] = end_date

        cursor = self._collection.find(query).sort("created_at", -1).limit(limit)

        return [
            {
                "result_id": doc["result_id"],
                "document_id": doc["document_id"],
                "task_type": doc["task_type"],
                "result": doc["result"],
                "metadata": doc.get("metadata"),
                "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None,
            }
            for doc in cursor
        ]


def create_database(
    backend: str = "postgresql",
    **kwargs,
) -> DatabaseBackend:
    """
    Create a database backend.

    Args:
        backend: Backend type (postgresql, mongodb)
        **kwargs: Backend-specific configuration

    Returns:
        Database backend instance
    """
    backends = {
        "postgresql": PostgreSQLBackend,
        "postgres": PostgreSQLBackend,
        "mongodb": MongoDBBackend,
        "mongo": MongoDBBackend,
    }

    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(backends.keys())}")

    return backends[backend](**kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("DATABASE BACKENDS")
    print("=" * 60)
    print("  Available: postgresql, mongodb")
    print("=" * 60)
