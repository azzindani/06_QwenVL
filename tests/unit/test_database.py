"""Unit tests for database backends."""

import pytest

from qwen_vl.api.database import create_database


@pytest.mark.unit
class TestCreateDatabase:
    """Tests for database factory function."""

    def test_create_unknown_backend(self):
        """Test creating unknown backend raises error."""
        with pytest.raises(ValueError) as exc_info:
            create_database("unknown")

        assert "Unknown backend" in str(exc_info.value)

    def test_create_postgresql_backend(self):
        """Test creating PostgreSQL backend."""
        from qwen_vl.api.database import PostgreSQLBackend

        backend = create_database(
            "postgresql",
            host="localhost",
            database="test",
        )

        assert isinstance(backend, PostgreSQLBackend)
        assert backend.host == "localhost"

    def test_create_mongodb_backend(self):
        """Test creating MongoDB backend."""
        from qwen_vl.api.database import MongoDBBackend

        backend = create_database(
            "mongodb",
            uri="mongodb://localhost:27017",
            database="test",
        )

        assert isinstance(backend, MongoDBBackend)
        assert backend.database_name == "test"
