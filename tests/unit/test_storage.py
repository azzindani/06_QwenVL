"""Unit tests for storage backends."""

import json
import os
import tempfile
import pytest

from qwen_vl.api.storage import (
    LocalStorage,
    create_storage,
)


@pytest.mark.unit
class TestLocalStorage:
    """Tests for LocalStorage backend."""

    def test_save_string(self):
        """Test saving string data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(tmpdir)

            path = storage.save("test.txt", "Hello, World!")

            assert os.path.exists(path)
            assert storage.exists("test.txt")

    def test_save_dict(self):
        """Test saving dictionary as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(tmpdir)

            data = {"key": "value", "number": 42}
            storage.save("data.json", data)

            loaded = storage.load("data.json")
            assert json.loads(loaded) == data

    def test_save_bytes(self):
        """Test saving binary data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(tmpdir)

            data = b"\x00\x01\x02\x03"
            storage.save("binary.dat", data)

            loaded = storage.load("binary.dat")
            assert loaded == data

    def test_save_with_metadata(self):
        """Test saving with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(tmpdir)

            storage.save(
                "test.txt",
                "content",
                metadata={"author": "test"},
            )

            # Check metadata file exists
            meta_path = os.path.join(tmpdir, "test.txt.meta")
            assert os.path.exists(meta_path)

    def test_save_nested_path(self):
        """Test saving to nested directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(tmpdir)

            storage.save("a/b/c/test.txt", "nested")

            assert storage.exists("a/b/c/test.txt")

    def test_load_nonexistent(self):
        """Test loading non-existent file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(tmpdir)
            assert storage.load("nonexistent.txt") is None

    def test_delete(self):
        """Test deleting a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(tmpdir)

            storage.save("test.txt", "content")
            assert storage.exists("test.txt")

            storage.delete("test.txt")
            assert not storage.exists("test.txt")

    def test_delete_nonexistent(self):
        """Test deleting non-existent file returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(tmpdir)
            assert storage.delete("nonexistent.txt") is False

    def test_list_keys(self):
        """Test listing keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(tmpdir)

            storage.save("file1.txt", "content1")
            storage.save("file2.txt", "content2")
            storage.save("subdir/file3.txt", "content3")

            keys = storage.list_keys()
            assert len(keys) == 3
            assert "file1.txt" in keys
            assert "subdir/file3.txt" in keys

    def test_list_keys_with_prefix(self):
        """Test listing keys with prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(tmpdir)

            storage.save("results/file1.txt", "content1")
            storage.save("results/file2.txt", "content2")
            storage.save("other/file3.txt", "content3")

            keys = storage.list_keys("results")
            assert len(keys) == 2

    def test_list_keys_empty_dir(self):
        """Test listing keys in empty/nonexistent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(tmpdir)
            keys = storage.list_keys("nonexistent")
            assert keys == []


@pytest.mark.unit
class TestCreateStorage:
    """Tests for storage factory function."""

    def test_create_local_storage(self):
        """Test creating local storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = create_storage("local", base_path=tmpdir)
            assert isinstance(storage, LocalStorage)

    def test_create_unknown_backend(self):
        """Test creating unknown backend raises error."""
        with pytest.raises(ValueError) as exc_info:
            create_storage("unknown")

        assert "Unknown backend" in str(exc_info.value)
