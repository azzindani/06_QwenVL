"""Storage integrations for document processing results."""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save(
        self,
        key: str,
        data: Union[bytes, str, Dict[str, Any]],
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Save data to storage."""
        pass

    @abstractmethod
    def load(self, key: str) -> Optional[bytes]:
        """Load data from storage."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data from storage."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List keys with prefix."""
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: str = "./storage"):
        """
        Initialize local storage.

        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        key: str,
        data: Union[bytes, str, Dict[str, Any]],
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Save data to local filesystem."""
        file_path = self.base_path / key

        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert data to bytes
        if isinstance(data, dict):
            content = json.dumps(data, indent=2).encode()
        elif isinstance(data, str):
            content = data.encode()
        else:
            content = data

        file_path.write_bytes(content)

        # Save metadata
        if metadata:
            meta_path = file_path.with_suffix(file_path.suffix + ".meta")
            meta_path.write_text(json.dumps(metadata))

        return str(file_path)

    def load(self, key: str) -> Optional[bytes]:
        """Load data from local filesystem."""
        file_path = self.base_path / key
        if file_path.exists():
            return file_path.read_bytes()
        return None

    def delete(self, key: str) -> bool:
        """Delete file from local filesystem."""
        file_path = self.base_path / key
        if file_path.exists():
            file_path.unlink()
            # Remove metadata if exists
            meta_path = file_path.with_suffix(file_path.suffix + ".meta")
            if meta_path.exists():
                meta_path.unlink()
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if file exists."""
        return (self.base_path / key).exists()

    def list_keys(self, prefix: str = "") -> List[str]:
        """List files with prefix."""
        search_path = self.base_path / prefix if prefix else self.base_path
        if not search_path.exists():
            return []

        keys = []
        for path in search_path.rglob("*"):
            if path.is_file() and not path.suffix == ".meta":
                keys.append(str(path.relative_to(self.base_path)))

        return keys


class S3Storage(StorageBackend):
    """AWS S3 storage backend."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        """
        Initialize S3 storage.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix
            region: AWS region
            access_key: AWS access key (or use env)
            secret_key: AWS secret key (or use env)
        """
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")

        # Lazy import boto3
        try:
            import boto3
            self._client = boto3.client(
                "s3",
                region_name=self.region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")

    def _full_key(self, key: str) -> str:
        """Get full key with prefix."""
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def save(
        self,
        key: str,
        data: Union[bytes, str, Dict[str, Any]],
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Save data to S3."""
        full_key = self._full_key(key)

        # Convert data to bytes
        if isinstance(data, dict):
            content = json.dumps(data, indent=2).encode()
        elif isinstance(data, str):
            content = data.encode()
        else:
            content = data

        extra_args = {}
        if metadata:
            extra_args["Metadata"] = metadata

        self._client.put_object(
            Bucket=self.bucket,
            Key=full_key,
            Body=content,
            **extra_args,
        )

        return f"s3://{self.bucket}/{full_key}"

    def load(self, key: str) -> Optional[bytes]:
        """Load data from S3."""
        try:
            response = self._client.get_object(
                Bucket=self.bucket,
                Key=self._full_key(key),
            )
            return response["Body"].read()
        except self._client.exceptions.NoSuchKey:
            return None

    def delete(self, key: str) -> bool:
        """Delete object from S3."""
        try:
            self._client.delete_object(
                Bucket=self.bucket,
                Key=self._full_key(key),
            )
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        """Check if object exists in S3."""
        try:
            self._client.head_object(
                Bucket=self.bucket,
                Key=self._full_key(key),
            )
            return True
        except Exception:
            return False

    def list_keys(self, prefix: str = "") -> List[str]:
        """List objects with prefix in S3."""
        full_prefix = self._full_key(prefix)

        keys = []
        paginator = self._client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self.prefix:
                    key = key[len(self.prefix) + 1:]
                keys.append(key)

        return keys


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        credentials_path: Optional[str] = None,
    ):
        """
        Initialize GCS storage.

        Args:
            bucket: GCS bucket name
            prefix: Key prefix
            credentials_path: Path to service account JSON
        """
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")

        # Lazy import google-cloud-storage
        try:
            from google.cloud import storage

            if credentials_path:
                self._client = storage.Client.from_service_account_json(credentials_path)
            else:
                self._client = storage.Client()

            self._bucket = self._client.bucket(bucket)
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required. Install with: pip install google-cloud-storage"
            )

    def _full_key(self, key: str) -> str:
        """Get full key with prefix."""
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def save(
        self,
        key: str,
        data: Union[bytes, str, Dict[str, Any]],
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Save data to GCS."""
        full_key = self._full_key(key)
        blob = self._bucket.blob(full_key)

        # Convert data
        if isinstance(data, dict):
            content = json.dumps(data, indent=2)
            blob.upload_from_string(content, content_type="application/json")
        elif isinstance(data, str):
            blob.upload_from_string(data)
        else:
            blob.upload_from_string(data)

        if metadata:
            blob.metadata = metadata
            blob.patch()

        return f"gs://{self.bucket_name}/{full_key}"

    def load(self, key: str) -> Optional[bytes]:
        """Load data from GCS."""
        blob = self._bucket.blob(self._full_key(key))
        if blob.exists():
            return blob.download_as_bytes()
        return None

    def delete(self, key: str) -> bool:
        """Delete blob from GCS."""
        blob = self._bucket.blob(self._full_key(key))
        if blob.exists():
            blob.delete()
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if blob exists in GCS."""
        return self._bucket.blob(self._full_key(key)).exists()

    def list_keys(self, prefix: str = "") -> List[str]:
        """List blobs with prefix in GCS."""
        full_prefix = self._full_key(prefix)

        keys = []
        for blob in self._bucket.list_blobs(prefix=full_prefix):
            key = blob.name
            if self.prefix:
                key = key[len(self.prefix) + 1:]
            keys.append(key)

        return keys


def create_storage(
    backend: str = "local",
    **kwargs,
) -> StorageBackend:
    """
    Create a storage backend.

    Args:
        backend: Backend type (local, s3, gcs)
        **kwargs: Backend-specific configuration

    Returns:
        Storage backend instance
    """
    backends = {
        "local": LocalStorage,
        "s3": S3Storage,
        "gcs": GCSStorage,
    }

    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(backends.keys())}")

    return backends[backend](**kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("STORAGE BACKEND TEST")
    print("=" * 60)

    # Test local storage
    storage = LocalStorage("/tmp/qwen_vl_test")

    # Save data
    path = storage.save(
        "results/test.json",
        {"text": "Hello", "confidence": 0.95},
        metadata={"task": "ocr"},
    )
    print(f"  Saved to: {path}")

    # Load data
    data = storage.load("results/test.json")
    print(f"  Loaded: {json.loads(data)}")

    # List keys
    keys = storage.list_keys("results")
    print(f"  Keys: {keys}")

    # Delete
    storage.delete("results/test.json")
    print(f"  Exists after delete: {storage.exists('results/test.json')}")

    print("=" * 60)
