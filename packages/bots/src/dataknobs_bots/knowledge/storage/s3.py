"""S3-based knowledge resource backend for production deployments.

This backend stores files in Amazon S3 (or S3-compatible storage like MinIO,
LocalStack), making it ideal for production deployments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import BinaryIO

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .models import IngestionStatus, KnowledgeBaseInfo, KnowledgeFile

logger = logging.getLogger(__name__)


class S3KnowledgeBackend:
    """S3-backed knowledge resource storage.

    Structure in S3:
        s3://{bucket}/{prefix}/
            {domain_id}/
                _metadata.json
                content/
                    file1.md
                    subdir/file2.json

    Suitable for:
    - Production deployments
    - Multi-instance deployments
    - Large knowledge bases

    Example:
        ```python
        backend = S3KnowledgeBackend(
            bucket="my-bucket",
            prefix="knowledge/",
            region="us-east-1"
        )
        await backend.initialize()

        await backend.create_kb("my-domain")
        await backend.put_file("my-domain", "intro.md", b"# Hello")

        content = await backend.get_file("my-domain", "intro.md")
        print(content.decode())

        await backend.close()
        ```
    """

    METADATA_FILE = "_metadata.json"
    CONTENT_DIR = "content"

    def __init__(
        self,
        bucket: str,
        prefix: str = "knowledge/",
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        """Initialize the S3 backend.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all objects (default: "knowledge/")
            region: AWS region (default: "us-east-1")
            endpoint_url: Custom endpoint URL (for S3-compatible services)
            aws_access_key_id: AWS access key (optional, uses default chain)
            aws_secret_access_key: AWS secret key (optional, uses default chain)
        """
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._region = region
        self._endpoint_url = endpoint_url
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._client: boto3.client | None = None
        self._initialized = False

    @classmethod
    def from_config(cls, config: dict) -> S3KnowledgeBackend:
        """Create from configuration dict.

        Args:
            config: Configuration with bucket, prefix, region, endpoint_url keys

        Returns:
            New S3KnowledgeBackend instance
        """
        return cls(
            bucket=config["bucket"],
            prefix=config.get("prefix", "knowledge/"),
            region=config.get("region", "us-east-1"),
            endpoint_url=config.get("endpoint_url"),
            aws_access_key_id=config.get("aws_access_key_id"),
            aws_secret_access_key=config.get("aws_secret_access_key"),
        )

    async def initialize(self) -> None:
        """Initialize the S3 client and verify bucket access."""
        boto_config = Config(
            region_name=self._region,
            retries={"max_attempts": 3, "mode": "standard"},
        )

        kwargs: dict = {"config": boto_config}
        if self._endpoint_url:
            kwargs["endpoint_url"] = self._endpoint_url
        if self._aws_access_key_id:
            kwargs["aws_access_key_id"] = self._aws_access_key_id
        if self._aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self._aws_secret_access_key

        self._client = boto3.client("s3", **kwargs)

        # Verify bucket exists
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                raise ValueError(f"Bucket '{self._bucket}' does not exist") from e
            raise

        self._initialized = True
        logger.info("Initialized S3 backend: s3://%s/%s", self._bucket, self._prefix)

    async def close(self) -> None:
        """Close the S3 client."""
        if self._client:
            self._client.close()
        self._initialized = False

    def _s3_key(self, domain_id: str, path: str = "") -> str:
        """Get the S3 key for a path within a domain."""
        if path:
            return f"{self._prefix}{domain_id}/{self.CONTENT_DIR}/{path}"
        return f"{self._prefix}{domain_id}/"

    def _metadata_key(self, domain_id: str) -> str:
        """Get the S3 key for a KB's metadata file."""
        return f"{self._prefix}{domain_id}/{self.METADATA_FILE}"

    def _load_metadata(self, domain_id: str) -> dict:
        """Load metadata from S3."""
        if not self._client:
            raise RuntimeError("Backend not initialized")

        key = self._metadata_key(domain_id)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                return {}
            raise

    def _save_metadata(self, domain_id: str, metadata: dict) -> None:
        """Save metadata to S3."""
        if not self._client:
            raise RuntimeError("Backend not initialized")

        key = self._metadata_key(domain_id)
        body = json.dumps(metadata, indent=2, default=str).encode("utf-8")
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )

    def _kb_exists(self, domain_id: str) -> bool:
        """Check if a knowledge base exists."""
        if not self._client:
            raise RuntimeError("Backend not initialized")

        key = self._metadata_key(domain_id)
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return False
            raise

    # --- File Operations ---

    async def put_file(
        self,
        domain_id: str,
        path: str,
        content: bytes | BinaryIO,
        content_type: str | None = None,
        metadata: dict | None = None,
    ) -> KnowledgeFile:
        """Upload or update a file."""
        if not self._client:
            raise RuntimeError("Backend not initialized")

        if not self._kb_exists(domain_id):
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        # Get content as bytes
        if isinstance(content, bytes):
            data = content
        else:
            data = content.read()

        # Auto-detect content type
        if content_type is None:
            guessed_type, _ = mimetypes.guess_type(path)
            content_type = guessed_type or "application/octet-stream"

        # Calculate checksum
        checksum = hashlib.md5(data).hexdigest()

        # Upload file
        key = self._s3_key(domain_id, path)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata={"checksum": checksum} if not metadata else {**metadata, "checksum": checksum},
        )

        # Create file metadata
        file_info = KnowledgeFile(
            path=path,
            content_type=content_type,
            size_bytes=len(data),
            checksum=checksum,
            uploaded_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

        # Update KB metadata
        kb_metadata = self._load_metadata(domain_id)
        files = kb_metadata.get("files", {})
        is_new = path not in files
        files[path] = file_info.to_dict()
        kb_metadata["files"] = files

        # Update KB info
        kb_info = KnowledgeBaseInfo.from_dict(kb_metadata.get("info", {"domain_id": domain_id}))
        kb_info.last_updated = datetime.now(timezone.utc)
        kb_info.increment_version()
        kb_info.file_count = len(files)
        kb_info.total_size_bytes = sum(f.get("size_bytes", 0) for f in files.values())
        kb_metadata["info"] = kb_info.to_dict()

        self._save_metadata(domain_id, kb_metadata)

        logger.debug(
            "%s file: s3://%s/%s (%d bytes)",
            "Created" if is_new else "Updated",
            self._bucket,
            key,
            len(data),
        )

        return file_info

    async def get_file(self, domain_id: str, path: str) -> bytes | None:
        """Get file content."""
        if not self._client:
            raise RuntimeError("Backend not initialized")

        key = self._s3_key(domain_id, path)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                return None
            raise

    async def stream_file(
        self, domain_id: str, path: str, chunk_size: int = 8192
    ) -> AsyncIterator[bytes] | None:
        """Stream file content."""
        if not self._client:
            raise RuntimeError("Backend not initialized")

        key = self._s3_key(domain_id, path)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                return None
            raise

        async def _generator() -> AsyncIterator[bytes]:
            body = response["Body"]
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                yield chunk
            body.close()

        return _generator()

    async def delete_file(self, domain_id: str, path: str) -> bool:
        """Delete a file."""
        if not self._client:
            raise RuntimeError("Backend not initialized")

        key = self._s3_key(domain_id, path)

        # Check if file exists
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return False
            raise

        # Delete file
        self._client.delete_object(Bucket=self._bucket, Key=key)

        # Update metadata
        kb_metadata = self._load_metadata(domain_id)
        files = kb_metadata.get("files", {})
        if path in files:
            del files[path]
        kb_metadata["files"] = files

        # Update KB info
        kb_info = KnowledgeBaseInfo.from_dict(kb_metadata.get("info", {"domain_id": domain_id}))
        kb_info.last_updated = datetime.now(timezone.utc)
        kb_info.increment_version()
        kb_info.file_count = len(files)
        kb_info.total_size_bytes = sum(f.get("size_bytes", 0) for f in files.values())
        kb_metadata["info"] = kb_info.to_dict()

        self._save_metadata(domain_id, kb_metadata)

        logger.debug("Deleted file: s3://%s/%s", self._bucket, key)
        return True

    async def list_files(
        self, domain_id: str, prefix: str | None = None
    ) -> list[KnowledgeFile]:
        """List all files in a knowledge base."""
        kb_metadata = self._load_metadata(domain_id)
        files_dict = kb_metadata.get("files", {})

        files = [KnowledgeFile.from_dict(f) for f in files_dict.values()]

        if prefix:
            files = [f for f in files if f.path.startswith(prefix)]

        # Sort by path for consistent ordering
        files.sort(key=lambda f: f.path)
        return files

    async def file_exists(self, domain_id: str, path: str) -> bool:
        """Check if a file exists."""
        if not self._client:
            raise RuntimeError("Backend not initialized")

        key = self._s3_key(domain_id, path)
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return False
            raise

    # --- Knowledge Base Operations ---

    async def create_kb(
        self, domain_id: str, metadata: dict | None = None
    ) -> KnowledgeBaseInfo:
        """Create a new knowledge base."""
        if not self._client:
            raise RuntimeError("Backend not initialized")

        if self._kb_exists(domain_id):
            raise ValueError(f"Knowledge base '{domain_id}' already exists")

        # Create initial metadata
        kb_info = KnowledgeBaseInfo(
            domain_id=domain_id,
            file_count=0,
            total_size_bytes=0,
            last_updated=datetime.now(timezone.utc),
            version="1",
            ingestion_status=IngestionStatus.PENDING,
            metadata=metadata or {},
        )

        kb_metadata = {
            "info": kb_info.to_dict(),
            "files": {},
        }
        self._save_metadata(domain_id, kb_metadata)

        logger.info("Created knowledge base: s3://%s/%s%s/", self._bucket, self._prefix, domain_id)
        return kb_info

    async def get_info(self, domain_id: str) -> KnowledgeBaseInfo | None:
        """Get knowledge base metadata."""
        if not self._kb_exists(domain_id):
            return None

        kb_metadata = self._load_metadata(domain_id)
        info_dict = kb_metadata.get("info", {"domain_id": domain_id})
        return KnowledgeBaseInfo.from_dict(info_dict)

    async def delete_kb(self, domain_id: str) -> bool:
        """Delete entire knowledge base and all files."""
        if not self._client:
            raise RuntimeError("Backend not initialized")

        if not self._kb_exists(domain_id):
            return False

        # Delete all objects with the domain prefix
        prefix = f"{self._prefix}{domain_id}/"
        paginator = self._client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            contents = page.get("Contents", [])
            if not contents:
                continue

            objects = [{"Key": obj["Key"]} for obj in contents]
            self._client.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": objects},
            )

        logger.info("Deleted knowledge base: s3://%s/%s", self._bucket, prefix)
        return True

    async def list_kbs(self) -> list[KnowledgeBaseInfo]:
        """List all knowledge bases."""
        if not self._client:
            raise RuntimeError("Backend not initialized")

        kbs = []

        # List all "directories" under the prefix
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self._bucket,
            Prefix=self._prefix,
            Delimiter="/",
        ):
            for prefix_obj in page.get("CommonPrefixes", []):
                # Extract domain_id from prefix
                prefix_path = prefix_obj["Prefix"]
                domain_id = prefix_path[len(self._prefix) :].rstrip("/")

                if domain_id:
                    info = await self.get_info(domain_id)
                    if info:
                        kbs.append(info)

        return sorted(kbs, key=lambda kb: kb.domain_id)

    # --- Ingestion Status ---

    async def set_ingestion_status(
        self,
        domain_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        """Update ingestion status for a knowledge base."""
        if not self._kb_exists(domain_id):
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        kb_metadata = self._load_metadata(domain_id)
        info_dict = kb_metadata.get("info", {"domain_id": domain_id})
        info_dict["ingestion_status"] = status
        info_dict["ingestion_error"] = error
        kb_metadata["info"] = info_dict

        self._save_metadata(domain_id, kb_metadata)

    # --- Change Detection ---

    async def get_checksum(self, domain_id: str) -> str:
        """Get combined checksum of all files."""
        if not self._kb_exists(domain_id):
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        kb_metadata = self._load_metadata(domain_id)
        files = kb_metadata.get("files", {})

        if not files:
            return ""

        # Combine all file checksums sorted by path
        checksums = sorted(f"{path}:{info['checksum']}" for path, info in files.items())
        combined = ":".join(checksums)
        return hashlib.md5(combined.encode()).hexdigest()

    async def has_changes_since(self, domain_id: str, version: str) -> bool:
        """Check if KB has changed since given version."""
        info = await self.get_info(domain_id)
        if info is None:
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")
        return info.version != version
