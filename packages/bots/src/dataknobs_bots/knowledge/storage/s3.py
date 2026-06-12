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
from typing import TYPE_CHECKING, BinaryIO

from botocore.exceptions import ClientError

from dataknobs_data.pooling.s3 import S3SessionConfig, create_boto3_s3_client

from .key_layout import KnowledgeKeyKind
from .mixin import KnowledgeResourceBackendMixin
from .models import (
    IngestionStatus,
    InvalidVersionError,
    KnowledgeBaseInfo,
    KnowledgeFile,
    normalize_ingestion_status,
)

_CHANGE_DETECTION_MODES = ("snapshot", "s3_versioning")

if TYPE_CHECKING:
    import boto3

logger = logging.getLogger(__name__)


class S3KnowledgeBackend(KnowledgeResourceBackendMixin):
    """S3-backed knowledge resource storage.

    Structure in S3:
        s3://{bucket}/{prefix}/
            {domain_id}/
                content/              # consumer-controlled (put_file)
                    file1.md
                    subdir/file2.json
                _metadata.json        # DK-managed state
                _snapshots/           # DK-managed state (snapshot mode)
                    <version>.json

    Suitable for:
    - Production deployments
    - Multi-instance deployments
    - Large knowledge bases

    Event triggers (S3 → EventBridge / SQS / SNS / Lambda) MUST filter
    to the ``content/`` subtree to avoid retriggering on the DK-managed
    ``_metadata.json`` and ``_snapshots/`` writes the manager emits
    during ingest. Call :meth:`key_pattern` to derive the appropriate
    S3 wildcard for the configured prefix (forwarded verbatim to
    EventBridge ``wildcard`` rules or composed into bucket-notification
    ``prefix`` + ``suffix`` pairs); see the
    "Event triggers for knowledge backends" docs page for per-source
    recipes. Use :meth:`classify_key` for per-event filtering when
    pattern-based filtering at the source is unavailable.

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

    def __init__(
        self,
        bucket: str,
        prefix: str = "knowledge/",
        region: str | None = None,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        *,
        session_config: S3SessionConfig | None = None,
        change_detection_mode: str = "snapshot",
    ) -> None:
        """Initialize the S3 backend.

        Args:
            bucket: S3 bucket name.
            prefix: Key prefix for all objects (default: ``"knowledge/"``).
            region: AWS region. Defaults to ``None`` — boto's default
                chain (``AWS_DEFAULT_REGION`` env, ``~/.aws/config``,
                EC2/ECS instance metadata, then ``us-east-1`` as
                terminal fallback) resolves the value. Note that
                botocore reads ``AWS_DEFAULT_REGION``, not ``AWS_REGION``.
            endpoint_url: Custom endpoint URL (for S3-compatible services).
            aws_access_key_id: AWS access key (optional, uses default chain).
            aws_secret_access_key: AWS secret key (optional, uses default chain).
            aws_session_token: AWS session token for temporary credentials
                (optional, uses default chain when unset).
            session_config: Pre-built :class:`S3SessionConfig`. When
                provided, it wins over the individual kwargs above —
                useful for sharing one config across multiple backends.
            change_detection_mode: How per-version snapshots are
                resolved for minimal :meth:`list_changes_since` diffs.

                - ``"snapshot"`` (default): a small ``{path: checksum}``
                  JSON object is written under
                  ``{domain}/_snapshots/<version>.json`` after every
                  mutation. Self-contained; works on any bucket.
                - ``"s3_versioning"``: no extra objects are written —
                  the metadata object's own S3 version history *is* the
                  snapshot store, walked via ``ListObjectVersions``.
                  Requires **bucket versioning enabled** (a deploy-time
                  responsibility); with versioning off only the current
                  version resolves and stale versions fall back to a
                  full re-ingest (correct, non-minimal).

        Raises:
            ValueError: If ``change_detection_mode`` is not one of
                ``"snapshot"`` / ``"s3_versioning"`` (fail closed —
                an unrecognized mode is never silently treated as a
                default).
        """
        if change_detection_mode not in _CHANGE_DETECTION_MODES:
            raise ValueError(
                f"Unknown change_detection_mode "
                f"{change_detection_mode!r}; expected one of "
                f"{list(_CHANGE_DETECTION_MODES)}"
            )
        self._change_detection_mode = change_detection_mode
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        if session_config is None:
            session_config = S3SessionConfig(
                region_name=region,
                endpoint_url=endpoint_url,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
            )
        self._session_config = session_config
        self._client: boto3.client | None = None
        self._initialized = False

    @classmethod
    def from_config(cls, config: dict) -> S3KnowledgeBackend:
        """Create from configuration dict.

        Accepts both ``region`` (legacy) and ``region_name``
        (boto-native) keys, plus the canonical ``aws_*`` credential
        keys. Routes through :meth:`S3SessionConfig.from_dict` so the
        accepted shape stays in sync with the rest of dataknobs' S3
        constructs.
        """
        return cls(
            bucket=config["bucket"],
            prefix=config.get("prefix", "knowledge/"),
            session_config=S3SessionConfig.from_dict(config),
            change_detection_mode=config.get(
                "change_detection_mode", "snapshot"
            ),
        )

    async def initialize(self) -> None:
        """Initialize the S3 client and verify bucket access."""
        self._client = create_boto3_s3_client(self._session_config)

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

    def _snapshot_key(self, domain_id: str, version: str) -> str:
        """S3 key of the snapshot object for ``version`` (snapshot mode)."""
        return (
            f"{self._prefix}{domain_id}/{self.SNAPSHOTS_DIR}/"
            f"{version}.json"
        )

    def key_pattern(
        self,
        kind: KnowledgeKeyKind = KnowledgeKeyKind.CONTENT,
        domain_id: str | None = None,
    ) -> str:
        """S3-native wildcard pattern matching keys of the given kind.

        Forwarded verbatim into EventBridge ``wildcard`` rules or
        composed into S3 bucket-notification ``prefix`` + ``suffix``
        pairs. See the "Event triggers for knowledge backends" docs
        page for per-source recipes.

        :attr:`KnowledgeKeyKind.UNKNOWN` raises :class:`ValueError`
        (fails closed — there is no shape for "unrecognized keys").
        """
        domain_segment = domain_id if domain_id else "*"
        if kind is KnowledgeKeyKind.CONTENT:
            return (
                f"{self._prefix}{domain_segment}/{self.CONTENT_DIR}/*"
            )
        if kind is KnowledgeKeyKind.METADATA:
            return (
                f"{self._prefix}{domain_segment}/{self.METADATA_FILE}"
            )
        if kind is KnowledgeKeyKind.SNAPSHOT:
            return (
                f"{self._prefix}{domain_segment}/{self.SNAPSHOTS_DIR}/*"
            )
        raise ValueError(
            f"key_pattern is not defined for kind {kind!r} "
            f"(only CONTENT / METADATA / SNAPSHOT)"
        )

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
        self._record_snapshot(domain_id, files)

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
        self._record_snapshot(domain_id, files)

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
        status: IngestionStatus | str,
        error: str | None = None,
        *,
        generation: str | None = None,
    ) -> None:
        """Update ingestion status for a knowledge base."""
        if not self._kb_exists(domain_id):
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        kb_metadata = self._load_metadata(domain_id)
        info_dict = kb_metadata.get("info", {"domain_id": domain_id})
        # Persist the string value so the JSON round-trips and
        # KnowledgeBaseInfo.from_dict rehydrates the enum.
        info_dict["ingestion_status"] = normalize_ingestion_status(
            status
        ).value
        info_dict["ingestion_error"] = error
        # Always written through: a non-SWAPPING transition passes the
        # default None and so clears any stale in-flight swap token.
        info_dict["generation"] = generation
        kb_metadata["info"] = info_dict

        self._save_metadata(domain_id, kb_metadata)

    # --- Change Detection ---
    #
    # get_checksum / has_changes_since / list_changes_since come from
    # KnowledgeResourceBackendMixin (one canonical algorithm over
    # list_files()). This backend overrides _load_snapshot so
    # list_changes_since yields a minimal file-level diff. Two
    # strategies, selected by change_detection_mode:
    #   - "snapshot": a {path: checksum} object per version (mirrors the
    #     file backend).
    #   - "s3_versioning": the metadata object's own S3 version history
    #     IS the snapshot store (no extra writes) — the fast path.

    @staticmethod
    def _snapshot_from_metadata(metadata: dict) -> dict[str, str]:
        """Reconstruct a ``{path: checksum}`` map from a KB metadata dict."""
        files = metadata.get("files", {})
        return {
            path: info.get("checksum", "")
            for path, info in files.items()
        }

    def _record_snapshot(
        self, domain_id: str, files: dict[str, dict]
    ) -> None:
        """Persist the post-mutation ``{path: checksum}`` map.

        No-op in ``s3_versioning`` mode — there the metadata object's
        own version history is the snapshot store, so writing a separate
        object would be redundant. In ``snapshot`` mode the map is
        written under ``{domain}/_snapshots/<version>.json`` keyed by the
        canonical :meth:`get_checksum` identity (computed from the same
        map via the shared mixin formula). The empty KB has identity
        ``""`` and needs no object — :meth:`_load_snapshot` resolves
        ``""`` to the empty snapshot directly.
        """
        if self._change_detection_mode != "snapshot":
            return
        if not self._client:
            raise RuntimeError("Backend not initialized")
        snapshot = {
            path: info.get("checksum", "") for path, info in files.items()
        }
        version = self._identity_of_snapshot(snapshot)
        if not version:
            return
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._snapshot_key(domain_id, version),
            Body=json.dumps(snapshot).encode("utf-8"),
            ContentType="application/json",
        )

    async def _load_snapshot(
        self, domain_id: str, version: str
    ) -> dict[str, str]:
        """Resolve ``version`` to its retained ``{path: checksum}`` map.

        ``""`` is the empty-KB baseline (every current file diffs as
        ``added``). Otherwise the resolution depends on
        ``change_detection_mode``; an unresolvable version raises
        :class:`InvalidVersionError` so callers fall back to a full
        re-ingest.
        """
        if not version:
            return {}
        if not self._client:
            raise RuntimeError("Backend not initialized")
        if self._change_detection_mode == "s3_versioning":
            return self._load_snapshot_from_versions(domain_id, version)
        # snapshot mode
        try:
            response = self._client.get_object(
                Bucket=self._bucket,
                Key=self._snapshot_key(domain_id, version),
            )
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                raise InvalidVersionError(
                    f"Version {version!r} is not retained for domain "
                    f"{domain_id!r}"
                ) from e
            raise
        data: dict[str, str] = json.loads(
            response["Body"].read().decode("utf-8")
        )
        return data

    def _load_snapshot_from_versions(
        self, domain_id: str, version: str
    ) -> dict[str, str]:
        """Walk the metadata object's S3 version history for ``version``.

        Each historical metadata version reconstructs a
        ``{path: checksum}`` map; the first whose canonical identity
        equals ``version`` is the answer. Requires bucket versioning;
        with it disabled only the current version is listed, so a stale
        ``version`` raises :class:`InvalidVersionError` (a correct,
        non-minimal full re-ingest — never a wrong diff).
        """
        if not self._client:
            raise RuntimeError("Backend not initialized")
        metadata_key = self._metadata_key(domain_id)
        paginator = self._client.get_paginator("list_object_versions")
        for page in paginator.paginate(
            Bucket=self._bucket, Prefix=metadata_key
        ):
            # DeleteMarkers are returned under "DeleteMarkers", not
            # "Versions", so iterating "Versions" already excludes them
            # (the metadata object is overwritten, never deleted, anyway).
            for entry in page.get("Versions", []):
                if entry["Key"] != metadata_key:
                    continue  # Prefix is a prefix-match; require exact key
                obj = self._client.get_object(
                    Bucket=self._bucket,
                    Key=metadata_key,
                    VersionId=entry["VersionId"],
                )
                metadata = json.loads(
                    obj["Body"].read().decode("utf-8")
                )
                snapshot = self._snapshot_from_metadata(metadata)
                if self._identity_of_snapshot(snapshot) == version:
                    return snapshot
        raise InvalidVersionError(
            f"Version {version!r} is not resolvable from the S3 "
            f"version history of domain {domain_id!r} (bucket "
            f"versioning may be disabled or the version predates it)"
        )
