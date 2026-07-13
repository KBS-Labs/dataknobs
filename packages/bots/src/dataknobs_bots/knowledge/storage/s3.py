"""S3-based knowledge resource backend for production deployments.

This backend stores files in Amazon S3 (or S3-compatible storage like MinIO,
LocalStack), making it ideal for production deployments.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, BinaryIO, ClassVar

from botocore.exceptions import ClientError

from dataknobs_common.aws import AwsSessionConfig, create_aioboto3_session
from dataknobs_common.capabilities import Capability, CapabilityLike
from dataknobs_common.exceptions import ConcurrencyError

from .key_layout import KnowledgeKeyKind
from .mixin import KnowledgeResourceBackendMixin
from .models import (
    IngestionStatus,
    InvalidVersionError,
    KnowledgeBaseInfo,
    KnowledgeFile,
    normalize_ingestion_status,
)

if TYPE_CHECKING:
    from dataknobs_common.tenancy import TenantContext

_CHANGE_DETECTION_MODES = ("snapshot", "s3_versioning")
# head_object surfaces a missing key as HTTP "404"; get_object as
# "NoSuchKey". Accept both wherever we translate absence into a return.
_MISSING_KEY_CODES = ("404", "NoSuchKey")
# A failed If-Match conditional PUT surfaces as HTTP 412. Real S3 sets
# the error Code to "PreconditionFailed"; accept the raw status too so
# S3-compatible services that omit the symbolic code still map to a
# ConcurrencyError.
_PRECONDITION_FAILED_CODE = "PreconditionFailed"
_PRECONDITION_FAILED_STATUS = 412

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

    # State methods honor ``ctx.state_key_prefix()`` — per-tenant ingest
    # state (metadata + snapshots) is isolated under the prefix while
    # content stays keyed by ``domain_id``. Unions onto the mixin's base
    # set (does not replace it).
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]] = (
        KnowledgeResourceBackendMixin.SUPPORTED_CAPABILITIES
        | frozenset({
            Capability.TENANT_SCOPED_STATE,
            Capability.SNAPSHOT_ISOLATION,
            # Conditional metadata writes use S3's server-enforced
            # If-Match precondition on the metadata object's ETag — the
            # race-free CAS primitive for many replicas over one bucket.
            Capability.CONDITIONAL_WRITE,
        })
    )

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
        session_config: AwsSessionConfig | None = None,
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
            session_config: Pre-built :class:`AwsSessionConfig`. When
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
            session_config = AwsSessionConfig(
                region_name=region,
                endpoint_url=endpoint_url,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
            )
        self._session_config = session_config
        # aioboto3 Session (created in initialize). Clients are short-lived
        # per-operation context managers opened off this session — the
        # async-native counterpart to a single long-lived boto3 client.
        self._session: Any | None = None
        self._client_kwargs: dict[str, Any] = {}
        self._initialized = False

    @classmethod
    def from_config(cls, config: dict) -> S3KnowledgeBackend:
        """Create from configuration dict.

        Accepts both ``region`` (legacy) and ``region_name``
        (boto-native) keys, plus the canonical ``aws_*`` credential
        keys. Routes through :meth:`AwsSessionConfig.from_dict` so the
        accepted shape stays in sync with the rest of dataknobs' S3
        constructs.
        """
        return cls(
            bucket=config["bucket"],
            prefix=config.get("prefix", "knowledge/"),
            session_config=AwsSessionConfig.from_dict(config),
            change_detection_mode=config.get(
                "change_detection_mode", "snapshot"
            ),
        )

    @property
    def bucket(self) -> str:
        """The S3 bucket name this backend writes to.

        Stable read-only accessor for the constructor's ``bucket``
        argument. Use this from event-trigger / CDK / CloudFormation
        glue code that needs the bucket name to wire up notifications
        — never reach for the private ``_bucket`` attribute.
        """
        return self._bucket

    @property
    def prefix(self) -> str:
        """The normalized S3 key prefix for this backend.

        Always ends in ``"/"`` (or is empty for a bucket-root layout).
        Stable read-only accessor for the constructor's ``prefix``
        argument after normalization. Use this from event-trigger /
        CDK / CloudFormation glue code rather than the private
        ``_prefix`` attribute.
        """
        return self._prefix

    async def initialize(self) -> None:
        """Initialize the aioboto3 session and verify bucket access.

        Note: creating the first aioboto3 client lazily loads botocore's
        service data from disk, a one-time synchronous read that briefly
        blocks the loop during this startup call. Steady-state operations
        (``put_file`` / ``get_file`` / ``list_files`` / ...) are fully
        non-blocking. Eliminating the startup read depends on an
        async-aware client-data path in the underlying data layer and is
        tracked for a follow-up there; it is deliberately out of scope
        here so this backend's per-operation guarantees aren't held up by
        a one-time init cost.
        """
        self._session = await create_aioboto3_session(self._session_config)
        # endpoint_url / use_ssl / credentials / region in one shape —
        # to_client_kwargs() handles the LocalStack/MinIO http use_ssl=False
        # case that a session-only setup misses.
        self._client_kwargs = self._session_config.to_client_kwargs()

        # Verify bucket exists
        try:
            async with self._session.client("s3", **self._client_kwargs) as s3:
                await s3.head_bucket(Bucket=self._bucket)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                raise ValueError(f"Bucket '{self._bucket}' does not exist") from e
            raise

        self._initialized = True
        logger.info("Initialized S3 backend: s3://%s/%s", self._bucket, self._prefix)

    async def close(self) -> None:
        """Release the aioboto3 session.

        Clients are per-operation context managers that close themselves;
        the session holds no open transport between calls, so dropping the
        reference is the full teardown.
        """
        self._session = None
        self._initialized = False

    def _s3_key(self, domain_id: str, path: str = "") -> str:
        """Get the S3 key for a path within a domain."""
        if path:
            return f"{self._prefix}{domain_id}/{self.CONTENT_DIR}/{path}"
        return f"{self._prefix}{domain_id}/"

    def _metadata_key(
        self, domain_id: str, ctx: TenantContext | None = None
    ) -> str:
        """Get the S3 key for a KB's metadata file.

        With ``ctx=None`` this is the pre-tenancy key
        (``{prefix}{domain}/_metadata.json``); a tenant context inserts
        ``ctx.state_key_prefix()`` between the backend prefix and the
        domain, isolating per-tenant ingest **state**.
        """
        return (
            f"{self._prefix}{self._state_prefix(ctx)}"
            f"{domain_id}/{self.METADATA_FILE}"
        )

    def _snapshot_key(
        self,
        domain_id: str,
        version: str,
        ctx: TenantContext | None = None,
    ) -> str:
        """S3 key of the snapshot object for ``version`` (snapshot mode).

        Tenant-scoped via ``ctx`` (the per-tenant snapshot lineage lives
        under the state prefix); ``ctx=None`` is the pre-tenancy key.
        """
        return (
            f"{self._prefix}{self._state_prefix(ctx)}"
            f"{domain_id}/{self.SNAPSHOTS_DIR}/{version}.json"
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

    async def _load_metadata(
        self, domain_id: str, ctx: TenantContext | None = None
    ) -> dict:
        """Load metadata from S3 (tenant-scoped via ``ctx``)."""
        if not self._session:
            raise RuntimeError("Backend not initialized")

        key = self._metadata_key(domain_id, ctx)
        try:
            async with self._session.client("s3", **self._client_kwargs) as s3:
                response = await s3.get_object(Bucket=self._bucket, Key=key)
                raw = await response["Body"].read()
            return json.loads(raw.decode("utf-8"))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in _MISSING_KEY_CODES:
                return {}
            raise

    @staticmethod
    def _is_precondition_failed(error: ClientError) -> bool:
        """Whether ``error`` is a failed If-Match precondition (HTTP 412)."""
        err = error.response.get("Error", {})
        if err.get("Code") == _PRECONDITION_FAILED_CODE:
            return True
        status = error.response.get("ResponseMetadata", {}).get(
            "HTTPStatusCode"
        )
        return status == _PRECONDITION_FAILED_STATUS

    async def _save_metadata(
        self,
        domain_id: str,
        metadata: dict,
        ctx: TenantContext | None = None,
        *,
        expected_version: str | None = None,
    ) -> None:
        """Save metadata to S3, then fire a state-write event.

        Tenant-scoped via ``ctx`` (the per-tenant metadata object lives
        under the state prefix); ``ctx=None`` writes the pre-tenancy key.

        ``expected_version`` (default ``None``) selects the write mode.
        ``None`` is the unconditional PUT (byte-identical to a backend
        with no conditional-write support). A non-``None`` token is the
        object's ETag from a prior :meth:`get_state_version`; it is passed
        as the ``IfMatch`` precondition so S3 server-side rejects the PUT
        when another writer has overwritten the object since (HTTP 412) OR
        deleted it (HTTP 404) — both surface as :class:`ConcurrencyError`,
        symmetric with the file backend (where a vanished document also
        conflicts rather than silently re-creating it).
        """
        if not self._session:
            raise RuntimeError("Backend not initialized")

        key = self._metadata_key(domain_id, ctx)
        body = json.dumps(metadata, indent=2, default=str).encode("utf-8")
        if expected_version is None:
            async with self._session.client(
                "s3", **self._client_kwargs
            ) as s3:
                await s3.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=body,
                    ContentType="application/json",
                )
        else:
            try:
                async with self._session.client(
                    "s3", **self._client_kwargs
                ) as s3:
                    await s3.put_object(
                        Bucket=self._bucket,
                        Key=key,
                        Body=body,
                        ContentType="application/json",
                        IfMatch=expected_version,
                    )
            except ClientError as e:
                # A conditional write carries a token, so two outcomes are
                # the same optimistic-concurrency conflict: the object was
                # overwritten (412 PreconditionFailed) OR it was deleted out
                # from under the token (404 NoSuchKey on the If-Match PUT).
                # Both surface as ConcurrencyError so the contract is
                # symmetric with the file backend, where a vanished document
                # (current_version=None != expected_version) already raises
                # ConcurrencyError.
                code = e.response.get("Error", {}).get("Code")
                if self._is_precondition_failed(e) or code in _MISSING_KEY_CODES:
                    raise ConcurrencyError(
                        "Knowledge-base state document was modified by a "
                        "concurrent writer",
                        context={
                            "domain_id": domain_id,
                            "expected_version": expected_version,
                        },
                    ) from e
                raise
        await self._fire_state_write(
            domain_id=domain_id,
            key=key,
            kind=KnowledgeKeyKind.METADATA,
            byte_size=len(body),
        )

    async def _kb_exists(self, domain_id: str) -> bool:
        """Check if a knowledge base exists."""
        if not self._session:
            raise RuntimeError("Backend not initialized")

        key = self._metadata_key(domain_id)
        try:
            async with self._session.client("s3", **self._client_kwargs) as s3:
                await s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in _MISSING_KEY_CODES:
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
        if not self._session:
            raise RuntimeError("Backend not initialized")

        if not await self._kb_exists(domain_id):
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        # Get content as bytes (file-like reads are offloaded off-loop).
        data = await self._read_content_bytes(content)

        # Auto-detect content type. The first mimetypes lookup lazily
        # reads the system mime database from disk, so offload it.
        if content_type is None:
            content_type = await asyncio.to_thread(
                self._guess_content_type, path
            )

        # Calculate checksum
        checksum = hashlib.md5(data).hexdigest()

        # Upload file
        key = self._s3_key(domain_id, path)
        object_metadata = (
            {"checksum": checksum}
            if not metadata
            else {**metadata, "checksum": checksum}
        )
        async with self._session.client("s3", **self._client_kwargs) as s3:
            await s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                Metadata=object_metadata,
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
        kb_metadata = await self._load_metadata(domain_id)
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

        await self._save_metadata(domain_id, kb_metadata)
        await self._record_snapshot(domain_id, files)

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
        if not self._session:
            raise RuntimeError("Backend not initialized")

        key = self._s3_key(domain_id, path)
        try:
            async with self._session.client("s3", **self._client_kwargs) as s3:
                response = await s3.get_object(Bucket=self._bucket, Key=key)
                return await response["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in _MISSING_KEY_CODES:
                return None
            raise

    async def stream_file(
        self, domain_id: str, path: str, chunk_size: int = 8192
    ) -> AsyncIterator[bytes] | None:
        """Stream file content.

        Returns ``None`` (not an empty stream) for a missing object, so a
        cheap ``head_object`` probe runs first in its own client context.
        The returned generator opens its own client context and keeps it
        alive for the lifetime of the stream — the aioboto3 ``Body`` is
        only readable while its client is open, so the ``async with`` must
        wrap the whole read loop rather than just the ``get_object`` call.
        """
        if not self._session:
            raise RuntimeError("Backend not initialized")

        key = self._s3_key(domain_id, path)
        try:
            async with self._session.client("s3", **self._client_kwargs) as s3:
                await s3.head_object(Bucket=self._bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in _MISSING_KEY_CODES:
                return None
            raise

        async def _generator() -> AsyncIterator[bytes]:
            async with self._session.client("s3", **self._client_kwargs) as s3:
                response = await s3.get_object(Bucket=self._bucket, Key=key)
                body = response["Body"]
                while True:
                    chunk = await body.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        return _generator()

    async def delete_file(self, domain_id: str, path: str) -> bool:
        """Delete a file."""
        if not self._session:
            raise RuntimeError("Backend not initialized")

        key = self._s3_key(domain_id, path)

        # Check if file exists, then delete — within one client context.
        async with self._session.client("s3", **self._client_kwargs) as s3:
            try:
                await s3.head_object(Bucket=self._bucket, Key=key)
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") in _MISSING_KEY_CODES:
                    return False
                raise
            await s3.delete_object(Bucket=self._bucket, Key=key)

        # Update metadata
        kb_metadata = await self._load_metadata(domain_id)
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

        await self._save_metadata(domain_id, kb_metadata)
        await self._record_snapshot(domain_id, files)

        logger.debug("Deleted file: s3://%s/%s", self._bucket, key)
        return True

    async def list_files(
        self, domain_id: str, prefix: str | None = None
    ) -> list[KnowledgeFile]:
        """List all files in a knowledge base."""
        kb_metadata = await self._load_metadata(domain_id)
        files_dict = kb_metadata.get("files", {})

        files = [KnowledgeFile.from_dict(f) for f in files_dict.values()]

        if prefix:
            files = [f for f in files if f.path.startswith(prefix)]

        # Sort by path for consistent ordering
        files.sort(key=lambda f: f.path)
        return files

    async def file_exists(self, domain_id: str, path: str) -> bool:
        """Check if a file exists."""
        if not self._session:
            raise RuntimeError("Backend not initialized")

        key = self._s3_key(domain_id, path)
        try:
            async with self._session.client("s3", **self._client_kwargs) as s3:
                await s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in _MISSING_KEY_CODES:
                return False
            raise

    # --- Knowledge Base Operations ---

    async def create_kb(
        self, domain_id: str, metadata: dict | None = None
    ) -> KnowledgeBaseInfo:
        """Create a new knowledge base."""
        if not self._session:
            raise RuntimeError("Backend not initialized")

        if await self._kb_exists(domain_id):
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
        await self._save_metadata(domain_id, kb_metadata)

        logger.info("Created knowledge base: s3://%s/%s%s/", self._bucket, self._prefix, domain_id)
        return kb_info

    async def get_info(
        self, domain_id: str, *, ctx: TenantContext | None = None
    ) -> KnowledgeBaseInfo | None:
        """Get knowledge base metadata.

        KB existence is checked against the domain-keyed metadata object;
        the returned ingest **state** view is read from the per-tenant
        metadata object when ``ctx`` carries a state prefix (the shared
        single-tenant object otherwise).
        """
        if not await self._kb_exists(domain_id):
            return None

        kb_metadata = await self._load_metadata(domain_id, ctx)
        info_dict = kb_metadata.get("info", {"domain_id": domain_id})
        return KnowledgeBaseInfo.from_dict(info_dict)

    async def delete_kb(self, domain_id: str) -> bool:
        """Delete entire knowledge base and all files."""
        if not self._session:
            raise RuntimeError("Backend not initialized")

        if not await self._kb_exists(domain_id):
            return False

        # Delete all objects with the domain prefix
        prefix = f"{self._prefix}{domain_id}/"
        async with self._session.client("s3", **self._client_kwargs) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self._bucket, Prefix=prefix
            ):
                contents = page.get("Contents", [])
                if not contents:
                    continue

                objects = [{"Key": obj["Key"]} for obj in contents]
                await s3.delete_objects(
                    Bucket=self._bucket,
                    Delete={"Objects": objects},
                )

        logger.info("Deleted knowledge base: s3://%s/%s", self._bucket, prefix)
        return True

    async def list_kbs(self) -> list[KnowledgeBaseInfo]:
        """List all knowledge bases."""
        if not self._session:
            raise RuntimeError("Backend not initialized")

        # List all "directories" under the prefix, collecting domain ids
        # inside the client context, then resolve each KB's info after
        # (get_info opens its own client context).
        domain_ids: list[str] = []
        async with self._session.client("s3", **self._client_kwargs) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self._bucket,
                Prefix=self._prefix,
                Delimiter="/",
            ):
                for prefix_obj in page.get("CommonPrefixes", []):
                    # Extract domain_id from prefix
                    prefix_path = prefix_obj["Prefix"]
                    domain_id = prefix_path[len(self._prefix) :].rstrip("/")
                    if domain_id:
                        domain_ids.append(domain_id)

        kbs = []
        for domain_id in domain_ids:
            info = await self.get_info(domain_id)
            if info:
                kbs.append(info)

        return sorted(kbs, key=lambda kb: kb.domain_id)

    # --- Ingestion Status ---

    async def get_state_version(
        self, domain_id: str, *, ctx: TenantContext | None = None
    ) -> str | None:
        """The metadata object's ETag (opaque), or ``None`` if absent.

        Returned verbatim as S3 hands it back (quoted), so it round-trips
        directly into the ``IfMatch`` precondition on a conditional
        :meth:`set_ingestion_status` write. ``None`` when the (per-tenant)
        metadata object does not exist. See the protocol for the
        round-trip contract.
        """
        if not self._session:
            raise RuntimeError("Backend not initialized")

        key = self._metadata_key(domain_id, ctx)
        try:
            async with self._session.client(
                "s3", **self._client_kwargs
            ) as s3:
                response = await s3.head_object(Bucket=self._bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in _MISSING_KEY_CODES:
                return None
            raise
        etag = response.get("ETag")
        return etag if etag else None

    async def set_ingestion_status(
        self,
        domain_id: str,
        status: IngestionStatus | str,
        error: str | None = None,
        *,
        generation: str | None = None,
        ctx: TenantContext | None = None,
        expected_version: str | None = None,
    ) -> None:
        """Update ingestion status for a knowledge base.

        KB existence is checked against the domain-keyed metadata object;
        the status is written to the per-tenant metadata object when
        ``ctx`` carries a state prefix (the shared single-tenant object
        otherwise).

        When ``expected_version`` is supplied, the save is conditional
        (see :meth:`_save_metadata`): the PUT carries an ``IfMatch``
        precondition on the object ETag and raises
        :class:`ConcurrencyError` if a concurrent writer overwrote the
        object first.
        """
        if not await self._kb_exists(domain_id):
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        kb_metadata = await self._load_metadata(domain_id, ctx)
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

        await self._save_metadata(
            domain_id, kb_metadata, ctx, expected_version=expected_version
        )

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

    async def _record_snapshot(
        self,
        domain_id: str,
        files: dict[str, dict],
        *,
        ctx: TenantContext | None = None,
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

        Called after :meth:`put_file` / :meth:`delete_file` (``ctx=None``
        — content-mutation snapshots stay under the domain-keyed store,
        byte-identical to the pre-tenancy layout); a tenant context
        writes the snapshot object under the per-tenant state prefix.
        """
        if self._change_detection_mode != "snapshot":
            return
        if not self._session:
            raise RuntimeError("Backend not initialized")
        snapshot = {
            path: info.get("checksum", "") for path, info in files.items()
        }
        version = self._identity_of_snapshot(snapshot)
        if not version:
            return
        key = self._snapshot_key(domain_id, version, ctx)
        body = json.dumps(snapshot).encode("utf-8")
        async with self._session.client("s3", **self._client_kwargs) as s3:
            await s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
        await self._fire_state_write(
            domain_id=domain_id,
            key=key,
            kind=KnowledgeKeyKind.SNAPSHOT,
            byte_size=len(body),
        )

    async def _load_snapshot(
        self,
        domain_id: str,
        version: str,
        *,
        ctx: TenantContext | None = None,
    ) -> dict[str, str]:
        """Resolve ``version`` to its retained ``{path: checksum}`` map.

        ``""`` is the empty-KB baseline (every current file diffs as
        ``added``). Otherwise the resolution depends on
        ``change_detection_mode``; an unresolvable version raises
        :class:`InvalidVersionError` so callers fall back to a full
        re-ingest. The snapshot is read from the tenant-scoped store
        when ``ctx`` carries a state prefix.
        """
        if not version:
            return {}
        if not self._session:
            raise RuntimeError("Backend not initialized")
        if self._change_detection_mode == "s3_versioning":
            return await self._load_snapshot_from_versions(
                domain_id, version, ctx
            )
        # snapshot mode
        try:
            async with self._session.client("s3", **self._client_kwargs) as s3:
                response = await s3.get_object(
                    Bucket=self._bucket,
                    Key=self._snapshot_key(domain_id, version, ctx),
                )
                raw = await response["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in _MISSING_KEY_CODES:
                raise InvalidVersionError(
                    f"Version {version!r} is not retained for domain "
                    f"{domain_id!r}"
                ) from e
            raise
        data: dict[str, str] = json.loads(raw.decode("utf-8"))
        return data

    async def _load_snapshot_from_versions(
        self,
        domain_id: str,
        version: str,
        ctx: TenantContext | None = None,
    ) -> dict[str, str]:
        """Walk the metadata object's S3 version history for ``version``.

        Each historical metadata version reconstructs a
        ``{path: checksum}`` map; the first whose canonical identity
        equals ``version`` is the answer. Requires bucket versioning;
        with it disabled only the current version is listed, so a stale
        ``version`` raises :class:`InvalidVersionError` (a correct,
        non-minimal full re-ingest — never a wrong diff). The metadata
        object walked is the tenant-scoped one when ``ctx`` carries a
        state prefix.
        """
        if not self._session:
            raise RuntimeError("Backend not initialized")
        metadata_key = self._metadata_key(domain_id, ctx)
        async with self._session.client("s3", **self._client_kwargs) as s3:
            paginator = s3.get_paginator("list_object_versions")
            async for page in paginator.paginate(
                Bucket=self._bucket, Prefix=metadata_key
            ):
                # DeleteMarkers are returned under "DeleteMarkers", not
                # "Versions", so iterating "Versions" already excludes them
                # (the metadata object is overwritten, never deleted, anyway).
                for entry in page.get("Versions", []):
                    if entry["Key"] != metadata_key:
                        continue  # Prefix is a prefix-match; require exact key
                    obj = await s3.get_object(
                        Bucket=self._bucket,
                        Key=metadata_key,
                        VersionId=entry["VersionId"],
                    )
                    raw = await obj["Body"].read()
                    metadata = json.loads(raw.decode("utf-8"))
                    snapshot = self._snapshot_from_metadata(metadata)
                    if self._identity_of_snapshot(snapshot) == version:
                        return snapshot
        raise InvalidVersionError(
            f"Version {version!r} is not resolvable from the S3 "
            f"version history of domain {domain_id!r} (bucket "
            f"versioning may be disabled or the version predates it)"
        )
