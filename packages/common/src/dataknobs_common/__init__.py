"""Common utilities and base classes for dataknobs packages.

This package provides shared functionality used across all dataknobs packages:

- **Exceptions**: Unified exception hierarchy with context support
- **Expressions**: Safe expression evaluation engine with restricted builtins
- **Registry**: Generic registry pattern for managing named items
- **Records**: Typed fields and the records built from them
- **Serialization**: Protocols and utilities for to_dict/from_dict patterns
- **Retry**: Configurable retry execution with backoff strategies
- **Transitions**: Stateless transition validation for status graphs
- **Events**: Event bus abstraction for pub/sub messaging
- **Paths**: Compose a path from an untrusted name without leaving a base
- **Testing**: Test utilities, markers, and configuration factories

Example:
    ```python
    from dataknobs_common import DataknobsError, Registry, serialize

    # Use common exceptions
    raise DataknobsError("Something went wrong", context={"details": "here"})

    # Create a registry
    registry = Registry[MyType]("my_registry")
    registry.register("key", my_item)

    # Serialize objects
    data = serialize(my_object)

    # Use event bus
    from dataknobs_common.events import create_event_bus, Event, EventType
    bus = create_event_bus({"backend": "memory"})
    ```
"""

# Import all public APIs from submodules
from dataknobs_common.expressions import (
    SAFE_BUILTINS,
    YAML_ALIASES,
    ExpressionResult,
    safe_eval,
    safe_eval_validate,
    safe_eval_value,
)
from dataknobs_common.events import (
    Event,
    EventBus,
    EventType,
    InMemoryEventBus,
    Subscription,
    create_event_bus,
    create_event_bus_async,
)
from dataknobs_common.callbacks import (
    BatchedCallbackError,
    CallbackEntry,
    CallbackOrdering,
    CallbackRegistry,
    CapturingCallbackRegistry,
    CompositeOrdering,
    ErrorPolicy,
    FIFOOrdering,
    PriorityOrdering,
    RecordingCallbackRegistry,
    StageOrdering,
    is_async_callable,
    run_callback,
    run_callback_off_loop,
)
from dataknobs_common.async_iter import (
    aiter_sync_in_thread,
)
from dataknobs_common.aws import (
    AwsSessionConfig,
    SupportsToSessionConfig,
    clear_aioboto3_session_cache,
    create_aioboto3_session,
)
from dataknobs_common.sync_bridge import (
    SyncLoopBridge,
    run_coro_sync,
)
from dataknobs_common.lifecycle import (
    aclose_if_owned,
    close_if_owned,
    close_if_owned_sync,
)
from dataknobs_common.bounded_cache import (
    BoundedLRUCache,
)
from dataknobs_common.capabilities import (
    CAPABILITY_FAMILIES,
    Capability,
    CapabilityContract,
    CapabilityLike,
    CapabilityMixin,
    CapabilityNotSupportedError,
    DynamicCapabilityMixin,
    require_capability,
    supports_capability,
)
from dataknobs_common.config_loading import (
    DEFAULT_CONFIG_EXTENSIONS,
    ConfigLoadError,
    ConfigParseError,
    ConfigPathEscapeError,
    ConfigShapeError,
    ConfigUnsupportedFormatError,
    ConfigYAMLNotInstalledError,
    find_config_file,
    load_yaml_or_json,
    parse_yaml_or_json,
)
from dataknobs_common.copying import copy_structure
from dataknobs_common.discriminator import (
    AsyncCallableDiscriminator,
    AsyncChainedDiscriminator,
    AsyncDiscriminator,
    CallableDiscriminator,
    ChainedDiscriminator,
    Discriminator,
    MappingDiscriminator,
    MultiFieldDiscriminator,
)
from dataknobs_common.entity_resolution import (
    AliasSignal,
    AsyncAliasSignal,
    AsyncCascadingResolver,
    AsyncDeclaredSignal,
    AsyncEntityResolver,
    AsyncExactNormalizedSignal,
    AsyncMatchSignal,
    BridgedEntityResolver,
    CascadingResolver,
    Coverage,
    DeclaredSignal,
    ENTITY_TYPE_KEY,
    EntityCandidate,
    EntityResolver,
    EvidenceKind,
    ExactNormalizedSignal,
    MatchEvidence,
    MatchSignal,
    ResolutionResult,
    ScopeAuthority,
    Within,
    async_signal_backends,
    refuse_unknown_axes,
    signal_backends,
    within_admits,
    within_axes,
    within_axis_names,
    within_memberships,
)
from dataknobs_common.exceptions import (
    ConcurrencyError,
    ConfigurationError,
    ConsentRequiredError,
    DataknobsError,
    DottedPathError,
    DottedPathReason,
    DottedPathTypeError,
    NotFoundError,
    OperationError,
    RateLimitError,
    ResourceError,
    SerializationError,
    TimeoutError,
    ValidationError,
)
from dataknobs_common.fields import (
    Field,
    FieldType,
    field_type_backends,
    register_field_class,
)
from dataknobs_common.hierarchy import (
    Ask,
    AsyncBulkHierarchy,
    AsyncEnumerableHierarchy,
    AsyncHierarchy,
    AsyncMappingHierarchy,
    BulkHierarchy,
    DEFAULT_FRONTIER_CONCURRENCY,
    EnumerableHierarchy,
    Hierarchy,
    MappingHierarchy,
    Member,
    Walk,
    ancestors,
    async_ancestors,
    async_drive,
    drive,
)
from dataknobs_common.imports import (
    ClassConstraint,
    dotted_path,
    resolve_callable,
    resolve_class,
    resolve_dotted,
    resolve_optional_callable,
)
from dataknobs_common.locks import (
    DistributedLock,
    FileLock,
    InProcessLock,
    LockFactory,
    create_lock,
    create_lock_async,
    lock_backends,
)
from dataknobs_common.metadata import enforce_immutable_keys
from dataknobs_common.ontology import (
    AliasFormSource,
    Assertion,
    AssertionHierarchy,
    AssertionSource,
    AsyncAliasFormSource,
    AsyncAssertionHierarchy,
    AsyncAssertionSource,
    AsyncEntitySource,
    AsyncMappingAssertionSource,
    AsyncMappingEntitySource,
    AsyncOntology,
    AttributeDef,
    AUTHORED_SOURCE_ID,
    AUTHORED_SOURCE_KINDS,
    CompatibilityVerdict,
    CyclePolicy,
    DEFAULT_NESTED_RELATION,
    DK_ENTITY_TYPE,
    DK_RELATION_TYPE,
    EdgeCriteria,
    Entity,
    ENTITY_TYPE_ISA_KEY,
    EntityRef,
    EntitySource,
    EntityType,
    InferenceMode,
    Literal,
    MappingAssertionSource,
    MappingEntitySource,
    Materialization,
    MembershipOracle,
    Ontology,
    OntologyConfig,
    OntologyParts,
    ParentChoice,
    Polarity,
    ProjectionContext,
    Provenance,
    QualifiedId,
    RelationRef,
    RelationType,
    RESERVED_ONTOLOGY_ID,
    ResolutionRef,
    Scoring,
    SiblingOrder,
    SourceDescription,
    SourceRef,
    TaxonomyDefinition,
    Term,
    TreeProjection,
    async_build_resolver,
    async_load_ontology,
    build_ontology,
    build_resolver,
    default_normalizer,
    edge_criteria,
    load_ontology,
    qualify,
    relation_id,
    split_qualified,
)
from dataknobs_common.packs import (
    UNSET,
    CompositionRule,
    MergeKind,
    PackRegistry,
    PackResolution,
    PackResolutionError,
    PackResolutionReason,
    PackSpec,
    PackWarning,
    PackWarningCode,
    Reducer,
    compose_packs,
    merge_bindings,
)
from dataknobs_common.paths import (
    PathAnchor,
    PathEscapeError,
    SegmentEscapeError,
    safe_join,
    safe_join_or_raise,
    safe_segment,
)
from dataknobs_common.postgres_config import (
    build_postgres_dsn,
    normalize_postgres_connection_config,
)
from dataknobs_common.ratelimit import (
    InMemoryRateLimiter,
    RateLimit,
    RateLimiter,
    RateLimiterConfig,
    RateLimitStatus,
    create_rate_limiter,
    create_rate_limiter_async,
    rate_limiter_backends,
)
from dataknobs_common.retry import (
    BackoffStrategy,
    RetryConfig,
    RetryExecutor,
    compute_backoff_delay,
)
from dataknobs_common.transitions import (
    InvalidTransitionError,
    TransitionValidator,
)
from dataknobs_common.records import Record
from dataknobs_common.registry import (
    AsyncRegistry,
    BackendRegistry,
    CachedRegistry,
    PluginFactory,
    PluginRegistry,
    Registry,
)
from dataknobs_common.resolver import (
    AsyncCachedResolver,
    AsyncCallableResolver,
    AsyncResourceResolver,
    CachedResolver,
    CallablePartitionResolver,
    CallableResolver,
    CompositeResolver,
    DefaultingResolver,
    JoiningPartitionResolver,
    MappingResolver,
    MetadataKeyPartitionResolver,
    NullPartitionResolver,
    NullResolver,
    ResourceResolver,
    TemporalPartitionResolver,
    partition_resolver_backends,
    resolver_backends,
)
from dataknobs_common.scope import (
    CachedProjector,
    CallableProjector,
    ChainedProjector,
    IdentityProjector,
    ReadOnlyProjector,
    ScopeProjector,
    WhitelistProjector,
)
from dataknobs_common.serialization import (
    Serializable,
    deserialize,
    deserialize_list,
    is_deserializable,
    is_serializable,
    jsonify,
    serialize,
    serialize_list,
)
from dataknobs_common.structured_config import (
    SKIP_VALIDATION,
    ConfigClassResolution,
    ConfigClassResolver,
    StructuredConfig,
    StructuredConfigConsumer,
    config_registries,
    register_sensitive_interior_key,
)
from dataknobs_common.tenancy import (
    BoundTenantContext,
    PrefixedTenantContext,
    SharedCorpusTenantContext,
    SingleTenantContext,
    TenantContext,
    create_tenant_context,
    tenant_context_from_env,
)
from dataknobs_common.testing import (
    create_test_json_files,
    create_test_markdown_files,
    get_test_bot_config,
    get_test_rag_config,
    is_chromadb_available,
    is_faiss_available,
    is_ollama_available,
    is_ollama_model_available,
    is_ollama_model_usable,
    is_package_available,
    is_redis_available,
    requires_chromadb,
    requires_faiss,
    requires_ollama,
    requires_ollama_model,
    requires_ollama_usable_model,
    requires_package,
    requires_redis,
)

__version__ = "3.2.0"

__all__ = [
    # Version
    "__version__",
    # Expressions
    "SAFE_BUILTINS",
    "YAML_ALIASES",
    "ExpressionResult",
    "safe_eval",
    "safe_eval_validate",
    "safe_eval_value",
    # Events
    "Event",
    "EventBus",
    "EventType",
    "InMemoryEventBus",
    "Subscription",
    "create_event_bus",
    "create_event_bus_async",
    # AWS session
    "AwsSessionConfig",
    "SupportsToSessionConfig",
    "clear_aioboto3_session_cache",
    "create_aioboto3_session",
    # Exceptions
    "DataknobsError",
    "ValidationError",
    "ConfigurationError",
    "ResourceError",
    "NotFoundError",
    "ConsentRequiredError",
    "OperationError",
    "ConcurrencyError",
    "SerializationError",
    "TimeoutError",
    "RateLimitError",
    "DottedPathError",
    "DottedPathReason",
    "DottedPathTypeError",
    # Dotted-path resolution
    #
    # Deliberately its own block, and deliberately NOT appended to the
    # "Resource resolvers" group below. Those resolve a *key* to a value
    # already in hand; these import a module and hand back what is in it.
    # `CallableResolver` (wraps a callable) and `resolve_callable` (returns
    # one) sitting adjacent in `__all__` is precisely the misreading that
    # made the primitive hard to find in the first place.
    #
    # `ClassConstraint` is exported for the same reason it exists: a consumer
    # wrapping `resolve_class` needs to annotate its own `base` parameter, and
    # would otherwise rediscover the `type[T]` problem and solve it worse.
    "ClassConstraint",
    "resolve_callable",
    "resolve_class",
    "dotted_path",
    "resolve_dotted",
    "resolve_optional_callable",
    # Callbacks
    "BatchedCallbackError",
    "CallbackEntry",
    "CallbackOrdering",
    "CallbackRegistry",
    "CapturingCallbackRegistry",
    "CompositeOrdering",
    "ErrorPolicy",
    "FIFOOrdering",
    "PriorityOrdering",
    "RecordingCallbackRegistry",
    "StageOrdering",
    "is_async_callable",
    "run_callback",
    "run_callback_off_loop",
    # Async iteration
    "aiter_sync_in_thread",
    # Async->sync bridge
    "SyncLoopBridge",
    "run_coro_sync",
    # Lifecycle
    "aclose_if_owned",
    "close_if_owned",
    "close_if_owned_sync",
    # Bounded cache
    "BoundedLRUCache",
    # Capabilities
    "CAPABILITY_FAMILIES",
    "Capability",
    "CapabilityContract",
    "CapabilityLike",
    "CapabilityMixin",
    "DynamicCapabilityMixin",
    "CapabilityNotSupportedError",
    "require_capability",
    "supports_capability",
    # Config loading
    "DEFAULT_CONFIG_EXTENSIONS",
    "ConfigLoadError",
    "ConfigParseError",
    "ConfigPathEscapeError",
    "ConfigShapeError",
    "ConfigUnsupportedFormatError",
    "ConfigYAMLNotInstalledError",
    "find_config_file",
    "load_yaml_or_json",
    "parse_yaml_or_json",
    # Path containment
    "PathAnchor",
    "PathEscapeError",
    "SegmentEscapeError",
    "safe_join",
    "safe_join_or_raise",
    "safe_segment",
    # Record fields
    #
    # The vocabulary a Record is made of, and the registry Field.from_dict
    # dispatches through. `VectorField` is NOT here: it needs numpy at
    # runtime, so it stays in `dataknobs_data.fields` and registers itself.
    "Field",
    "FieldType",
    "Record",
    "field_type_backends",
    "register_field_class",
    # Discriminators
    "Discriminator",
    "AsyncDiscriminator",
    "CallableDiscriminator",
    "MappingDiscriminator",
    "MultiFieldDiscriminator",
    "ChainedDiscriminator",
    "AsyncCallableDiscriminator",
    "AsyncChainedDiscriminator",
    # Distributed locks
    "DistributedLock",
    "create_lock",
    "create_lock_async",
    "lock_backends",
    "LockFactory",
    "InProcessLock",
    # Advisory single-file lock — a separate primitive, not a backend
    "FileLock",
    # Structural copying — between dict() and copy.deepcopy()
    "copy_structure",
    # Metadata helpers
    "enforce_immutable_keys",
    # Pack composition
    "UNSET",
    "MergeKind",
    "Reducer",
    "CompositionRule",
    "PackSpec",
    "PackWarning",
    "PackWarningCode",
    "PackResolution",
    "PackResolutionError",
    "PackResolutionReason",
    "compose_packs",
    "merge_bindings",
    "PackRegistry",
    # Postgres config
    "build_postgres_dsn",
    "normalize_postgres_connection_config",
    # Rate Limiting
    "RateLimiter",
    "create_rate_limiter",
    "create_rate_limiter_async",
    "rate_limiter_backends",
    "RateLimit",
    "RateLimiterConfig",
    "RateLimitStatus",
    "InMemoryRateLimiter",
    # Retry
    "BackoffStrategy",
    "RetryConfig",
    "RetryExecutor",
    "compute_backoff_delay",
    # Transitions
    "InvalidTransitionError",
    "TransitionValidator",
    # Registry
    "AsyncRegistry",
    "BackendRegistry",
    "CachedRegistry",
    "PluginFactory",
    "PluginRegistry",
    "Registry",
    # Resource resolvers
    "ResourceResolver",
    "AsyncResourceResolver",
    "MappingResolver",
    "CallableResolver",
    "DefaultingResolver",
    "CachedResolver",
    "CompositeResolver",
    "NullResolver",
    "AsyncCallableResolver",
    "AsyncCachedResolver",
    "NullPartitionResolver",
    "MetadataKeyPartitionResolver",
    "TemporalPartitionResolver",
    "CallablePartitionResolver",
    "JoiningPartitionResolver",
    "partition_resolver_backends",
    "resolver_backends",
    # Scope projectors
    "ScopeProjector",
    "IdentityProjector",
    "ReadOnlyProjector",
    "WhitelistProjector",
    "ChainedProjector",
    "CallableProjector",
    "CachedProjector",
    # Tenancy
    "TenantContext",
    "SingleTenantContext",
    "BoundTenantContext",
    "PrefixedTenantContext",
    "SharedCorpusTenantContext",
    "create_tenant_context",
    "tenant_context_from_env",
    # Serialization
    "Serializable",
    "serialize",
    "deserialize",
    "serialize_list",
    "deserialize_list",
    "is_serializable",
    "is_deserializable",
    "jsonify",
    # Structured configuration
    "SKIP_VALIDATION",
    "ConfigClassResolution",
    "ConfigClassResolver",
    "StructuredConfig",
    "StructuredConfigConsumer",
    "config_registries",
    "register_sensitive_interior_key",
    # Ontology — an authored vocabulary, and what loads and reads one
    #
    # `Taxonomy`, `AsyncTaxonomy`, `TaxonomyView` and `AsyncTaxonomyView` are
    # deliberately absent. Each still gains members, and a name on this list is
    # a promise; they are reachable at `dataknobs_common.ontology.taxonomy`,
    # and a module path is not a claim of public API. Nothing needs to import
    # them to use them — `onto.taxonomy("species")` returns one.
    "AliasFormSource",
    "Assertion",
    "AssertionHierarchy",
    "AssertionSource",
    "AsyncAliasFormSource",
    "AsyncAssertionHierarchy",
    "AsyncAssertionSource",
    "AsyncEntitySource",
    "AsyncMappingAssertionSource",
    "AsyncMappingEntitySource",
    "AsyncOntology",
    "AttributeDef",
    "AUTHORED_SOURCE_ID",
    "AUTHORED_SOURCE_KINDS",
    "CompatibilityVerdict",
    "CyclePolicy",
    "DEFAULT_NESTED_RELATION",
    "DK_ENTITY_TYPE",
    "DK_RELATION_TYPE",
    "EdgeCriteria",
    "Entity",
    "ENTITY_TYPE_ISA_KEY",
    "EntityRef",
    "EntitySource",
    "EntityType",
    "InferenceMode",
    "Literal",
    "MappingAssertionSource",
    "MappingEntitySource",
    "Materialization",
    "MembershipOracle",
    "Ontology",
    "OntologyConfig",
    "OntologyParts",
    "ParentChoice",
    "Polarity",
    "ProjectionContext",
    "Provenance",
    "QualifiedId",
    "RelationRef",
    "RelationType",
    "RESERVED_ONTOLOGY_ID",
    "ResolutionRef",
    "Scoring",
    "SiblingOrder",
    "SourceDescription",
    "SourceRef",
    "TaxonomyDefinition",
    "Term",
    "TreeProjection",
    "async_build_resolver",
    "async_load_ontology",
    "build_ontology",
    "build_resolver",
    "default_normalizer",
    "edge_criteria",
    "load_ontology",
    "qualify",
    "relation_id",
    "split_qualified",
    # Hierarchy — the structural protocols, and the walks generic over them
    #
    # The walks are module-level functions rather than methods, so implementing
    # `Hierarchy` earns all of them and overriding none. `HierarchyView` and
    # `AsyncHierarchyView` are absent for the same reason the taxonomy views
    # are, and sit at `dataknobs_common.hierarchy`.
    "Ask",
    "AsyncBulkHierarchy",
    "AsyncEnumerableHierarchy",
    "AsyncHierarchy",
    "AsyncMappingHierarchy",
    "BulkHierarchy",
    "DEFAULT_FRONTIER_CONCURRENCY",
    "EnumerableHierarchy",
    "Hierarchy",
    "MappingHierarchy",
    "Member",
    "Walk",
    "ancestors",
    "async_ancestors",
    "async_drive",
    "drive",
    # Entity resolution — placing a surface form in a vocabulary
    #
    # The extension point is `MatchSignal` and the `signal_backends` registry,
    # both here: a consumer adds a rung by implementing and registering one.
    # The rung merge itself — `CascadeState`, `merge_rung`, `finish` — is not
    # the extension point and is not exported; it is at
    # `dataknobs_common.entity_resolution.cascade` for anything testing against
    # the cascade's internals.
    "AliasSignal",
    "AsyncAliasSignal",
    "AsyncCascadingResolver",
    "AsyncDeclaredSignal",
    "AsyncEntityResolver",
    "AsyncExactNormalizedSignal",
    "AsyncMatchSignal",
    "BridgedEntityResolver",
    "CascadingResolver",
    "Coverage",
    "DeclaredSignal",
    "ENTITY_TYPE_KEY",
    "EntityCandidate",
    "EntityResolver",
    "EvidenceKind",
    "ExactNormalizedSignal",
    "MatchEvidence",
    "MatchSignal",
    "ResolutionResult",
    "ScopeAuthority",
    "Within",
    "async_signal_backends",
    "refuse_unknown_axes",
    "signal_backends",
    "within_admits",
    "within_axes",
    "within_axis_names",
    "within_memberships",
    # Testing - Availability Checks
    "is_ollama_available",
    "is_ollama_model_available",
    "is_ollama_model_usable",
    "is_faiss_available",
    "is_chromadb_available",
    "is_redis_available",
    "is_package_available",
    # Testing - Pytest Markers
    "requires_ollama",
    "requires_faiss",
    "requires_chromadb",
    "requires_redis",
    "requires_package",
    "requires_ollama_model",
    "requires_ollama_usable_model",
    # Testing - Configuration Factories
    "get_test_bot_config",
    "get_test_rag_config",
    # Testing - File Helpers
    "create_test_markdown_files",
    "create_test_json_files",
]
