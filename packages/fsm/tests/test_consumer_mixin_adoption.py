"""Construction-path tests for the FSM consumer-mixin adoption.

The FSM pattern/runtime consumers — ``CircuitBreaker``, ``Bulkhead``,
``ErrorRecoveryWorkflow``, ``APIOrchestrator``, ``DatabaseETL``,
``FileProcessor``, ``StreamContext``, ``AsyncStreamContext`` — are built
from a frozen ``StructuredConfig`` via
:class:`~dataknobs_common.structured_config.StructuredConfigConsumer`.
Each gains a uniform construction surface: a typed-config ctor, a
dict-dispatch ``cls.from_config({...})``, and a typed read-only
``self.config`` property — replacing the per-class
``def __init__(self, config): self.config = config`` boilerplate.

These tests pin that contract per adopter: the typed-config, dict, and
``from_config`` paths reach identical config state; ``self.config`` is the
typed config (not a dict); mixing a typed config with loose kwargs raises
``TypeError``; the derived ``_setup`` attributes are computed; and the
parity guard (:func:`assert_structured_config_consumer`) holds — including
the MRO-ordering check (the consumer mixin must precede other bases).

``ResourcePool`` is also an adopter, but it carries a required ``provider``
collaborator (a live resource provider, not config data). It keeps the
back-compat ``ResourcePool(provider, config=None)`` positional shortcut
(mirroring ``PostgresEventBus``'s ``connection_string`` shortcut): the
provider is funnelled through the mixin's collaborator channel while the
config flows onto ``self.config``, and ``from_config(config, provider=...)``
delivers the provider alongside the config. Its provider-first ctor is
covered by dedicated tests rather than the generic config-first matrix.

No external service is required — construction only.
"""

from __future__ import annotations

import pytest
from dataknobs_common.structured_config import StructuredConfigConsumer
from dataknobs_common.testing import assert_structured_config_consumer

from dataknobs_fsm.patterns.api_orchestration import (
    APIEndpoint,
    APIOrchestrationConfig,
    APIOrchestrator,
)
from dataknobs_fsm.patterns.error_recovery import (
    Bulkhead,
    BulkheadConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    ErrorRecoveryConfig,
    ErrorRecoveryWorkflow,
    RecoveryStrategy,
)
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig
from dataknobs_fsm.patterns.file_processing import (
    FileFormat,
    FileProcessingConfig,
    FileProcessor,
)
from dataknobs_fsm.resources.pool import PoolConfig, ResourcePool
from dataknobs_fsm.streaming.core import (
    AsyncStreamContext,
    StreamConfig,
    StreamContext,
    StreamStatus,
)


class _FakeResourceProvider:
    """Minimal duck-typed resource provider (no external resource).

    Implements only the surface ``ResourcePool`` calls — ``name`` and
    ``acquire`` / ``release`` / ``validate``.
    """

    name = "fake"

    def acquire(self) -> object:
        return object()

    def release(self, _resource: object) -> None:
        pass

    def validate(self, _resource: object) -> bool:
        return True

# (consumer_cls, config_cls, typed_config, equivalent_dict, default_constructible)
ADOPTERS = [
    (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerConfig(failure_threshold=3),
        {"failure_threshold": 3},
        True,
    ),
    (
        Bulkhead,
        BulkheadConfig,
        BulkheadConfig(max_concurrent=2),
        {"max_concurrent": 2},
        True,
    ),
    (
        ErrorRecoveryWorkflow,
        ErrorRecoveryConfig,
        ErrorRecoveryConfig(primary_strategy=RecoveryStrategy.RETRY),
        {"primary_strategy": "retry"},
        False,
    ),
    (
        APIOrchestrator,
        APIOrchestrationConfig,
        APIOrchestrationConfig(endpoints=[APIEndpoint(name="a", url="http://x")]),
        {"endpoints": [{"name": "a", "url": "http://x"}]},
        False,
    ),
    (
        DatabaseETL,
        ETLConfig,
        ETLConfig(source_db={"type": "memory"}, target_db={"type": "memory"}),
        {"source_db": {"type": "memory"}, "target_db": {"type": "memory"}},
        False,
    ),
    (
        FileProcessor,
        FileProcessingConfig,
        FileProcessingConfig(input_path="data.json"),
        {"input_path": "data.json"},
        False,
    ),
    (
        StreamContext,
        StreamConfig,
        StreamConfig(buffer_size=42),
        {"buffer_size": 42},
        True,
    ),
    (
        AsyncStreamContext,
        StreamConfig,
        StreamConfig(parallelism=3),
        {"parallelism": 3},
        True,
    ),
]

_IDS = [a[0].__name__ for a in ADOPTERS]


@pytest.mark.parametrize("consumer_cls", [a[0] for a in ADOPTERS], ids=_IDS)
def test_is_structured_config_consumer(consumer_cls: type) -> None:
    """Each adopter mixes in StructuredConfigConsumer."""
    assert issubclass(consumer_cls, StructuredConfigConsumer)


@pytest.mark.parametrize("consumer_cls", [a[0] for a in ADOPTERS], ids=_IDS)
def test_parity_guard(consumer_cls: type) -> None:
    """The unified parity contract holds (CONFIG_CLS, field/ctor match, MRO)."""
    assert_structured_config_consumer(consumer_cls)


@pytest.mark.parametrize(
    "consumer_cls, config_cls, typed_config, equivalent_dict, _default",
    ADOPTERS,
    ids=_IDS,
)
def test_typed_dict_and_from_config_reach_same_state(
    consumer_cls, config_cls, typed_config, equivalent_dict, _default
) -> None:
    """Typed-config ctor, dict ctor, and from_config reach identical config."""
    via_typed = consumer_cls(typed_config)
    via_dict = consumer_cls(equivalent_dict)
    via_from_config = consumer_cls.from_config(equivalent_dict)

    # ``self.config`` is the typed config (not a dict).
    assert isinstance(via_typed.config, config_cls)
    assert isinstance(via_dict.config, config_cls)
    assert isinstance(via_from_config.config, config_cls)

    # All three paths produce an equal config.
    assert via_typed.config == typed_config
    assert via_dict.config == typed_config
    assert via_from_config.config == typed_config


@pytest.mark.parametrize(
    "consumer_cls, _config_cls, typed_config, _equivalent_dict, _default",
    ADOPTERS,
    ids=_IDS,
)
def test_mixing_typed_config_with_kwargs_raises(
    consumer_cls, _config_cls, typed_config, _equivalent_dict, _default
) -> None:
    """A typed config plus loose kwargs is a construction error."""
    with pytest.raises(TypeError):
        consumer_cls(typed_config, some_unexpected_kwarg=1)


@pytest.mark.parametrize(
    "consumer_cls, config_cls",
    [(a[0], a[1]) for a in ADOPTERS if a[4]],
    ids=[a[0].__name__ for a in ADOPTERS if a[4]],
)
def test_default_construction_for_all_default_configs(
    consumer_cls, config_cls
) -> None:
    """Adopters whose config has all-default fields build with no args."""
    obj = consumer_cls()
    assert isinstance(obj.config, config_cls)
    assert obj.config == config_cls()


@pytest.mark.parametrize(
    "consumer_cls, _config_cls, _typed_config, _equivalent_dict, _default",
    ADOPTERS,
    ids=_IDS,
)
def test_config_property_is_read_only(
    consumer_cls, _config_cls, _typed_config, _equivalent_dict, _default
) -> None:
    """``config`` is a read-only property — reassignment raises."""
    obj = consumer_cls(_typed_config)
    with pytest.raises(AttributeError):
        obj.config = _typed_config


# --- Per-adopter derived-state (_setup) checks ---------------------------


def test_circuit_breaker_setup_state() -> None:
    cb = CircuitBreaker.from_config({"failure_threshold": 3})
    assert cb.state.value == "closed"
    assert cb.failure_count == 0
    assert cb.config.failure_threshold == 3


def test_bulkhead_setup_state() -> None:
    bh = Bulkhead.from_config({"max_concurrent": 2})
    assert bh.semaphore._value == 2
    assert bh.metrics == {"executed": 0, "rejected": 0, "timeout": 0}


def test_error_recovery_workflow_setup_state() -> None:
    erw = ErrorRecoveryWorkflow.from_config({"primary_strategy": "retry"})
    assert erw.config.primary_strategy is RecoveryStrategy.RETRY
    assert erw._fsm is not None
    # No retry_config supplied → executor stays uninitialized.
    assert erw._retry_executor is None


def test_api_orchestrator_setup_state() -> None:
    ao = APIOrchestrator.from_config(
        {"endpoints": [{"name": "b", "url": "http://y"}]}
    )
    # Nested APIEndpoint is rebuilt as a typed instance by from_dict recursion.
    assert isinstance(ao.config.endpoints[0], APIEndpoint)
    assert ao.config.endpoints[0].name == "b"
    assert "b" in ao._circuit_breakers
    assert ao._fsm is not None


def test_database_etl_setup_state() -> None:
    etl = DatabaseETL.from_config(
        {"source_db": {"type": "memory"}, "target_db": {"type": "memory"}}
    )
    assert etl._fsm is not None
    assert etl._metrics["extracted"] == 0


def test_file_processor_setup_resolves_format_off_config() -> None:
    fp = FileProcessor(FileProcessingConfig(input_path="data.json"))
    # Auto-detection lands on the processor, leaving the frozen config intact.
    assert fp.resolved_format is FileFormat.JSON
    assert fp.config.format is None
    fp_csv = FileProcessor.from_config({"input_path": "data.csv"})
    assert fp_csv.resolved_format is FileFormat.CSV


def test_stream_context_setup_sizes_queues() -> None:
    sc = StreamContext.from_config({"buffer_size": 42})
    assert sc.status is StreamStatus.IDLE
    assert sc._input_queue.maxsize == 42
    assert sc._output_queue.maxsize == 42


def test_async_stream_context_setup_state() -> None:
    asc = AsyncStreamContext.from_config({"parallelism": 3})
    assert asc.status is StreamStatus.IDLE
    assert asc.config.parallelism == 3


# --- ResourcePool: collaborator-bearing adopter -------------------------
#
# ResourcePool carries a required ``provider`` collaborator (not config
# data) and keeps the back-compat ``ResourcePool(provider, config)``
# positional shortcut, so it is tested apart from the generic config-first
# matrix above.


def test_resource_pool_is_consumer_and_parity_holds() -> None:
    """ResourcePool is a mixin adopter; the parity guard holds.

    ``provider`` is the back-compat positional collaborator (not a config
    field), so it is passed to ``ignore_params``.
    """
    assert issubclass(ResourcePool, StructuredConfigConsumer)
    assert_structured_config_consumer(ResourcePool, ignore_params={"provider"})


def test_resource_pool_backcompat_positional() -> None:
    """``ResourcePool(provider, config)`` (the existing call shape) works."""
    provider = _FakeResourceProvider()
    pool = ResourcePool(provider, PoolConfig(min_size=0, max_size=5))
    assert pool.provider is provider
    assert isinstance(pool.config, PoolConfig)
    assert pool.config.max_size == 5


def test_resource_pool_default_and_dict_and_kwargs_config() -> None:
    """provider-positional with default, dict, and loose-kwarg config."""
    provider = _FakeResourceProvider()

    # The default case deliberately omits ``min_size=0`` (used by the sibling
    # tests to skip pool pre-fill): here we want the default ``PoolConfig()``
    # to flow through unchanged, which means ``_initialize_pool`` runs with the
    # default ``min_size=1`` and calls ``_FakeResourceProvider.acquire()``.
    # The assertion is therefore coupled to ``PoolConfig()``'s field defaults
    # by design — it pins that the no-config path yields the canonical default.
    default = ResourcePool(provider)
    assert default.config == PoolConfig()

    via_dict = ResourcePool(provider, {"min_size": 0, "max_size": 3})
    assert via_dict.config.max_size == 3

    via_kwargs = ResourcePool(provider, min_size=0, max_size=7)
    assert via_kwargs.config.max_size == 7


def test_resource_pool_from_config_delivers_provider() -> None:
    """``from_config(config, provider=...)`` threads the collaborator."""
    provider = _FakeResourceProvider()
    pool = ResourcePool.from_config(
        {"min_size": 0, "max_size": 2}, provider=provider
    )
    assert pool.provider is provider
    assert pool.config.max_size == 2


def test_resource_pool_config_is_read_only() -> None:
    """``config`` is the mixin's read-only property."""
    pool = ResourcePool(_FakeResourceProvider(), PoolConfig(min_size=0))
    with pytest.raises(AttributeError):
        pool.config = PoolConfig()


def test_resource_pool_mixing_typed_config_with_kwargs_raises() -> None:
    """A typed PoolConfig plus loose config kwargs is a construction error."""
    with pytest.raises(TypeError):
        ResourcePool(
            _FakeResourceProvider(), PoolConfig(min_size=0), max_size=9
        )
