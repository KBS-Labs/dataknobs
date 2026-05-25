"""StructuredConfig migration tests for the FSM patterns-family configs.

Covers construction parity, symmetric round-tripping, and immutability for
the error-recovery, API-orchestration, ETL, and file-processing config
classes, plus the structural complications the migration surfaced:

- the ``ErrorRecoveryConfig`` nested composition, where the five ``Optional``
  sub-config fields (including ``RetryConfig`` from ``dataknobs_common``) are
  rebuilt as typed instances by the base ``from_dict`` recursion with **no**
  ``_normalize_dict`` override;
- the ``APIOrchestrationConfig`` nested ``list[APIEndpoint]`` round-trip,
  which required migrating ``APIEndpoint`` to ``StructuredConfig`` too;
- ``CompensationConfig``'s now-defaulted ``compensation_actions``; and
- the ``FileProcessor`` format auto-detection now resolving onto the
  processor rather than mutating the (frozen) config.

Real constructs only — no mocks.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile

import pytest
from dataknobs_common.retry import RetryConfig
from dataknobs_common.structured_config import StructuredConfig
from dataknobs_common.testing import assert_structured_config_roundtrip

from dataknobs_fsm.patterns.api_orchestration import (
    APIEndpoint,
    APIOrchestrationConfig,
    OrchestrationMode,
)
from dataknobs_fsm.patterns.error_recovery import (
    BulkheadConfig,
    CircuitBreakerConfig,
    CompensationConfig,
    ErrorRecoveryConfig,
    FallbackConfig,
    RecoveryStrategy,
)
from dataknobs_fsm.patterns.etl import ETLConfig, ETLMode
from dataknobs_fsm.patterns.file_processing import (
    FileFormat,
    FileProcessingConfig,
    FileProcessor,
)

ALL_PATTERN_CONFIGS = [
    CircuitBreakerConfig,
    FallbackConfig,
    CompensationConfig,
    BulkheadConfig,
    ErrorRecoveryConfig,
    APIEndpoint,
    APIOrchestrationConfig,
    ETLConfig,
    FileProcessingConfig,
]


class TestPatternsConfigsAreStructured:
    """All patterns-family configs are frozen ``StructuredConfig`` subclasses."""

    @pytest.mark.parametrize("config_cls", ALL_PATTERN_CONFIGS)
    def test_is_structured_config(self, config_cls):
        assert issubclass(config_cls, StructuredConfig)

    @pytest.mark.parametrize("config_cls", ALL_PATTERN_CONFIGS)
    def test_is_frozen(self, config_cls):
        assert config_cls.__dataclass_params__.frozen


class TestErrorRecoveryConfigStructured:
    """Error-recovery family: parity, round-trip, nesting, immutability."""

    def test_circuit_breaker_construction_parity(self):
        assert CircuitBreakerConfig.from_dict(
            {"failure_threshold": 9, "timeout": 30.0}
        ) == CircuitBreakerConfig(failure_threshold=9, timeout=30.0)

    def test_fallback_construction_parity(self):
        assert FallbackConfig.from_dict(
            {"use_cache": True, "cache_ttl": 120.0}
        ) == FallbackConfig(use_cache=True, cache_ttl=120.0)

    def test_bulkhead_construction_parity(self):
        assert BulkheadConfig.from_dict(
            {"max_concurrent": 3, "use_thread_pool": True}
        ) == BulkheadConfig(max_concurrent=3, use_thread_pool=True)

    def test_circuit_breaker_roundtrip_default(self):
        assert_structured_config_roundtrip(CircuitBreakerConfig())

    def test_circuit_breaker_roundtrip_with_callbacks(self):
        # Callable fields round-trip by identity (deepcopy treats functions
        # as atomic), so equality holds.
        def on_open() -> None:
            pass

        cfg = CircuitBreakerConfig(on_open=on_open)
        restored = CircuitBreakerConfig.from_dict(cfg.to_dict())
        assert restored == cfg
        assert restored.on_open is on_open

    def test_fallback_roundtrip_with_exception_types(self):
        cfg = FallbackConfig(fallback_on_exceptions=[ValueError, KeyError])
        restored = FallbackConfig.from_dict(cfg.to_dict())
        assert restored == cfg
        assert restored.fallback_on_exceptions == [ValueError, KeyError]

    def test_compensation_defaults_to_empty_actions(self):
        # D4: compensation_actions is no longer a required field — from_dict
        # with no actions yields an empty list rather than raising.
        cfg = CompensationConfig.from_dict({})
        assert cfg.compensation_actions == []
        assert cfg.save_state is True

    def test_compensation_roundtrip_with_actions(self):
        def undo(_state) -> None:
            pass

        cfg = CompensationConfig(compensation_actions=[undo], use_sagas=True)
        restored = CompensationConfig.from_dict(cfg.to_dict())
        assert restored == cfg
        assert restored.compensation_actions == [undo]

    def test_error_recovery_nested_composition_typed(self):
        # D2: the five Optional sub-config fields are rebuilt as typed
        # instances by the base from_dict recursion — no _normalize_dict.
        cfg = ErrorRecoveryConfig.from_dict(
            {
                "primary_strategy": RecoveryStrategy.RETRY,
                "retry_config": {"max_attempts": 7, "initial_delay": 0.25},
                "circuit_breaker_config": {"failure_threshold": 9},
                "fallback_config": {"use_cache": True},
                "compensation_config": {"save_state": False},
                "bulkhead_config": {"max_concurrent": 3},
            }
        )
        assert isinstance(cfg.retry_config, RetryConfig)
        assert cfg.retry_config.max_attempts == 7
        assert cfg.retry_config.initial_delay == 0.25
        assert isinstance(cfg.circuit_breaker_config, CircuitBreakerConfig)
        assert cfg.circuit_breaker_config.failure_threshold == 9
        assert isinstance(cfg.fallback_config, FallbackConfig)
        assert cfg.fallback_config.use_cache is True
        assert isinstance(cfg.compensation_config, CompensationConfig)
        assert cfg.compensation_config.save_state is False
        assert isinstance(cfg.bulkhead_config, BulkheadConfig)
        assert cfg.bulkhead_config.max_concurrent == 3

    def test_error_recovery_nested_roundtrip(self):
        cfg = ErrorRecoveryConfig(
            primary_strategy=RecoveryStrategy.CIRCUIT_BREAKER,
            secondary_strategies=[RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            retry_config=RetryConfig(max_attempts=4),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=8),
        )
        restored = ErrorRecoveryConfig.from_dict(cfg.to_dict())
        assert restored == cfg
        assert isinstance(restored.retry_config, RetryConfig)
        assert restored.secondary_strategies == [
            RecoveryStrategy.RETRY,
            RecoveryStrategy.FALLBACK,
        ]

    def test_error_recovery_frozen(self):
        cfg = ErrorRecoveryConfig(primary_strategy=RecoveryStrategy.RETRY)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.max_total_attempts = 99  # type: ignore[misc]


class TestAPIOrchestrationConfigStructured:
    """API-orchestration family: endpoint parity and nested round-trip."""

    def test_endpoint_construction_parity(self):
        assert APIEndpoint.from_dict(
            {"name": "search", "url": "http://x", "method": "POST"}
        ) == APIEndpoint(name="search", url="http://x", method="POST")

    def test_endpoint_roundtrip_with_callables(self):
        def parse(_data):
            return _data

        cfg = APIEndpoint(name="a", url="http://x", response_parser=parse)
        restored = APIEndpoint.from_dict(cfg.to_dict())
        assert restored == cfg
        assert restored.response_parser is parse

    def test_orchestration_nested_endpoints_typed(self):
        # D5: list[APIEndpoint] is rebuilt as typed APIEndpoint instances.
        cfg = APIOrchestrationConfig.from_dict(
            {
                "endpoints": [
                    {"name": "a", "url": "http://x"},
                    {"name": "b", "url": "http://y", "method": "POST"},
                ],
                "mode": OrchestrationMode.PARALLEL,
            }
        )
        assert all(isinstance(e, APIEndpoint) for e in cfg.endpoints)
        assert cfg.endpoints[0].name == "a"
        assert cfg.endpoints[1].method == "POST"
        assert cfg.mode is OrchestrationMode.PARALLEL

    def test_orchestration_nested_roundtrip(self):
        cfg = APIOrchestrationConfig(
            endpoints=[
                APIEndpoint(name="a", url="http://x"),
                APIEndpoint(name="b", url="http://y", retry_count=5),
            ],
            mode=OrchestrationMode.PIPELINE,
        )
        restored = APIOrchestrationConfig.from_dict(cfg.to_dict())
        assert restored == cfg
        assert all(isinstance(e, APIEndpoint) for e in restored.endpoints)
        assert restored.endpoints[1].retry_count == 5

    def test_orchestration_frozen(self):
        cfg = APIOrchestrationConfig(endpoints=[APIEndpoint(name="a", url="http://x")])
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.max_concurrent = 99  # type: ignore[misc]


class TestETLConfigStructured:
    """ETLConfig: parity, Enum round-trip, immutability."""

    def test_construction_parity(self):
        assert ETLConfig.from_dict(
            {"source_db": {"backend": "memory"}, "target_db": {"backend": "memory"}}
        ) == ETLConfig(
            source_db={"backend": "memory"}, target_db={"backend": "memory"}
        )

    def test_roundtrip_with_enum_mode(self):
        cfg = ETLConfig(
            source_db={"backend": "memory"},
            target_db={"backend": "sqlite"},
            mode=ETLMode.INCREMENTAL,
            key_columns=["id"],
        )
        restored = ETLConfig.from_dict(cfg.to_dict())
        assert restored == cfg
        assert restored.mode is ETLMode.INCREMENTAL

    def test_frozen(self):
        cfg = ETLConfig(source_db={}, target_db={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.batch_size = 99  # type: ignore[misc]


class TestFileProcessingConfigStructured:
    """FileProcessingConfig: parity, round-trip, and the D6 detection fix."""

    def test_construction_parity(self):
        assert FileProcessingConfig.from_dict(
            {"input_path": "in.json", "chunk_size": 50}
        ) == FileProcessingConfig(input_path="in.json", chunk_size=50)

    def test_roundtrip_with_format_enum(self):
        cfg = FileProcessingConfig(
            input_path="in.csv",
            format=FileFormat.CSV,
            output_format=FileFormat.JSON,
        )
        restored = FileProcessingConfig.from_dict(cfg.to_dict())
        assert restored == cfg
        assert restored.format is FileFormat.CSV
        assert restored.output_format is FileFormat.JSON

    def test_frozen(self):
        cfg = FileProcessingConfig(input_path="in.json")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.chunk_size = 99  # type: ignore[misc]

    def test_processor_resolves_format_without_mutating_config(self):
        # D6 (reproduce-style): a FileProcessor over a .json input with
        # format unset must auto-detect JSON onto the processor and NOT
        # write back to the now-frozen config (which would raise
        # FrozenInstanceError pre-fix).
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"a": 1}')

            config = FileProcessingConfig(input_path=path)  # format=None
            processor = FileProcessor(config)

            # Config is untouched (stays "auto-detect").
            assert config.format is None
            assert config.output_format is None
            # Resolved format lives on the processor.
            assert processor._format == FileFormat.JSON
            assert processor._output_format == FileFormat.JSON

    def test_processor_respects_explicit_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.bin")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("name,value\na,1\n")

            config = FileProcessingConfig(input_path=path, format=FileFormat.CSV)
            processor = FileProcessor(config)

            assert processor._format == FileFormat.CSV
            assert processor._output_format == FileFormat.CSV
