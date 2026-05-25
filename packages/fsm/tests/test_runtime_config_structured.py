"""StructuredConfig migration tests for the FSM runtime-config family.

Covers construction parity, symmetric round-tripping, and immutability for
the resources/IO/storage/streaming/functions config classes:

- ``PoolConfig`` (``resources/pool.py``),
- ``IOConfig`` (``io/base.py``) — including its callable ``error_handler``
  and ``IOMode``/``IOFormat`` Enum fields, which round-trip by identity;
- ``StreamConfig`` (``streaming/core.py``),
- ``ResourceConfig`` (``functions/base.py``),
- ``StorageConfig`` (``storage/base.py``) — converted from a plain class to a
  frozen dataclass, with its ``StorageBackend`` field and the Enum-keyed
  ``mode_specific_config`` mapping; plus the root-cause refactor where the
  memory/file storage backends build a local working copy of
  ``connection_params`` instead of mutating the (now frozen) caller config.

Real constructs only — no mocks.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from dataknobs_common.structured_config import StructuredConfig
from dataknobs_common.testing import assert_structured_config_roundtrip

from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.functions.base import ResourceConfig
from dataknobs_fsm.io.base import IOConfig, IOFormat, IOMode
from dataknobs_fsm.resources.pool import PoolConfig
from dataknobs_fsm.storage.base import StorageBackend, StorageConfig
from dataknobs_fsm.storage.file import FileStorage
from dataknobs_fsm.storage.memory import InMemoryStorage
from dataknobs_fsm.streaming.core import StreamConfig

ALL_RUNTIME_CONFIGS = [
    PoolConfig,
    IOConfig,
    StreamConfig,
    ResourceConfig,
    StorageConfig,
]

# One representative, callable-free instance per config class for the uniform
# round-trip assertion. Classes with required fields (``IOConfig``,
# ``ResourceConfig``) get their minimal valid args.
ROUNDTRIP_INSTANCES = [
    PoolConfig(),
    IOConfig(mode=IOMode.READ, format=IOFormat.JSON, source="data.json"),
    StreamConfig(),
    ResourceConfig(name="db", type="postgres", connection_params={"host": "x"}),
    StorageConfig(),
]


class TestRuntimeConfigsAreStructured:
    """Every runtime config is a frozen ``StructuredConfig`` subclass."""

    @pytest.mark.parametrize("config_cls", ALL_RUNTIME_CONFIGS)
    def test_is_structured_config(self, config_cls: type) -> None:
        assert issubclass(config_cls, StructuredConfig)

    @pytest.mark.parametrize("config_cls", ALL_RUNTIME_CONFIGS)
    def test_is_frozen_dataclass(self, config_cls: type) -> None:
        assert dataclasses.is_dataclass(config_cls)
        assert config_cls.__dataclass_params__.frozen  # type: ignore[attr-defined]

    @pytest.mark.parametrize("config", ROUNDTRIP_INSTANCES)
    def test_roundtrip(self, config: StructuredConfig) -> None:
        assert_structured_config_roundtrip(config)


class TestPoolConfigStructured:
    """``PoolConfig`` parity, round-trip, immutability."""

    def test_from_dict_parity(self) -> None:
        cfg = PoolConfig(min_size=2, max_size=20, acquire_timeout=15.0)
        loaded = PoolConfig.from_dict(
            {"min_size": 2, "max_size": 20, "acquire_timeout": 15.0}
        )
        assert loaded == cfg

    def test_defaults_roundtrip(self) -> None:
        assert PoolConfig.from_dict({}) == PoolConfig()

    def test_frozen(self) -> None:
        cfg = PoolConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.max_size = 99  # type: ignore[misc]


class TestIOConfigStructured:
    """``IOConfig`` Enum + callable + ``options`` round-trip."""

    def test_enum_and_options_roundtrip(self) -> None:
        cfg = IOConfig(
            mode=IOMode.WRITE,
            format=IOFormat.CSV,
            source="out.csv",
            delimiter=";",
            options={"quoting": "minimal"},
        )
        loaded = IOConfig.from_dict(cfg.to_dict())
        assert loaded == cfg
        assert loaded.mode is IOMode.WRITE
        assert loaded.format is IOFormat.CSV
        assert loaded.options == {"quoting": "minimal"}

    def test_callable_error_handler_roundtrips_by_identity(self) -> None:
        def handler(exc: Exception, item: object) -> None:
            return None

        cfg = IOConfig(
            mode=IOMode.READ,
            format=IOFormat.JSON,
            source="in.json",
            error_handler=handler,
        )
        loaded = IOConfig.from_dict(cfg.to_dict())
        assert loaded.error_handler is handler
        assert loaded == cfg

    def test_string_enum_coercion_from_load_shape(self) -> None:
        # The YAML/JSON load shape supplies the Enum fields as raw strings;
        # ``from_dict`` coerces them to the member.
        loaded = IOConfig.from_dict(
            {"mode": "read", "format": "json", "source": "in.json"}
        )
        assert loaded.mode is IOMode.READ
        assert loaded.format is IOFormat.JSON

    def test_json_roundtrip(self) -> None:
        # No callable fields here, so the full JSON round-trip holds via
        # ``to_json_dict`` (plain ``Enum`` members render as their values).
        cfg = IOConfig(
            mode=IOMode.STREAM,
            format=IOFormat.PARQUET,
            source="s3://bucket/key",
            batch_size=256,
        )
        loaded = IOConfig.from_dict(json.loads(json.dumps(cfg.to_json_dict())))
        assert loaded == cfg

    def test_independent_options_default(self) -> None:
        a = IOConfig(mode=IOMode.READ, format=IOFormat.JSON, source="x")
        b = IOConfig(mode=IOMode.READ, format=IOFormat.JSON, source="y")
        assert a.options is not b.options

    def test_frozen(self) -> None:
        cfg = IOConfig(mode=IOMode.READ, format=IOFormat.JSON, source="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.batch_size = 1  # type: ignore[misc]


class TestStreamConfigStructured:
    """FSM ``StreamConfig`` parity, round-trip, immutability."""

    def test_from_dict_parity(self) -> None:
        cfg = StreamConfig(chunk_size=500, parallelism=4, retry_on_error=False)
        loaded = StreamConfig.from_dict(
            {"chunk_size": 500, "parallelism": 4, "retry_on_error": False}
        )
        assert loaded == cfg

    def test_defaults_roundtrip(self) -> None:
        assert StreamConfig.from_dict({}) == StreamConfig()

    def test_frozen(self) -> None:
        cfg = StreamConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.chunk_size = 1  # type: ignore[misc]


class TestResourceConfigStructured:
    """``ResourceConfig`` parity, round-trip, immutability."""

    def test_from_dict_parity(self) -> None:
        cfg = ResourceConfig(
            name="cache",
            type="redis",
            connection_params={"host": "localhost", "port": 6379},
            pool_size=5,
            retry_policy={"max_attempts": 3},
        )
        loaded = ResourceConfig.from_dict(cfg.to_dict())
        assert loaded == cfg
        assert loaded.retry_policy == {"max_attempts": 3}

    def test_frozen(self) -> None:
        cfg = ResourceConfig(name="db", type="pg", connection_params={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.pool_size = 1  # type: ignore[misc]


class TestStorageConfigStructured:
    """``StorageConfig`` D7: dataclass conversion + Enum round-trip + frozen."""

    def test_from_dict_parity(self) -> None:
        cfg = StorageConfig(
            backend=StorageBackend.FILE,
            connection_params={"path": "/tmp/hist"},
            compression=True,
            batch_size=50,
        )
        loaded = StorageConfig.from_dict(cfg.to_dict())
        assert loaded == cfg
        assert loaded.backend is StorageBackend.FILE

    def test_string_backend_coercion_from_load_shape(self) -> None:
        loaded = StorageConfig.from_dict({"backend": "file"})
        assert loaded.backend is StorageBackend.FILE

    def test_json_roundtrip_scalar_fields(self) -> None:
        # With no Enum-keyed mapping, the full JSON round-trip holds: the
        # ``StorageBackend`` member renders to its value and coerces back.
        cfg = StorageConfig(
            backend=StorageBackend.SQLITE,
            connection_params={"database": "/tmp/h.db"},
            compression=True,
            batch_size=25,
        )
        loaded = StorageConfig.from_dict(json.loads(json.dumps(cfg.to_json_dict())))
        assert loaded == cfg

    def test_mode_specific_config_enum_keyed_inprocess_roundtrip(self) -> None:
        # ``mode_specific_config`` is keyed by the ``DataHandlingMode`` Enum.
        # The base coerces dict *values*, not *keys*, so the in-process
        # ``to_dict``/``from_dict`` round-trip (which preserves the Enum-member
        # keys) is the supported path for this field — a JSON round-trip would
        # stringify the keys without coercing them back.
        cfg = StorageConfig(
            backend=StorageBackend.MEMORY,
            mode_specific_config={DataHandlingMode.COPY: {"deep": True}},
        )
        loaded = StorageConfig.from_dict(cfg.to_dict())
        assert loaded == cfg
        assert loaded.get_mode_config(DataHandlingMode.COPY) == {"deep": True}

    def test_mode_specific_config_json_roundtrip_fails_closed(self) -> None:
        # The Enum-keyed ``mode_specific_config`` is in-process-round-trip
        # only: ``to_json_dict``/``jsonify`` normalises dict *values* but not
        # *keys*, and ``from_dict`` coerces neither. Rather than silently
        # corrupting the config, a JSON round-trip fails closed — ``json.dumps``
        # raises ``TypeError`` on the un-stringified ``DataHandlingMode`` keys.
        # This pins that loud failure so the documented limitation cannot
        # regress into silent data loss (e.g. string keys that
        # ``get_mode_config`` would never match).
        cfg = StorageConfig(
            backend=StorageBackend.MEMORY,
            mode_specific_config={DataHandlingMode.COPY: {"deep": True}},
        )
        with pytest.raises(TypeError):
            json.dumps(cfg.to_json_dict())

    def test_defaults_have_independent_dicts(self) -> None:
        a = StorageConfig()
        b = StorageConfig()
        assert a.connection_params is not b.connection_params
        assert a.mode_specific_config is not b.mode_specific_config

    def test_frozen(self) -> None:
        cfg = StorageConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.backend = StorageBackend.FILE  # type: ignore[misc]


class TestStorageBackendsDoNotMutateConfig:
    """D7 step 2: storage backends build a local working copy of the params.

    Before the frozen flip, ``InMemoryStorage`` / ``FileStorage`` mutated the
    caller's ``StorageConfig.connection_params`` in place. Under a frozen
    config that would raise ``FrozenInstanceError``; the root-cause fix builds
    a local ``dict`` and reconstructs the config via ``dataclasses.replace``,
    leaving the caller's config untouched while still applying backend defaults
    to the instance the storage actually uses.
    """

    def test_memory_backend_leaves_caller_config_untouched(self) -> None:
        cfg = StorageConfig(backend=StorageBackend.MEMORY)
        storage = InMemoryStorage(cfg)

        # Caller's config is unchanged (no backend defaults leaked in).
        assert cfg.connection_params == {}
        # The storage operates on a distinct config carrying the defaults.
        assert storage.config is not cfg
        assert storage.config.connection_params.get("max_size") == 1000
        assert storage.config.connection_params.get("enable_indexing") is True

    def test_file_backend_leaves_caller_config_untouched(self) -> None:
        cfg = StorageConfig(backend=StorageBackend.FILE, compression=True)
        storage = FileStorage(cfg)

        assert cfg.connection_params == {}
        assert storage.config is not cfg
        assert storage.config.connection_params.get("path") == "./fsm_history"
        assert storage.config.connection_params.get("format") == "json"
        assert storage.config.connection_params.get("compression") == "gzip"

    def test_file_backend_respects_caller_supplied_params(self) -> None:
        cfg = StorageConfig(
            backend=StorageBackend.FILE,
            connection_params={"path": "/custom/path", "format": "yaml"},
        )
        storage = FileStorage(cfg)

        # Caller-supplied values win over the defaults; caller config intact.
        assert storage.config.connection_params["path"] == "/custom/path"
        assert storage.config.connection_params["format"] == "yaml"
        assert cfg.connection_params == {"path": "/custom/path", "format": "yaml"}
