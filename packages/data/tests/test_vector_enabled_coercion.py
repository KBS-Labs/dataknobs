"""``vector_enabled`` is read as a boolean however a configuration spelled it.

YAML and environment substitution hand a backend the string ``"false"``, and
the field was stored as given: the string is truthy, so a backend told
``vector_enabled: "false"`` enabled vectors, and a native Postgres table told
the same refused it as an instruction to create something. Every other flag on
these configs is coerced through ``SQLTableManager.coerce_bool``; this one is
declared on the shared base and was coerced nowhere.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data.backends import config as backend_config
from dataknobs_data.backends.config import PostgresDatabaseConfig, VectorBackendConfig
from dataknobs_data.backends.memory import SyncMemoryDatabase

#: What a concrete config needs beyond the flag to construct at all.
REQUIRED: dict[str, dict[str, Any]] = {"bucket": {"bucket": "b"}}


def _vector_configs() -> list[type[VectorBackendConfig]]:
    found: list[type[VectorBackendConfig]] = []
    pending = list(VectorBackendConfig.__subclasses__())
    while pending:
        cls = pending.pop()
        found.append(cls)
        pending.extend(cls.__subclasses__())
    return sorted(found, key=lambda c: c.__name__)


def _build(cls: type[VectorBackendConfig], **values: Any) -> VectorBackendConfig:
    extra = REQUIRED["bucket"] if "S3" in cls.__name__ else {}
    return cls.from_dict({**extra, **values})


def test_every_vector_config_is_swept() -> None:
    names = {cls.__name__ for cls in _vector_configs()}
    assert {"MemoryDatabaseConfig", "PostgresDatabaseConfig", "SyncS3DatabaseConfig"} <= names
    assert backend_config.FileDatabaseConfig in _vector_configs()


@pytest.mark.parametrize("cls", _vector_configs(), ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    ("given", "expected"),
    [("false", False), ("False", False), ("0", False), ("no", False), ("true", True),
     (True, True), (False, False), (None, False)],
)  # fmt: skip
def test_vector_enabled_is_a_boolean(
    cls: type[VectorBackendConfig], given: Any, expected: bool
) -> None:
    assert _build(cls, vector_enabled=given).vector_enabled is expected


def test_a_backend_told_false_as_a_string_has_no_vectors() -> None:
    assert SyncMemoryDatabase({"vector_enabled": "false"}).vector_enabled is False


NATIVE = {"layout": "native", "id_column": "id", "schema": {"fields": {"id": "string"}}}


@pytest.mark.parametrize("key", ["vector_enabled", "auto_create_table", "ensure_database"])
@pytest.mark.parametrize("given", ["false", None])
def test_a_native_table_told_false_or_nothing_is_not_refused(key: str, given: Any) -> None:
    """``null`` is what a YAML key with no value reads as; it means the default,
    which under ``layout: native`` is off.
    """
    config = PostgresDatabaseConfig.from_dict({**NATIVE, key: given})
    assert getattr(config, key) is False
