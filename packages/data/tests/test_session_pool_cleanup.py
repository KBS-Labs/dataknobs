"""Guards for the session-end connection-pool cleanup in ``conftest.py``.

That cleanup reaches each backend's module-level pool manager by name. A name
that did not resolve used to be swallowed as though the backend's optional
driver were simply absent, so a target could sit dead indefinitely while the
suite reported nothing about it. These tests pin the distinction between the
two cases and check every registered target against the installed tree.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from dataknobs_data.pooling import BasePoolConfig


@dataclass(frozen=True)
class _NamedPoolConfig(BasePoolConfig):
    """A real ``BasePoolConfig`` keyed on a name, so two pools stay distinct."""

    name: str

    def to_connection_string(self) -> str:
        return f"test://{self.name}"

    def to_hash_key(self) -> tuple:
        return (self.name,)


class _RecordingPool:
    """A real pool object that records having been closed."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _conftest(request: pytest.FixtureRequest) -> ModuleType:
    """Return the ``conftest`` module pytest actually loaded for this package.

    Reached through the plugin manager rather than imported by name: the suite
    runs under ``--import-mode=importlib``, which puts neither the test
    directory nor the repository root on ``sys.path``. Matching the exact path
    also keeps the workspace-root ``conftest.py`` out of the way when the run
    spans more than one package.
    """
    wanted = Path(__file__).parent / "conftest.py"
    for plugin in request.config.pluginmanager.get_plugins():
        path = getattr(plugin, "__file__", None)
        if path and Path(path) == wanted:
            return cast("ModuleType", plugin)
    raise AssertionError(f"{wanted} is not registered as a pytest plugin")


def test_every_session_cleanup_target_resolves(request: pytest.FixtureRequest) -> None:
    """Every registered manager resolves, or is skipped only for an absent driver.

    This is the guard the defect needed: a target naming a module that does not
    exist reports here instead of being indistinguishable from a backend whose
    optional driver is not installed.
    """
    conftest = _conftest(request)
    unreachable = []
    for module_path, attribute in conftest._POOL_MANAGERS:
        try:
            conftest._load_pool_manager(module_path, attribute)
        except (AttributeError, ModuleNotFoundError) as exc:
            unreachable.append(f"{module_path}.{attribute} — {type(exc).__name__}: {exc}")

    assert not unreachable, (
        "session-end cleanup names targets that do not exist, so the pools they "
        "hold are never closed: " + "; ".join(unreachable)
    )


def test_a_missing_backend_module_is_not_mistaken_for_a_missing_driver(
    request: pytest.FixtureRequest,
) -> None:
    """A wrong module path raises rather than reading as an uninstalled driver."""
    conftest = _conftest(request)
    with pytest.raises(ModuleNotFoundError):
        conftest._load_pool_manager("dataknobs_data.backends.no_such_backend", "_pool_manager")


def test_a_missing_driver_is_skipped(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend whose optional driver is absent yields ``None``, not an error.

    Uses a real module on ``sys.path`` importing a package that is genuinely
    not installed, so the branch is exercised through a real ``ImportError``
    rather than a simulated one.
    """
    conftest = _conftest(request)
    (tmp_path / "dk_backend_with_absent_driver.py").write_text(
        "import a_driver_that_is_not_installed\n\n_pool_manager = object()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    assert conftest._load_pool_manager("dk_backend_with_absent_driver", "_pool_manager") is None


def test_a_renamed_manager_attribute_is_not_swallowed(
    request: pytest.FixtureRequest,
) -> None:
    """A module that imports fine but lacks the named attribute raises.

    Renaming a manager is the same class of drift as renaming its module, and
    the cleanup must not quietly skip the backend when it happens.
    """
    conftest = _conftest(request)
    with pytest.raises(AttributeError):
        conftest._load_pool_manager("json", "_no_such_manager")


async def test_close_pool_managers_closes_every_reachable_manager(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loop closes each reachable manager and steps past an absent driver.

    Points the table at purpose-built managers rather than the live singletons:
    ``close_all`` is a force teardown that ignores holders, so running it over
    the real ones mid-session would tear down pools other tests still hold. The
    managers and pools are real, so the loop drives the production close path.
    """
    conftest = _conftest(request)

    (tmp_path / "dk_pool_targets.py").write_text(
        "from dataknobs_data.pooling import ConnectionPoolManager\n\n"
        "first = ConnectionPoolManager()\n"
        "second = ConnectionPoolManager()\n"
    )
    (tmp_path / "dk_pool_target_without_driver.py").write_text(
        "import a_driver_that_is_not_installed\n\nthird = object()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    targets = importlib.import_module("dk_pool_targets")
    pools = []
    for manager, name in ((targets.first, "one"), (targets.second, "two")):
        pool = _RecordingPool()
        pools.append(pool)

        async def _create(_config: BasePoolConfig, _pool: _RecordingPool = pool) -> Any:
            return _pool

        await manager.get_pool(_NamedPoolConfig(name), _create)

    assert targets.first.get_pool_count() == 1
    assert targets.second.get_pool_count() == 1

    # Both kinds of unreachable entry sit between the two live ones — a wrong
    # module path, which is logged, and an absent driver, which is skipped
    # silently — so a pass proves the loop steps over each rather than
    # stopping at it.
    monkeypatch.setattr(
        conftest,
        "_POOL_MANAGERS",
        (
            ("dk_pool_targets", "first"),
            ("dk_pool_targets.no_such_module", "fourth"),
            ("dk_pool_target_without_driver", "third"),
            ("dk_pool_targets", "second"),
        ),
    )

    await conftest._close_pool_managers()

    assert [pool.closed for pool in pools] == [True, True]
    assert targets.first.get_pool_count() == 0
    assert targets.second.get_pool_count() == 0
