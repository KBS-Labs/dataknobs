"""Lifecycle tests for :class:`WizardFSM`.

``WizardFSM`` wraps an ``AdvancedFSM``, which allocates a daemon
event-loop thread the first time it is stepped synchronously. The wrapper
must expose the same six lifecycle members its wrapped object provides,
or that thread survives until the process exits — which is what made a
whole-suite run leak 32 of them and turned ``common``'s bridge-teardown
assertions red.
"""

from __future__ import annotations

import threading

import pytest

from dataknobs_bots.reasoning.wizard_fsm import WizardFSM
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_common.testing import DK_SYNC_BRIDGE_THREAD


def _bridge_threads() -> int:
    """Count live sync-bridge daemon threads."""
    return sum(1 for t in threading.enumerate() if t.name == DK_SYNC_BRIDGE_THREAD)


def _minimal_config(name: str = "lifecycle") -> dict:
    """Smallest wizard config that can be stepped."""
    return {
        "name": name,
        "stages": [
            {
                "name": "gather",
                "is_start": True,
                "prompt": "Tell me your name.",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
                "transitions": [
                    {"target": "done", "condition": "data.get('name')"}
                ],
            },
            {"name": "done", "is_end": True, "prompt": "All done!"},
        ],
    }


def _build(config: dict | None = None) -> WizardFSM:
    return WizardConfigLoader().load_from_dict(config or _minimal_config(), {})


# --------------------------------------------------------------------------
# The six members
# --------------------------------------------------------------------------


def test_close_releases_the_bridge_thread() -> None:
    """The reproduction: a synchronously-stepped FSM must be closeable.

    Without ``WizardFSM.close()`` the count stays at 1 for the life of the
    process and there is no supported way to bring it down.
    """
    before = _bridge_threads()
    fsm = _build()
    fsm.step({"name": "Alice"})
    assert _bridge_threads() == before + 1, "sync step should allocate a bridge"

    fsm.close()

    assert _bridge_threads() == before


def test_close_is_idempotent_and_leaves_the_fsm_usable() -> None:
    """A closed FSM is not a dead FSM.

    This is what makes an unconditional per-test teardown safe: closing
    twice is fine, and a later step lazily rebuilds the bridge rather than
    raising.
    """
    before = _bridge_threads()
    fsm = _build()
    fsm.step({"name": "Alice"})

    fsm.close()
    fsm.close()
    assert _bridge_threads() == before

    fsm.step({"name": "Bob"})
    assert _bridge_threads() == before + 1

    fsm.close()
    assert _bridge_threads() == before


async def test_aclose_releases_the_bridge_from_async() -> None:
    """The non-lossy path.

    ``aclose()`` awaits the resource manager's coroutine cleanup, which
    the synchronous ``close()`` skips.
    """
    before = _bridge_threads()
    fsm = _build()
    fsm.step({"name": "Alice"})
    assert _bridge_threads() == before + 1

    await fsm.aclose()

    assert _bridge_threads() == before


def test_sync_context_manager_closes_on_exit() -> None:
    before = _bridge_threads()
    with _build() as fsm:
        fsm.step({"name": "Alice"})
        assert _bridge_threads() == before + 1
    assert _bridge_threads() == before


def test_sync_context_manager_closes_on_exception() -> None:
    before = _bridge_threads()
    with pytest.raises(RuntimeError, match="boom"):
        with _build() as fsm:
            fsm.step({"name": "Alice"})
            raise RuntimeError("boom")
    assert _bridge_threads() == before


async def test_async_context_manager_closes_on_exit() -> None:
    before = _bridge_threads()
    async with _build() as fsm:
        fsm.step({"name": "Alice"})
        assert _bridge_threads() == before + 1
    assert _bridge_threads() == before


def test_context_manager_yields_the_fsm_itself() -> None:
    fsm = _build()
    with fsm as entered:
        assert entered is fsm
    fsm.close()


# --------------------------------------------------------------------------
# The subflow cascade
# --------------------------------------------------------------------------


def test_close_cascades_to_owned_subflows() -> None:
    """A parent's close releases its children's bridges too."""
    before = _bridge_threads()
    parent = _build()
    child = _build(_minimal_config("child"))
    parent.register_subflow("child", child)

    child.step({"name": "Alice"})
    assert _bridge_threads() == before + 1

    parent.close()

    assert _bridge_threads() == before


async def test_aclose_cascades_to_owned_subflows() -> None:
    before = _bridge_threads()
    parent = _build()
    child = _build(_minimal_config("child"))
    parent.register_subflow("child", child)
    child.step({"name": "Alice"})

    await parent.aclose()

    assert _bridge_threads() == before


def test_close_skips_subflows_registered_as_not_owned() -> None:
    """The hole the ``owns=`` parameter exists to close.

    A consumer-registered, consumer-owned subflow must survive its
    parent's teardown — the caller may still be stepping it. Remove the
    ``owns`` check from the cascade and this test fails.
    """
    before = _bridge_threads()
    parent = _build()
    borrowed = _build(_minimal_config("borrowed"))
    parent.register_subflow("borrowed", borrowed, owns=False)

    borrowed.step({"name": "Alice"})
    assert _bridge_threads() == before + 1

    parent.close()

    assert _bridge_threads() == before + 1, "a borrowed subflow was closed"

    borrowed.close()
    assert _bridge_threads() == before


async def test_aclose_skips_subflows_registered_as_not_owned() -> None:
    before = _bridge_threads()
    parent = _build()
    borrowed = _build(_minimal_config("borrowed"))
    parent.register_subflow("borrowed", borrowed, owns=False)
    borrowed.step({"name": "Alice"})

    await parent.aclose()

    assert _bridge_threads() == before + 1
    borrowed.close()


def test_loader_built_subflows_are_parent_owned() -> None:
    """Subflows the loader builds belong to the parent that wraps them.

    They arrive through the constructor's ``subflow_registry=`` argument
    rather than ``register_subflow``, so ownership has to be conferred
    there too — otherwise the cascade never fires in production, where
    ``register_subflow`` has no callers at all.
    """
    child = _build(_minimal_config("child"))
    parent = WizardFSM(
        child._fsm,
        {},
        subflow_registry={"child": child},
    )
    assert parent.subflow_names == ["child"]

    before = _bridge_threads()
    child.step({"name": "Alice"})
    assert _bridge_threads() == before + 1

    parent.close()
    assert _bridge_threads() == before


def test_one_failing_subflow_does_not_orphan_its_siblings() -> None:
    """Error isolation per child.

    Teardown is a cascade; a child that raises must not prevent the
    children after it from being released.
    """

    class ExplodingSubflow(WizardFSM):
        def close(self) -> None:
            raise RuntimeError("boom")

    before = _bridge_threads()
    parent = _build()
    exploding = ExplodingSubflow(_build(_minimal_config("bad"))._fsm, {})
    good = _build(_minimal_config("good"))

    # Registration order puts the failing child first.
    parent.register_subflow("bad", exploding)
    parent.register_subflow("good", good)

    good.step({"name": "Alice"})
    assert _bridge_threads() == before + 1

    parent.close()

    assert _bridge_threads() == before, "sibling orphaned by a failing close"


async def test_one_failing_subflow_does_not_orphan_siblings_async() -> None:
    class ExplodingSubflow(WizardFSM):
        async def aclose(self) -> None:
            raise RuntimeError("boom")

    before = _bridge_threads()
    parent = _build()
    exploding = ExplodingSubflow(_build(_minimal_config("bad"))._fsm, {})
    good = _build(_minimal_config("good"))
    parent.register_subflow("bad", exploding)
    parent.register_subflow("good", good)
    good.step({"name": "Alice"})

    await parent.aclose()

    assert _bridge_threads() == before


def test_register_subflow_defaults_to_owned() -> None:
    parent = _build()
    child = _build(_minimal_config("child"))
    parent.register_subflow("child", child)
    assert parent.get_subflow("child") is child
    assert "child" in parent._owns_subflows
    parent.close()


def test_re_registering_a_name_updates_its_ownership() -> None:
    """Ownership tracks the current registration, not the first one."""
    parent = _build()
    child = _build(_minimal_config("child"))

    parent.register_subflow("child", child, owns=True)
    assert "child" in parent._owns_subflows

    parent.register_subflow("child", child, owns=False)
    assert "child" not in parent._owns_subflows

    parent.close()
    child.close()


def test_replacing_an_owned_subflow_closes_the_one_it_replaces() -> None:
    """Reproduce-first: re-registration must not orphan the old subflow.

    ``register_subflow`` overwrote the registry entry and updated the
    ownership set, but never closed the object it displaced. An owned
    subflow that had been stepped therefore lost its only route to
    ``close()`` the moment its name was reused — its daemon thread became
    unreachable through the parent, which is precisely the defect this
    class's lifecycle exists to eliminate, reintroduced one level down.
    """
    before = _bridge_threads()
    parent = _build()
    first = _build(_minimal_config("first"))
    second = _build(_minimal_config("second"))

    parent.register_subflow("child", first)
    first.step({"name": "Alice"})
    assert _bridge_threads() == before + 1

    # Reusing the name is the parent's last chance to release `first`.
    parent.register_subflow("child", second)

    assert _bridge_threads() == before, (
        "the replaced subflow was dropped from the registry without being "
        "closed, leaking its bridge thread"
    )
    parent.close()


def test_replacing_an_unowned_subflow_leaves_it_open() -> None:
    """The ownership gate applies to replacement too.

    A subflow registered ``owns=False`` belongs to its caller, who may
    still be stepping it. Displacing it from this registry is not
    permission to tear it down.
    """
    before = _bridge_threads()
    parent = _build()
    borrowed = _build(_minimal_config("borrowed"))
    replacement = _build(_minimal_config("replacement"))

    parent.register_subflow("child", borrowed, owns=False)
    borrowed.step({"name": "Alice"})
    assert _bridge_threads() == before + 1

    parent.register_subflow("child", replacement)

    assert _bridge_threads() == before + 1, "closed a subflow it did not own"
    borrowed.close()
    parent.close()
    assert _bridge_threads() == before


def test_re_registering_the_same_object_does_not_close_it() -> None:
    """Re-registering a name to flip ownership must not close the subflow.

    ``register_subflow(name, child, owns=False)`` after an owned
    registration is the documented way to hand a subflow back to its
    caller. Closing on every replacement would make that call destroy the
    thing it is handing over.
    """
    before = _bridge_threads()
    parent = _build()
    child = _build(_minimal_config("child"))

    parent.register_subflow("child", child, owns=True)
    child.step({"name": "Alice"})
    assert _bridge_threads() == before + 1

    parent.register_subflow("child", child, owns=False)

    assert _bridge_threads() == before + 1, "closed the subflow it re-registered"
    child.close()
    parent.close()
    assert _bridge_threads() == before


def test_constructor_does_not_alias_the_callers_registry() -> None:
    """Reproduce-first: the ctor took a reference to the caller's dict.

    ``self._subflow_registry = subflow_registry or {}`` aliased the
    caller's mapping while ownership was snapshotted from it exactly once.
    A caller that kept its reference and added an entry afterwards got a
    subflow that the parent would step but never close — present in the
    registry, absent from the ownership set.
    """
    before = _bridge_threads()
    # Seeded, because ``subflow_registry or {}`` substitutes a fresh dict for
    # an *empty* one — only a non-empty registry was ever aliased, which is
    # exactly the case the loader produces.
    seed = _build(_minimal_config("seed"))
    caller_registry: dict[str, WizardFSM] = {"seed": seed}
    parent = WizardFSM(_build()._fsm, {}, subflow_registry=caller_registry)

    late = _build(_minimal_config("late"))
    caller_registry["late"] = late
    late.step({"name": "Alice"})

    assert parent.subflow_names == ["seed"], (
        "the parent's registry aliases the caller's dict, so an entry added "
        "after construction silently appears with no ownership recorded"
    )
    late.close()
    parent.close()
    assert _bridge_threads() == before
