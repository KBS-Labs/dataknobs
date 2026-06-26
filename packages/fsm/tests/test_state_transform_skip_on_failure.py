"""Reproduce-first: post-failure transform skip + run_on_failure opt-out.

Once a state transform raises for a record, the execution engines skip the
transforms of every *subsequent* state (and any remaining transforms in the
failing state) so indeterminate data is not mutated or persisted — e.g. an ETL
``load`` upsert must not run on a record whose ``transform`` raised. That skip
is a general-engine behavior, so a consumer FSM with an explicit recovery /
compensation / cleanup / dead-letter state would find its transforms silently
disabled once anything upstream failed.

``run_on_failure=True`` is the per-state opt-out: such a state runs its
transforms despite a prior failure. These tests lock both halves of the
contract across the sync engine (``SimpleFSM``) and async engine
(``AsyncSimpleFSM``):

* a normal downstream state's transform is **skipped** after an upstream failure;
* a ``run_on_failure`` state's transform **still runs**;
* a second transform in the *same* state is skipped after the first raises;
* and (regression guard) a clean record runs every state's transforms.

Reproduce-first: the run_on_failure assertions FAIL before the opt-out exists
(the recovery transform is skipped along with everything else) and PASS once
``should_skip_state_transforms`` honors the flag.
"""

from __future__ import annotations

import pytest

from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    ArcConfig,
    FSMConfig,
    FunctionReference,
    NetworkConfig,
    PushArcConfig,
    StateConfig,
)
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.network import NetworkExecutor


def _boom(*_args, **_kwargs):
    """A transform that always raises, however it is invoked."""
    raise RuntimeError("transform exploded")


def _spy(name: str, calls: list[str]):
    """A transform that records that it ran and passes data through."""

    def fn(*args, **_kwargs):
        calls.append(name)
        state = args[0]
        data = getattr(state, "data", state)
        return dict(data) if isinstance(data, dict) else {}

    fn.__name__ = name
    return fn


def _recovery_config() -> dict:
    """start -> fail(boom) -> cleanup(run_on_failure, spy) -> extra(spy) -> done."""
    return {
        "name": "RecoveryFSM",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {
                        "name": "fail",
                        "functions": {"transform": {"type": "registered", "name": "boom"}},
                    },
                    {
                        "name": "cleanup",
                        "run_on_failure": True,
                        "functions": {
                            "transform": {"type": "registered", "name": "spy_cleanup"}
                        },
                    },
                    {
                        "name": "extra",
                        "functions": {
                            "transform": {"type": "registered", "name": "spy_extra"}
                        },
                    },
                    {"name": "done", "is_end": True},
                ],
                "arcs": [
                    {"from": "start", "to": "fail"},
                    {"from": "fail", "to": "cleanup"},
                    {"from": "cleanup", "to": "extra"},
                    {"from": "extra", "to": "done"},
                ],
            }
        ],
    }


def test_sync_recovery_state_runs_normal_downstream_skipped() -> None:
    calls: list[str] = []
    fsm = SimpleFSM(
        _recovery_config(),
        custom_functions={
            "boom": _boom,
            "spy_cleanup": _spy("cleanup", calls),
            "spy_extra": _spy("extra", calls),
        },
    )

    result = fsm.process({"id": "1"})

    assert result["success"] is False, "a failed record must report failure"
    assert "cleanup" in calls, (
        "a run_on_failure state's transform must run despite the prior failure"
    )
    assert "extra" not in calls, (
        "a normal downstream state's transform must be skipped after a failure"
    )


@pytest.mark.asyncio
async def test_async_recovery_state_runs_normal_downstream_skipped() -> None:
    calls: list[str] = []
    fsm = AsyncSimpleFSM(
        _recovery_config(),
        custom_functions={
            "boom": _boom,
            "spy_cleanup": _spy("cleanup", calls),
            "spy_extra": _spy("extra", calls),
        },
    )
    try:
        result = await fsm.process({"id": "1"})
    finally:
        await fsm.close()

    assert result["success"] is False
    assert "cleanup" in calls, (
        "a run_on_failure state's transform must run despite the prior failure"
    )
    assert "extra" not in calls, (
        "a normal downstream state's transform must be skipped after a failure"
    )


def _within_state_config() -> dict:
    """start -> multi(transforms=[boom, spy_second]) -> done."""
    return {
        "name": "WithinStateFSM",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {
                        "name": "multi",
                        "transforms": [
                            {"type": "registered", "name": "boom"},
                            {"type": "registered", "name": "spy_second"},
                        ],
                    },
                    {"name": "done", "is_end": True},
                ],
                "arcs": [
                    {"from": "start", "to": "multi"},
                    {"from": "multi", "to": "done"},
                ],
            }
        ],
    }


def test_sync_within_state_second_transform_skipped_after_raise() -> None:
    calls: list[str] = []
    fsm = SimpleFSM(
        _within_state_config(),
        custom_functions={"boom": _boom, "spy_second": _spy("second", calls)},
    )

    result = fsm.process({"id": "1"})

    assert result["success"] is False
    assert "second" not in calls, (
        "a second transform in the same state must be skipped after the first raised"
    )


@pytest.mark.asyncio
async def test_async_within_state_second_transform_skipped_after_raise() -> None:
    """The async engine has its own transform loop — lock the same within-state
    skip there too (the sync test alone would not catch async drift)."""
    calls: list[str] = []
    fsm = AsyncSimpleFSM(
        _within_state_config(),
        custom_functions={"boom": _boom, "spy_second": _spy("second", calls)},
    )
    try:
        result = await fsm.process({"id": "1"})
    finally:
        await fsm.close()

    assert result["success"] is False
    assert "second" not in calls, (
        "a second transform in the same state must be skipped after the first raised"
    )


def _subnetwork_failure_config() -> FSMConfig:
    """main: start -PushArc-> sub, return_state=after -> end.

    The sub-network's ``sub_start`` transform raises. ``after`` (the parent's
    return state) has a transform that records into ``context.variables`` so the
    test can assert whether it ran. A default ``PushArc`` uses COPY isolation, so
    the sub-network executes in a fresh context with its own ``failed_states``.
    """
    raise_code = (
        "def transform(data, context):\n"
        "    raise RuntimeError('sub transform exploded')\n"
    )
    after_code = (
        "def transform(data, context):\n"
        "    context.variables['after_ran'] = True\n"
        "    return data\n"
    )
    return FSMConfig(
        name="subnet_fail",
        main_network="main",
        networks=[
            NetworkConfig(
                name="main",
                states=[
                    StateConfig(
                        name="start",
                        is_start=True,
                        arcs=[
                            PushArcConfig(
                                target="after",
                                target_network="sub",
                                return_state="after",
                            )
                        ],
                    ),
                    StateConfig(
                        name="after",
                        arcs=[ArcConfig(target="end")],
                        transforms=[FunctionReference(type="inline", code=after_code)],
                    ),
                    StateConfig(name="end", is_end=True),
                ],
            ),
            NetworkConfig(
                name="sub",
                states=[
                    StateConfig(
                        name="sub_start",
                        is_start=True,
                        arcs=[ArcConfig(target="sub_end")],
                        transforms=[FunctionReference(type="inline", code=raise_code)],
                    ),
                    StateConfig(name="sub_end", is_end=True),
                ],
            ),
        ],
    )


def test_subnetwork_transform_failure_propagates_to_parent() -> None:
    """A transform failure inside an (isolated) sub-network must reach the parent.

    Reproduce-first for the isolated-sub-network data-integrity hole: a default
    ``PushArc`` uses COPY isolation, so the sub-network runs in a fresh context
    with its own ``failed_states``. Before the merge-back, a sub-network
    transform that raised was recorded only in the sub-context and dropped on
    return — the parent's ``failed_states`` stayed empty, so it finalized as a
    success and ran its downstream (return-state) transforms against
    indeterminate data: the same silent-persistence class the skip contract
    exists to prevent.

    This FAILS pre-fix (``sub_start`` absent from the parent's ``failed_states``;
    the parent's ``after`` transform ran) and PASSES once ``_handle_push_arc``
    unions the sub-network failures back into the parent before entering the
    return state. Real constructs only — a config-built FSM driven through the
    real ``NetworkExecutor``.
    """
    fsm = FSMBuilder().build(_subnetwork_failure_config())
    executor = NetworkExecutor(fsm)
    context = ExecutionContext()
    context.data = {"id": "1"}

    executor.execute_network("main", context, context.data)

    assert "sub_start" in context.failed_states, (
        "an isolated sub-network's transform failure was lost on return to the "
        "parent"
    )
    # The `after_ran` assertion is valid because, even under COPY isolation, the
    # sub-context shares the parent's `variables` dict (NetworkExecutor sets
    # sub_context.variables = context.variables), so a write from the `after`
    # transform would be visible here. Its absence proves the transform was
    # skipped, not merely that the write landed in an isolated dict.
    assert context.variables.get("after_ran") is None, (
        "the parent's downstream (return-state) transform ran against a record "
        "that failed inside the sub-network"
    )


def _clean_config() -> dict:
    """A run_on_failure state on a clean record runs normally (regression guard)."""
    cfg = _recovery_config()
    # Replace the raising transform with a passthrough spy so nothing fails.
    cfg["networks"][0]["states"][1]["functions"] = {
        "transform": {"type": "registered", "name": "spy_fail"}
    }
    return cfg


def test_sync_clean_record_runs_all_transforms() -> None:
    calls: list[str] = []
    fsm = SimpleFSM(
        _clean_config(),
        custom_functions={
            "spy_fail": _spy("fail", calls),
            "spy_cleanup": _spy("cleanup", calls),
            "spy_extra": _spy("extra", calls),
        },
    )

    result = fsm.process({"id": "1"})

    assert result["success"] is True
    assert calls == ["fail", "cleanup", "extra"], (
        "with no failure, every state's transform must run in order"
    )
