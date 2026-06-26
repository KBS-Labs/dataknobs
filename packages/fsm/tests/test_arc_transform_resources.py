"""Reproduce-first tests for arc resource injection + function-context parity.

Companion to ``test_state_transform_resources.py`` (the state path). These
cover the *arc* path: an arc may declare ``resources`` and its transform AND
condition must receive them through a ``FunctionContext`` — on both the async
and sync engines — plus the dual-access (name / role) accessors and the
``transform_context_factory`` honoring on the async engine.

Each test is written to FAIL against the pre-injection engine for a specific
reason (documented inline) and PASS once arc resource injection + the shared
function-context builder land. Real constructs only (file/memory
``AsyncDatabase``, real FSM builds — no mocks); persistence is proven by
reopening the target through a fresh ``AsyncDatabase.from_backend``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataknobs_common.testing import assert_no_blocking
from dataknobs_data import AsyncDatabase, Record
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.core.context_factory import ContextFactory
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.execution.engine import ExecutionEngine
from dataknobs_fsm.functions.base import FunctionContext, TransformError
from dataknobs_fsm.functions.library.database import DatabaseUpsert, _require_resource


@pytest.fixture(scope="module", autouse=True)
def _warm_async_backend_registry() -> None:
    """Initialize the lazy async-backend registry outside any detector block.

    The first ``AsyncDatabase.from_backend`` call in a process triggers
    ``_register_async_backends``, which imports every async backend — including
    duckdb, whose import reads its version metadata file synchronously. That
    one-shot, setup-time read is acceptable, but if it first runs *inside* an
    ``assert_no_blocking()`` block it trips the detector. Warming the registry
    here mirrors what the rest of the suite gets for free.
    """
    import asyncio

    async def _go() -> None:
        db = await AsyncDatabase.from_backend("memory", {"type": "memory"})
        await db.close()

    asyncio.run(_go())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _arc_upsert_fsm(
    target_cfg: dict[str, Any],
    *,
    declare_arc_resources: bool = True,
    transform_name: str = "load",
) -> AsyncSimpleFSM:
    """Build an FSM whose *arc* (not state) upserts the record via a resource.

    The ``loaded`` arc carries the transform and (optionally) declares the
    ``target_db`` resource — exercising the arc-transform injection path.
    """
    arc: dict[str, Any] = {
        "target": "done",
        "transform": {"type": "registered", "name": transform_name},
        "metadata": {"name": "loaded"},
    }
    if declare_arc_resources:
        arc["resources"] = ["target_db"]
    config = {
        "name": "arc_upsert",
        "data_mode": DataHandlingMode.COPY.value,
        "resources": [
            {"name": "target_db", "type": "async_database", "config": target_cfg},
        ],
        "states": [
            {"name": "start", "is_start": True, "arcs": [arc]},
            {"name": "done", "is_end": True},
        ],
    }
    return AsyncSimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={
            transform_name: DatabaseUpsert(
                resource_name="target_db",
                table="rows",
                key_columns=["id"],
            )
        },
    )


def _find_arc(fsm: Any, source_state: str, arc_name: str = "loaded") -> Any:
    """Return the named outgoing arc from ``source_state`` in the built FSM."""
    for network in fsm.networks.values():
        state = network.states.get(source_state)
        if not state:
            continue
        for arc in getattr(state, "outgoing_arcs", []):
            if arc.metadata.get("name") == arc_name:
                return arc
    raise AssertionError(f"arc {arc_name!r} from {source_state!r} not found")


async def _read_row(target_cfg: dict[str, Any], record_id: str) -> Any:
    """Reopen the target backend and read a row back, independently."""
    db = await AsyncDatabase.from_backend("file", target_cfg)
    try:
        return await db.read(record_id)
    finally:
        await db.close()


# --------------------------------------------------------------------------- #
# 1-2: Arc transform persists via its resource (name-based) + no blocking
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_arc_transform_persists_row(tmp_path: Path) -> None:
    """An arc-transform DatabaseUpsert must persist via its declared resource.

    Fails today: the async arc path calls ``transform_func(data, context)`` on
    the raw ExecutionContext — an ``ITransformFunction`` is not callable and
    never reaches its resource, so the row is never written.
    """
    target = {"type": "file", "path": str(tmp_path / "target.json")}
    fsm = _arc_upsert_fsm(target)
    try:
        result = await fsm.process({"id": "1", "name": "Alice"})
        assert result["success"], f"FSM did not complete cleanly: {result}"
    finally:
        await fsm.close()

    record = await _read_row(target, "1")
    assert record is not None, (
        "arc-transform DatabaseUpsert did not persist — its declared "
        "'target_db' resource was never injected into FunctionContext.resources"
    )
    assert record.to_dict().get("name") == "Alice"


@pytest.mark.asyncio
async def test_arc_transform_does_not_block(tmp_path: Path) -> None:
    """The resource-bearing arc transform must not block the event loop."""
    target = {"type": "file", "path": str(tmp_path / "target.json")}
    fsm = _arc_upsert_fsm(target)
    try:
        with assert_no_blocking():
            result = await fsm.process({"id": "1", "name": "Alice"})
        assert result["success"], f"FSM did not complete cleanly: {result}"
    finally:
        await fsm.close()


# --------------------------------------------------------------------------- #
# 3: Arc condition reads its resource (async)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_arc_condition_reads_resource(tmp_path: Path) -> None:
    """A resource-aware async arc condition routes on a resources lookup.

    Fails today: ``_evaluate_arc`` passes the raw ExecutionContext, so the
    predicate reads the empty bookkeeping dict and the gated arc is skipped.
    """

    def gate_open(_data: Any, ctx: Any) -> bool:
        # Resource-aware predicate: route only when the gate resource is present.
        return ctx.resources.get("gate") is not None

    config = {
        "name": "gated",
        "data_mode": DataHandlingMode.COPY.value,
        "resources": [
            {
                "name": "gate",
                "type": "async_database",
                "config": {"type": "file", "path": str(tmp_path / "gate.json")},
            },
        ],
        "states": [
            {
                "name": "start",
                "is_start": True,
                "arcs": [
                    {
                        "target": "open",
                        "resources": ["gate"],
                        "condition": {"type": "registered", "name": "gate_open"},
                        "priority": 10,
                        "metadata": {"name": "to_open"},
                    },
                    {"target": "closed", "metadata": {"name": "to_closed"}},
                ],
            },
            {"name": "open", "is_end": True},
            {"name": "closed", "is_end": True},
        ],
    }
    fsm = AsyncSimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={"gate_open": gate_open},
    )
    try:
        result = await fsm.process({"id": "1"})
    finally:
        await fsm.close()

    assert result["success"], f"FSM did not complete cleanly: {result}"
    assert result["final_state"] == "open", (
        "resource-aware arc condition did not see its 'gate' resource — it was "
        f"not injected into the condition's FunctionContext (got {result['final_state']!r})"
    )


# --------------------------------------------------------------------------- #
# 4: Resourceless arc parity — FunctionContext built unconditionally
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_resourceless_arc_still_builds_function_context(tmp_path: Path) -> None:
    """A resourceless arc transform still receives a FunctionContext.

    The factory is honored and ``resources == {}``. Fails today: the async arc
    path never builds a FunctionContext, so the factory is never invoked.
    """
    captured: list[Any] = []

    def factory(fc: FunctionContext) -> FunctionContext:
        captured.append(fc)
        return fc

    def passthrough(data: Any, _ctx: Any) -> Any:
        return data

    config = {
        "name": "resourceless",
        "data_mode": DataHandlingMode.COPY.value,
        "states": [
            {
                "name": "start",
                "is_start": True,
                "arcs": [
                    {
                        "target": "done",
                        "transform": {"type": "registered", "name": "passthrough"},
                        "metadata": {"name": "loaded"},
                    }
                ],
            },
            {"name": "done", "is_end": True},
        ],
    }
    fsm = AsyncSimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={"passthrough": passthrough},
    )
    # Attach the factory to every context the engine creates.
    fsm._async_engine  # noqa: B018 - ensure engine built
    original = ContextFactory.create_context

    def _patched(*args: Any, **kwargs: Any) -> Any:
        ctx = original(*args, **kwargs)
        ctx.transform_context_factory = factory
        return ctx

    ContextFactory.create_context = staticmethod(_patched)  # type: ignore[assignment]
    try:
        result = await fsm.process({"id": "1"})
    finally:
        ContextFactory.create_context = original  # type: ignore[assignment]
        await fsm.close()

    assert result["success"], f"FSM did not complete cleanly: {result}"
    arc_contexts = [fc for fc in captured if fc.metadata.get("target_state") == "done"]
    assert arc_contexts, (
        "factory was never invoked for the resourceless arc transform — the "
        "FunctionContext was not built unconditionally on the async arc path"
    )
    assert arc_contexts[0].resources == {}


# --------------------------------------------------------------------------- #
# 5: Role-based access via resource_for_role
# --------------------------------------------------------------------------- #

class _RoleUpsert(DatabaseUpsert):
    """DatabaseUpsert variant that resolves its resource by logical role."""

    def __init__(self, role: str, **kwargs: Any) -> None:
        super().__init__(resource_name="_unused", **kwargs)
        self._role = role

    async def transform(self, data: dict[str, Any], context: Any = None) -> dict[str, Any]:
        resource = context.resource_for_role(self._role)
        records = [data]
        result = await resource.upsert(
            table=self.table,
            records=records,
            key_columns=self.key_columns,
        )
        return {**data, "upserted_count": result.get("affected_rows", 0)}


@pytest.mark.asyncio
async def test_arc_transform_role_based_access(tmp_path: Path) -> None:
    """A role-bound arc transform resolves {role: name} via resource_for_role.

    Fails today: FunctionContext has no resource_for_role and the arc path does
    not inject resources or a role map.
    """
    target = {"type": "file", "path": str(tmp_path / "target.json")}
    fsm = _arc_upsert_fsm(target, transform_name="load")
    # Re-key the arc to a role-based {role: name} declaration (hand-built shape)
    # and swap in a role-resolving transform.
    arc = _find_arc(fsm._fsm, "start")
    arc.required_resources = {"database": "target_db"}
    fsm._async_engine._custom_functions["load"] = _RoleUpsert(
        role="database", table="rows", key_columns=["id"]
    )
    try:
        result = await fsm.process({"id": "1", "name": "Bob"})
        assert result["success"], f"FSM did not complete cleanly: {result}"
    finally:
        await fsm.close()

    record = await _read_row(target, "1")
    assert record is not None and record.to_dict().get("name") == "Bob"


# --------------------------------------------------------------------------- #
# 6: require_resource error contract (one message across paths)
# --------------------------------------------------------------------------- #

def test_require_resource_error_contract() -> None:
    """FunctionContext.require_resource and the library helper share one error.

    Fails today: FunctionContext has no require_resource method.
    """
    fc = FunctionContext(state_name="s", function_name="f")

    with pytest.raises(TransformError) as exc_ctx:
        fc.require_resource("missing")
    ctx_msg = str(exc_ctx.value)

    with pytest.raises(TransformError) as exc_lib:
        _require_resource("missing", fc)
    lib_msg = str(exc_lib.value)

    assert ctx_msg == lib_msg, "library and FunctionContext error messages differ"
    assert "missing" in ctx_msg
    assert "not found" in ctx_msg


def test_resource_for_role_error_contract() -> None:
    """resource_for_role raises a clear error for an unbound role."""
    fc = FunctionContext(state_name="s", function_name="f")
    with pytest.raises(TransformError) as exc:
        fc.resource_for_role("database")
    assert "database" in str(exc.value)


# --------------------------------------------------------------------------- #
# 7: Sync arc condition resources (FU1a) — ExecutionEngine._evaluate_pre_test
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_sync_engine_arc_condition_resources(tmp_path: Path) -> None:
    """A resource-bearing arc condition evaluates correctly on the sync engine.

    Fails today: ExecutionEngine._evaluate_pre_test builds FunctionContext with
    no resources, so the predicate sees an empty mapping.
    """

    def gate_open(_data: Any, ctx: Any) -> bool:
        return ctx.resources.get("gate") is not None

    config = {
        "name": "sync_gated",
        "data_mode": DataHandlingMode.COPY.value,
        "resources": [
            {
                "name": "gate",
                "type": "async_database",
                "config": {"type": "file", "path": str(tmp_path / "gate.json")},
            },
        ],
        "states": [
            {
                "name": "start",
                "is_start": True,
                "arcs": [
                    {
                        "target": "open",
                        "resources": ["gate"],
                        "condition": {"type": "registered", "name": "gate_open"},
                        "metadata": {"name": "to_open"},
                    },
                ],
            },
            {"name": "open", "is_end": True},
        ],
    }
    # Build through AsyncSimpleFSM to register the resource provider, then drive
    # the *sync* ExecutionEngine's pre-test path directly.
    helper = AsyncSimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={"gate_open": gate_open},
    )
    try:
        fsm = helper._fsm
        engine = ExecutionEngine(fsm, custom_functions={"gate_open": gate_open})
        ctx = ContextFactory.create_context(
            fsm=fsm,
            data=Record({"id": "1"}),
            data_mode=ProcessingMode.SINGLE,
            resource_manager=helper._resource_manager,
        )
        ctx.set_state("start")
        arc = _find_arc(fsm, "start", "to_open")
        assert engine._evaluate_pre_test(arc, ctx) is True, (
            "sync arc condition did not see its 'gate' resource — "
            "_evaluate_pre_test injected no resources into the FunctionContext"
        )
    finally:
        await helper.close()


# --------------------------------------------------------------------------- #
# 8: transform_context_factory honored on the async engine (state + arc)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_transform_context_factory_honored_on_async_engine(tmp_path: Path) -> None:
    """A factory set on the context wraps both arc and state function contexts.

    Fails today: the async engine builds FunctionContext directly and never
    consults transform_context_factory on either path.
    """
    captured: list[FunctionContext] = []

    def factory(fc: FunctionContext) -> FunctionContext:
        captured.append(fc)
        return fc

    def state_xf(data: Any, _ctx: Any) -> Any:
        return data

    def arc_xf(data: Any, _ctx: Any) -> Any:
        return data

    config = {
        "name": "factory_async",
        "data_mode": DataHandlingMode.COPY.value,
        "states": [
            {
                "name": "start",
                "is_start": True,
                "functions": {"transform": {"type": "registered", "name": "state_xf"}},
                "arcs": [
                    {
                        "target": "done",
                        "transform": {"type": "registered", "name": "arc_xf"},
                        "metadata": {"name": "loaded"},
                    }
                ],
            },
            {"name": "done", "is_end": True},
        ],
    }
    fsm = AsyncSimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={"state_xf": state_xf, "arc_xf": arc_xf},
    )
    original = ContextFactory.create_context

    def _patched(*args: Any, **kwargs: Any) -> Any:
        ctx = original(*args, **kwargs)
        ctx.transform_context_factory = factory
        return ctx

    ContextFactory.create_context = staticmethod(_patched)  # type: ignore[assignment]
    try:
        result = await fsm.process({"id": "1"})
    finally:
        ContextFactory.create_context = original  # type: ignore[assignment]
        await fsm.close()

    assert result["success"], f"FSM did not complete cleanly: {result}"
    state_seen = any(fc.metadata.get("state") == "start" for fc in captured)
    arc_seen = any(fc.metadata.get("target_state") == "done" for fc in captured)
    assert state_seen, "factory not honored on the async STATE transform path"
    assert arc_seen, "factory not honored on the async ARC transform path"


# --------------------------------------------------------------------------- #
# 9: Cross-engine keying parity (FU1b) — hand-built {type: name} arc
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_cross_engine_name_keying_parity(tmp_path: Path) -> None:
    """A {type: name} arc + a name-based function behaves identically on both engines.

    Sync side fails today: ArcExecution keys resources by type, so a
    DatabaseUpsert('target_db') cannot find its resource on a {type: name} arc.
    """
    async_target = {"type": "file", "path": str(tmp_path / "async.json")}
    sync_target = {"type": "file", "path": str(tmp_path / "sync.json")}

    # --- async side: AsyncSimpleFSM with a hand-built {type: name} arc --- #
    fsm = _arc_upsert_fsm(async_target)
    arc = _find_arc(fsm._fsm, "start")
    arc.required_resources = {"database": "target_db"}
    try:
        result = await fsm.process({"id": "1", "name": "Carol"})
        assert result["success"], f"async FSM did not complete: {result}"
    finally:
        await fsm.close()

    async_row = await _read_row(async_target, "1")
    assert async_row is not None and async_row.to_dict().get("name") == "Carol"

    # --- sync side: drive ArcExecution.execute with the same {type: name} arc --- #
    from dataknobs_fsm.core.arc import ArcDefinition, ArcExecution

    helper = AsyncSimpleFSM(
        {
            "name": "sync_arc",
            "data_mode": DataHandlingMode.COPY.value,
            "resources": [
                {"name": "target_db", "type": "async_database", "config": sync_target},
            ],
            "states": [
                {"name": "start", "is_start": True},
                {"name": "done", "is_end": True},
            ],
        },
        data_mode=DataHandlingMode.COPY,
    )
    try:
        sync_fsm = helper._fsm
        upsert = DatabaseUpsert(resource_name="target_db", table="rows", key_columns=["id"])
        arc_def = ArcDefinition(target_state="done", transform="load")
        arc_def.required_resources = {"database": "target_db"}
        arc_exec = ArcExecution(
            arc_def=arc_def,
            source_state="start",
            function_registry={"load": upsert},
        )
        ctx = ContextFactory.create_context(
            fsm=sync_fsm,
            data=Record({"id": "1", "name": "Carol"}),
            data_mode=ProcessingMode.SINGLE,
            resource_manager=helper._resource_manager,
        )
        ctx.set_state("start")
        await arc_exec.execute_async(ctx, {"id": "1", "name": "Carol"})
    finally:
        await helper.close()

    sync_row = await _read_row(sync_target, "1")
    assert sync_row is not None and sync_row.to_dict().get("name") == "Carol", (
        "sync arc transform did not persist on a {type: name} arc — resources "
        "were keyed by type, not name, so DatabaseUpsert('target_db') missed"
    )
