"""Reproduce-first tests for arc resource injection + function-context parity.

Companion to ``test_state_transform_resources.py`` (the state path). These
cover the *arc* path: an arc may declare ``resources`` and its transform AND
condition must receive them through a ``FunctionContext`` on the async engine —
plus the dual-access (name / role) accessors and the
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
from dataknobs_fsm.execution.async_engine import AsyncExecutionEngine
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
#
# Note: there is intentionally no test exercising a raw ``IStateTestFunction``
# instance used directly as an arc *condition* (pre-test). On the engine that
# now runs all execution, an ``IStateTestFunction`` supplied as an arc condition
# is not dispatched, so the resource-reaching-condition scenario that earlier
# relied on that dispatch was removed rather than re-homed against the
# non-dispatching behavior. The arc-condition coverage below uses inline /
# registered condition functions, which the engine does dispatch.

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
    """A {type: name} arc + a name-based function behaves identically on both paths.

    Both the AsyncSimpleFSM end-to-end path and the direct ``ArcExecution``
    path must resolve resources by name: a DatabaseUpsert('target_db') must find
    its resource on a {type: name} arc (keying by type would miss it).
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

    # --- direct ArcExecution path: same {type: name} arc via execute_async --- #
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


# --------------------------------------------------------------------------- #
# 10: transform_context_factory is transform-scoped — sync condition skips it
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_arc_condition_skips_transform_context_factory(
    tmp_path: Path,
) -> None:
    """An arc *condition* gets the plain context; a transform gets the factory.

    ``ArcExecution.can_execute_async`` builds its function context with
    ``apply_factory=False`` (matching the async engine's ``_evaluate_arc``),
    while ``execute_async`` applies the factory. The factory's documented scope
    is transforms, so a condition must NOT be wrapped by it.
    """
    from dataknobs_fsm.core.arc import ArcDefinition, ArcExecution

    condition_contexts: list[Any] = []
    transform_contexts: list[Any] = []

    def factory(fc: Any) -> Any:
        # Stamp every context the factory wraps so misapplication is detectable.
        fc.metadata["factory_applied"] = True
        return fc

    def cond(_data: Any, ctx: Any) -> bool:
        condition_contexts.append(ctx)
        return True

    def xf(data: Any, ctx: Any) -> Any:
        transform_contexts.append(ctx)
        return data

    helper = AsyncSimpleFSM(
        {
            "name": "factory_sync_cond",
            "data_mode": DataHandlingMode.COPY.value,
            "states": [
                {"name": "start", "is_start": True},
                {"name": "done", "is_end": True},
            ],
        },
        data_mode=DataHandlingMode.COPY,
    )
    try:
        fsm = helper._fsm
        arc_def = ArcDefinition(target_state="done", pre_test="cond", transform="xf")
        arc_exec = ArcExecution(
            arc_def=arc_def,
            source_state="start",
            function_registry={"cond": cond, "xf": xf},
        )
        ctx = ContextFactory.create_context(
            fsm=fsm,
            data=Record({"id": "1"}),
            data_mode=ProcessingMode.SINGLE,
            resource_manager=helper._resource_manager,
        )
        ctx.set_state("start")
        ctx.transform_context_factory = factory

        # Condition path: factory must NOT be applied.
        assert await arc_exec.can_execute_async(ctx, ctx.data) is True
        assert condition_contexts, "condition function did not run"
        assert not condition_contexts[0].metadata.get("factory_applied"), (
            "transform_context_factory was applied to an arc CONDITION on the "
            "can_execute_async path — it is transform-scoped"
        )

        # Transform path on the same context: factory SHOULD be applied,
        # proving the factory is actually wired (not merely absent everywhere).
        await arc_exec.execute_async(ctx, ctx.data)
        assert transform_contexts, "transform function did not run"
        assert transform_contexts[0].metadata.get("factory_applied"), (
            "transform_context_factory was not applied to the arc TRANSFORM"
        )
    finally:
        await helper.close()


# --------------------------------------------------------------------------- #
# 11: {role: name} arc resources are reachable from config (dict-shaped)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_arc_dict_resources_bind_roles_from_config(tmp_path: Path) -> None:
    """An arc may declare ``resources`` as a ``{role: name}`` map in config.

    Fails today: the builder produced an identity ``{name: name}`` map from a
    *list* only, so the role indirection (and any dict-shaped binding) was
    unreachable from YAML/dict — ``{r: r for r in {"database": "target_db"}}``
    iterates the KEYS, so ``DatabaseUpsert('target_db')`` never receives its
    resource and nothing persists.
    """
    target = {"type": "file", "path": str(tmp_path / "roles.json")}
    config = {
        "name": "arc_dict_resources",
        "data_mode": DataHandlingMode.COPY.value,
        "resources": [
            {"name": "target_db", "type": "async_database", "config": target},
        ],
        "states": [
            {
                "name": "start",
                "is_start": True,
                "arcs": [
                    {
                        "target": "done",
                        "transform": {"type": "registered", "name": "load"},
                        "resources": {"database": "target_db"},
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
        custom_functions={
            "load": DatabaseUpsert(
                resource_name="target_db", table="rows", key_columns=["id"]
            ),
        },
    )
    try:
        arc = _find_arc(fsm._fsm, "start")
        assert arc.required_resources == {"database": "target_db"}, (
            "dict-shaped arc 'resources' did not produce a {role: name} map; "
            f"got {arc.required_resources!r}"
        )
        result = await fsm.process({"id": "1", "name": "Dana"})
        assert result["success"], f"FSM did not complete cleanly: {result}"
    finally:
        await fsm.close()

    row = await _read_row(target, "1")
    assert row is not None and row.to_dict().get("name") == "Dana", (
        "arc with dict {role: name} resources did not persist via 'target_db'"
    )


# --------------------------------------------------------------------------- #
# 13: the engine acquires an arc's resources once per transition (not per retry)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_arc_resources_acquired_once_per_transition(
    tmp_path: Path,
) -> None:
    """The engine acquires/releases an arc's resources exactly once.

    Guard for the acquire-once invariant: ``_execute_transition`` pre-acquires
    arc resources *before* its retry loop and reuses the handles across attempts
    instead of re-acquiring — and re-running the transform — on every retry. A
    regression that moved allocation back inside the loop would acquire more
    than once whenever a transition retries.
    """
    config = {
        "name": "acquire_once",
        "data_mode": DataHandlingMode.COPY.value,
        "resources": [
            {
                "name": "target_db",
                "type": "async_database",
                "config": {"type": "file", "path": str(tmp_path / "once.json")},
            },
        ],
        "states": [
            {
                "name": "start",
                "is_start": True,
                "arcs": [
                    {
                        "target": "done",
                        "transform": {"type": "registered", "name": "touch"},
                        "resources": ["target_db"],
                        "metadata": {"name": "loaded"},
                    }
                ],
            },
            {"name": "done", "is_end": True},
        ],
    }

    def touch(data: Any, ctx: Any) -> Any:
        # Reading the resource proves injection; raising would surface as a
        # FunctionError rather than a silent miss.
        ctx.require_resource("target_db")
        return data

    helper = AsyncSimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={"touch": touch},
    )
    try:
        fsm = helper._fsm
        # Count acquire/release for the arc resource by wrapping the real
        # resource manager (after build, so build-time wiring isn't counted).
        rm = helper._resource_manager
        acquired: list[Any] = []
        released: list[Any] = []
        real_acquire = rm.acquire
        real_release = rm.release

        def counting_acquire(*args: Any, **kwargs: Any) -> Any:
            acquired.append(kwargs.get("name") or (args[0] if args else None))
            return real_acquire(*args, **kwargs)

        def counting_release(*args: Any, **kwargs: Any) -> Any:
            released.append(kwargs.get("name") or (args[0] if args else None))
            return real_release(*args, **kwargs)

        rm.acquire = counting_acquire  # type: ignore[method-assign]
        rm.release = counting_release  # type: ignore[method-assign]

        engine = AsyncExecutionEngine(fsm, custom_functions={"touch": touch})
        ctx = ContextFactory.create_context(
            fsm=fsm,
            data=Record({"id": "1"}),
            data_mode=ProcessingMode.SINGLE,
            resource_manager=rm,
        )
        ctx.set_state("start")
        arc = _find_arc(fsm, "start")
        await engine._execute_transition(arc, ctx)

        assert acquired.count("target_db") == 1, (
            "arc resource 'target_db' was acquired "
            f"{acquired.count('target_db')} times — expected exactly once "
            "(acquire-once across the retry loop)"
        )
        assert released.count("target_db") == 1, (
            "arc resource 'target_db' was released "
            f"{released.count('target_db')} times — expected exactly once"
        )
    finally:
        await helper.close()


# --------------------------------------------------------------------------- #
# 14: an UNEXPECTED arc-condition exception surfaces as a record error
# --------------------------------------------------------------------------- #

def _two_arc_gate_config(condition_name: str) -> dict[str, Any]:
    """A start state with a priority-10 gated arc + an unconditional fallback."""
    return {
        "name": "gated",
        "data_mode": DataHandlingMode.COPY.value,
        "states": [
            {
                "name": "start",
                "is_start": True,
                "arcs": [
                    {
                        "target": "open",
                        "condition": {"type": "registered", "name": condition_name},
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


@pytest.mark.asyncio
async def test_async_arc_condition_unexpected_exception_errors_record() -> None:
    """A condition raising an UNEXPECTED error makes the *record* error.

    The async engine evaluates conditions as concurrent ``asyncio`` tasks in
    ``_get_available_transitions``. A condition that raises a non-validation
    error — most realistically ``require_resource()`` for a resource that is
    missing or down — means the engine could not evaluate whether the arc
    applies. It must NOT silently de-select that arc and fall through to the
    unconditional ``to_closed`` arc: doing so would route the record on a
    routing decision made with incomplete information and hide an
    infrastructure failure as a clean outcome. Instead the exception surfaces
    as a record error (``success=False``), and sibling evaluation tasks are
    awaited so none is orphaned.

    Fails against the old "catch all exceptions -> de-select arc" behaviour,
    which would report ``success=True`` at ``closed``.
    """

    def gate_raises(_data: Any, ctx: Any) -> bool:
        ctx.require_resource("absent")  # TransformError — an unexpected failure
        return True  # unreachable

    fsm = AsyncSimpleFSM(
        _two_arc_gate_config("gate_raises"),
        data_mode=DataHandlingMode.COPY,
        custom_functions={"gate_raises": gate_raises},
    )
    try:
        result = await fsm.process({"id": "1"})
    finally:
        await fsm.close()

    assert result["success"] is False, (
        "an unexpected arc-condition exception was swallowed and the record "
        f"silently fell through to the fallback arc (got {result})"
    )


@pytest.mark.asyncio
async def test_async_arc_condition_validation_error_is_soft_reject() -> None:
    """A condition raising ``ValidationError`` is a soft reject (de-select).

    ``ValidationError`` is the explicit "this record is invalid" signal, so a
    condition raising it de-selects its arc (a clean reject) rather than
    erroring the record — the run falls through to the unconditional
    ``to_closed`` arc. This is the reject-vs-error boundary that keeps a
    validation gate's rejects distinct from infrastructure failures.
    """
    from dataknobs_fsm.functions.base import ValidationError as FSMValidationError

    def gate_invalid(_data: Any, _ctx: Any) -> bool:
        raise FSMValidationError("record is invalid")

    fsm = AsyncSimpleFSM(
        _two_arc_gate_config("gate_invalid"),
        data_mode=DataHandlingMode.COPY,
        custom_functions={"gate_invalid": gate_invalid},
    )
    try:
        result = await fsm.process({"id": "1"})
    finally:
        await fsm.close()

    assert result["success"], (
        "a ValidationError-raising condition should be a soft reject (arc "
        f"de-selected), not a record error (got {result})"
    )
    assert result["final_state"] == "closed", (
        "the soft-rejected arc was not de-selected — expected fall-through to "
        f"the unconditional 'to_closed' arc (got {result['final_state']!r})"
    )


# --------------------------------------------------------------------------- #
# 16: BATCH mode isolates a raising condition per record (no batch abort)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_batch_mode_isolates_a_raising_condition_per_record() -> None:
    """A condition raising in BATCH mode errors one record, not the batch.

    The engine's ``execute()`` wraps the data-mode dispatch in a try/except so a
    propagated condition error becomes a failed result. In BATCH/STREAM mode
    that wrapper must NOT abandon the whole batch when one record's condition
    raises — ``_execute_batch`` isolates each record via per-record
    ``gather(return_exceptions=True)`` so the others still process.

    Fails against an unisolated batch loop: the middle record's raise escapes
    ``_execute_batch``, the outer wrapper returns ``(False, str(e))`` (a string,
    not the per-record ``{results, errors}`` dict), and the last record is never
    processed.
    """

    def gate(data: Any, _ctx: Any) -> bool:
        if data.get("boom"):
            raise RuntimeError("gate blew up on this record")
        return True

    config = {
        "name": "batch_gate",
        "data_mode": DataHandlingMode.COPY.value,
        "states": [
            {
                "name": "start",
                "is_start": True,
                "arcs": [
                    {
                        "target": "done",
                        "condition": {"type": "registered", "name": "gate"},
                        "metadata": {"name": "go"},
                    },
                ],
            },
            {"name": "done", "is_end": True},
        ],
    }
    helper = AsyncSimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={"gate": gate},
    )
    try:
        fsm = helper._fsm
        engine = AsyncExecutionEngine(fsm, custom_functions={"gate": gate})
        ctx = ContextFactory.create_context(
            fsm=fsm,
            data={"id": "1"},
            data_mode=ProcessingMode.BATCH,
            resource_manager=helper._resource_manager,
        )
        ctx.batch_data = [{"id": "1"}, {"id": "2", "boom": True}, {"id": "3"}]
        ctx.set_state("start")
        success, result = await engine.execute(ctx)
    finally:
        await helper.close()

    assert isinstance(result, dict), (
        "the raising middle record abandoned the whole batch — the outer "
        f"execute() wrapper returned a non-dict result (got {result!r})"
    )
    assert len(result["results"]) == 2, (
        "the two clean records were not both processed — the batch aborted at "
        f"the raising record instead of isolating it (got {result})"
    )
    assert len(result["errors"]) == 1, (
        f"the raising record was not recorded as a single batch error: {result}"
    )
