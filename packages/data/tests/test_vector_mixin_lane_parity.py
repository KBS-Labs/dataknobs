"""A backend's vector methods belong to the lane the backend is in.

``VectorOperationsMixin`` declared every one of its methods ``async`` and was
mixed into sync backends as readily as async ones --- five of the seven sync
backends carry a vector surface, and all five got the async mixin. Three of
the methods it *implements* (rather than declares abstract) then called
``await self.read(...)`` / ``await self.delete(...)`` / ``await
self.search(...)`` on a sync database, so on a sync backend they raised
``TypeError: object NoneType can't be used in 'await' expression`` --- and the
sibling defect on the declared half was that those five overrode an async
declaration with a sync definition, which a type checker reported and nothing
else reported at all.

This is the same defect the branch already fixed one module away, in the
opposite direction: every async backend had inherited the *sync*
``BulkEmbedMixin``, whose ``self.exists`` / ``self.update`` / ``self.create``
calls produced coroutines nobody awaited --- and a coroutine object is truthy,
so the ``exists`` branch was taken unconditionally and nothing was ever
written, with no exception anywhere.

``TestEveryBackendsVectorMethodsMatchItsLane`` is the guard that would have
caught both. It is a property of the whole family rather than of any one
backend, which is why it is a sweep over all fourteen rather than a cell per
class.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from dataknobs_data import Record
from dataknobs_data.backends.file import SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.database import AsyncDatabase, SyncDatabase
from dataknobs_data.vector.python_vector_search import PythonVectorSearchMixin
from dataknobs_data.vector.types import DistanceMetric

BACKEND_MODULES = (
    "memory",
    "file",
    "sqlite",
    "sqlite_async",
    "s3",
    "s3_async",
    "duckdb",
    "postgres",
    "elasticsearch",
    "elasticsearch_async",
)

# Every method of the vector surface whose laneness a caller depends on ---
# the two abstract ones, the ones the mixin implements, and the two private
# helpers ``hybrid_search`` awaits, which have to agree with it or it breaks.
#
# ``_vector_search`` is here because ``vector_search`` no longer is, in the
# sense this sweep means: every backend now inherits the mixin's template, so
# the cell for it resolves to the same function fourteen times and checks no
# backend-authored body at all. The laneness question moved to the hook with
# the code, and a ``def _vector_search`` on an async backend fails only at the
# ``await`` inside the template it is called from --- one frame away from the
# file that would have to be edited.
LANE_METHODS = (
    "vector_search",
    "_vector_search",
    "bulk_embed_and_store",
    "update_vector",
    "delete_from_index",
    "hybrid_search",
    "create_vector_index",
    "drop_vector_index",
    "get_vector_index_stats",
    "_text_search_for_hybrid",
    "_supports_native_hybrid",
)


def _backend_classes() -> list[type]:
    found: list[type] = []
    for name in BACKEND_MODULES:
        module = importlib.import_module(f"dataknobs_data.backends.{name}")
        found.extend(
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj)
            and obj.__module__ == module.__name__
            and issubclass(obj, (AsyncDatabase, SyncDatabase))
        )
    return found


BACKENDS = _backend_classes()


def _cases() -> list[tuple[type, str]]:
    return [(cls, method) for cls in BACKENDS for method in LANE_METHODS]


# Exactly the keywords ``hybrid_search`` passes to ``vector_search`` in both
# lanes. An implementation that does not name every one of them is one the
# mixin cannot call, whichever lane it is in.
HYBRID_CALL_KEYWORDS = ("query_vector", "vector_field", "k", "metric", "filter")


def _named_parameters(function: object) -> dict[str, inspect.Parameter]:
    """The parameters a caller can reach *by name*.

    ``**kwargs`` is deliberately not consulted. A keyword it swallows binds
    at the signature and then goes wherever the body forwards it, which for
    ``SyncSQLiteDatabase`` was into a second value for a parameter already
    passed positionally --- a `TypeError` one frame further down than any
    signature check looks.
    """
    parameters = inspect.signature(function).parameters  # type: ignore[arg-type]
    return {
        name: parameter
        for name, parameter in parameters.items()
        if name != "self"
        and parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
    }


class TestEveryBackendsVectorMethodsMatchItsLane:
    """The recurrence guard, over the whole family at once."""

    def test_the_sweep_found_the_backends(self) -> None:
        """A parity sweep that enumerates nothing passes vacuously."""
        assert len(BACKENDS) == 14

    def test_the_sweep_found_live_cells(self) -> None:
        """...and neither does one whose every cell skips.

        ``test_laneness`` skips a `(class, method)` pair the class does not
        offer, which is legitimate --- both DuckDB classes carry no vector
        surface at all. They are now the *only* legitimate skip: a
        twenty-cell gap, and exactly the two classes that never had one.

        It was thirty until ``vector_search`` moved onto the mixin.
        `SyncElasticsearchDatabase` inherited neither vector mixin and
        defined two of the ten members from nowhere, so the other eight
        raised ``AttributeError`` on it and answered on its async twin ---
        which no cell here asserted, because a missing method skips.

        A mixin method *renamed* would empty every cell while the count
        above still passed, so the floor is asserted rather than assumed.
        """
        live = sum(1 for cls, method in _cases() if getattr(cls, method, None) is not None)

        absent = 2 * len(LANE_METHODS)  # both DuckDB classes, every method
        assert live == len(_cases()) - absent, (
            f"{live} of {len(_cases())} cells are live; the only absences "
            f"should be the two DuckDB classes' {len(LANE_METHODS)} each"
        )

    @pytest.mark.parametrize(
        ("cls", "method"), _cases(), ids=lambda x: x if isinstance(x, str) else x.__name__
    )
    def test_laneness(self, cls: type, method: str) -> None:
        function = getattr(cls, method, None)
        if function is None:
            pytest.skip(f"{cls.__name__} does not offer {method}")

        lane_is_async = issubclass(cls, AsyncDatabase)

        assert inspect.iscoroutinefunction(function) is lane_is_async, (
            f"{cls.__name__}.{method} is "
            f"{'async' if inspect.iscoroutinefunction(function) else 'sync'} "
            f"but {cls.__name__} is a "
            f"{'AsyncDatabase' if lane_is_async else 'SyncDatabase'}"
        )


class TestEveryBackendCanBeCalledTheWayTheMixinCallsIt:
    """Laneness was the wrong property to stop at.

    The split made ``hybrid_search`` reachable on five sync backends, and it
    calls ``vector_search`` by keyword. Four of the twelve implementations
    named the field parameter ``field_name`` instead --- so two of the five
    raised rather than searched, and the same spelling every shipped example
    uses (``vector_field=``) had never worked on them through a direct call
    either:

        SyncSQLiteDatabase   TypeError: python_vector_search_sync() got
                             multiple values for keyword argument
                             'vector_field'
        SyncPostgresDatabase TypeError: vector_search() got an unexpected
                             keyword argument 'vector_field'

    `test_laneness` cannot see this --- both methods are in the right lane.
    """

    VECTOR_SEARCHERS: ClassVar[list[type]] = [
        cls for cls in BACKENDS if getattr(cls, "vector_search", None)
    ]

    @pytest.mark.parametrize("cls", VECTOR_SEARCHERS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("keyword", HYBRID_CALL_KEYWORDS)
    def test_the_keyword_is_a_named_parameter(self, cls: type, keyword: str) -> None:
        named = _named_parameters(cls.vector_search)

        assert keyword in named, (
            f"{cls.__name__}.vector_search has no parameter named {keyword!r} "
            f"(it has {sorted(named)}) --- the mixin's hybrid_search passes it"
        )

    @pytest.mark.parametrize("cls", VECTOR_SEARCHERS, ids=lambda c: c.__name__)
    def test_only_the_query_vector_is_required(self, cls: type) -> None:
        """A field the caller must supply is one ``hybrid_search`` cannot omit.

        Both Postgres classes made ``vector_field`` a required positional
        where the other ten default it, so the twelve did not agree on what a
        minimal call looks like either.
        """
        required = [
            name
            for name, parameter in _named_parameters(cls.vector_search).items()
            if parameter.default is inspect.Parameter.empty and name != "query_vector"
        ]

        assert required == [], f"{cls.__name__}.vector_search also requires {required}"


@pytest.fixture(params=["memory", "file", "sqlite"])
def sync_db(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SyncDatabase]:
    """Every sync vector backend that runs in-process.

    Parametrised rather than hard-coded to memory, because memory is one of
    the three that happened to work --- the two that raised were sqlite and
    postgres, and a single-backend cell is what let that through.
    ``SyncS3Database`` and ``SyncPostgresDatabase`` are the other two sync
    backends with a vector surface; both need a service, so the signature
    sweep above is what covers them here.
    """
    backend = request.param
    if backend == "memory":
        database: SyncDatabase = SyncMemoryDatabase(config={"vector_enabled": True})
    elif backend == "file":
        database = SyncFileDatabase(
            config={"path": str(tmp_path / "records"), "vector_enabled": True}
        )
    else:
        database = SyncSQLiteDatabase(
            config={"path": str(tmp_path / "records.db"), "vector_enabled": True}
        )
    database.connect()
    database.create(Record(data={"id": "r1", "content": "hello world", "embedding": [1.0, 0.0]}))
    try:
        yield database
    finally:
        database.close()


class TestTheInheritedMethodsRunOnASyncBackend:
    """The reproduce cells: three ``TypeError``s, measured before the split."""

    def test_update_vector(self, sync_db: SyncDatabase) -> None:
        assert sync_db.update_vector("r1", "embedding", [0.0, 1.0]) is True

    def test_update_vector_actually_stores_the_vector(self, sync_db: SyncDatabase) -> None:
        """``update`` returns ``bool``, and ``bool is not None`` is always true.

        So the return value said nothing about whether the write landed ---
        a version conflict answered ``True`` exactly as a success did.
        """
        sync_db.update_vector("r1", "embedding", [0.0, 1.0])

        stored = sync_db.read("r1")
        assert stored is not None
        assert list(stored.get_value("embedding")) == [0.0, 1.0]

    def test_update_vector_missing_record(self, sync_db: SyncDatabase) -> None:
        assert sync_db.update_vector("nope", "embedding", [0.0, 1.0]) is False

    def test_update_vector_reports_a_write_that_did_not_happen(self, sync_db: SyncDatabase) -> None:
        """``update`` returns ``bool``; ``False is not None`` is ``True``.

        The record is readable when ``update_vector`` reads it and gone by
        the time it writes --- the shape of a delete landing between the two,
        which is the one path that reaches a ``False`` from ``update`` on a
        backend where the id exists at read time. The subclass exists to make
        that interleaving happen at a fixed point rather than by luck; it is
        a real ``SyncMemoryDatabase`` in every other respect, and it exercises
        the mixin's real code path.
        """

        class DeletedBetweenReadAndWrite(SyncMemoryDatabase):
            def update(self, id: str, record: Record, **kwargs: object) -> bool:
                self.delete(id)
                return super().update(id, record, **kwargs)  # type: ignore[arg-type]

        database = DeletedBetweenReadAndWrite(config={"vector_enabled": True})
        database.create(Record(data={"id": "r1", "content": "x", "embedding": [1.0, 0.0]}))

        assert database.update_vector("r1", "embedding", [0.0, 1.0]) is False

    def test_delete_from_index(self, sync_db: SyncDatabase) -> None:
        assert sync_db.delete_from_index("r1") is True

    def test_hybrid_search(self, sync_db: SyncDatabase) -> None:
        results = sync_db.hybrid_search("hello", np.array([1.0, 0.0]), text_fields=["content"])

        assert [r.record.get_value("content") for r in results] == ["hello world"]

    def test_vector_search_takes_the_documented_keyword(self, sync_db: SyncDatabase) -> None:
        """``vector_field=`` is the spelling every shipped example uses."""
        results = sync_db.vector_search(query_vector=np.array([1.0, 0.0]), vector_field="embedding")

        assert [r.record.get_value("content") for r in results] == ["hello world"]

    def test_the_index_helpers(self, sync_db: SyncDatabase) -> None:
        assert sync_db.create_vector_index("embedding") is True
        assert sync_db.drop_vector_index("embedding") is True
        assert sync_db.get_vector_index_stats("embedding")["field"] == "embedding"


class TestTheAsyncLaneStillWorks:
    """The other half of the split, unchanged in behaviour."""

    @pytest.fixture
    async def db(self) -> AsyncMemoryDatabase:
        database = AsyncMemoryDatabase(config={"vector_enabled": True})
        await database.create(
            Record(data={"id": "r1", "content": "hello world", "embedding": [1.0, 0.0]})
        )
        return database

    @pytest.mark.asyncio
    async def test_update_vector(self, db: AsyncMemoryDatabase) -> None:
        assert await db.update_vector("r1", "embedding", [0.0, 1.0]) is True

    @pytest.mark.asyncio
    async def test_delete_from_index(self, db: AsyncMemoryDatabase) -> None:
        assert await db.delete_from_index("r1") is True

    @pytest.mark.asyncio
    async def test_hybrid_search(self, db: AsyncMemoryDatabase) -> None:
        results = await db.hybrid_search("hello", np.array([1.0, 0.0]), text_fields=["content"])

        assert [r.record.get_value("content") for r in results] == ["hello world"]

    @pytest.mark.asyncio
    async def test_the_index_helpers(self, db: AsyncMemoryDatabase) -> None:
        assert await db.create_vector_index("embedding") is True
        assert await db.drop_vector_index("embedding") is True
        assert (await db.get_vector_index_stats("embedding"))["field"] == "embedding"


class TestBothLanesAreImportable:
    """Named separately, so a backend author picks rather than inherits one."""

    def test_the_sync_lane_declares_sync_methods(self) -> None:
        from dataknobs_data.vector import SyncVectorOperationsMixin

        for method in LANE_METHODS:
            function = getattr(SyncVectorOperationsMixin, method)
            assert not inspect.iscoroutinefunction(function), method

    def test_the_async_lane_declares_async_methods(self) -> None:
        from dataknobs_data.vector import AsyncVectorOperationsMixin

        for method in LANE_METHODS:
            function = getattr(AsyncVectorOperationsMixin, method)
            assert inspect.iscoroutinefunction(function), method

    def test_the_old_name_still_resolves_to_the_async_lane(self) -> None:
        """Back-compat: the bare name has always meant the async lane."""
        from dataknobs_data.vector import (
            AsyncVectorOperationsMixin,
            VectorOperationsMixin,
        )

        assert VectorOperationsMixin is AsyncVectorOperationsMixin


# What the mixin *declares*, as against what ``hybrid_search`` happens to
# pass. The two sets differed by exactly the two parameters no backend
# implemented, which is how the divergence stayed invisible to the sweep
# above: a guard keyed to the caller's keywords cannot see a keyword the
# caller never uses.
DECLARED_KEYWORDS = (
    "vector_field",
    "k",
    "metric",
    "filter",
    "include_source",
    "score_threshold",
)

# What the raw k-NN hook takes. The two the template owns are absent, and
# their absence is the point: a backend cannot answer them differently from
# the other eleven if it never sees them.
HOOK_KEYWORDS = ("vector_field", "k", "metric", "filter")


def _hook_implementers() -> list[type]:
    return [cls for cls in BACKENDS if "_vector_search" in cls.__dict__]


class TestOneSearchOverTwelveHooks:
    """The recurrence guard for the vector-search contract.

    ``vector_search`` was ``@abstractmethod`` with no body, so twelve
    backends wrote twelve answers to the two parameters it declared: two
    honoured them, eight swallowed them into ``**kwargs``, and two raised
    ``TypeError``. The swallow was the worst of the three because it is the
    only one that is silent --- on ``AsyncMemoryDatabase``, three records
    scoring 1.0, 0.994 and 0.0 all came back from a call that asked for
    ``score_threshold=0.99``.

    The fix implements the contract once, on the mixin that declares it, over
    a ``_vector_search`` hook carrying only the raw k-NN. This guard is what
    stops a thirteenth backend from re-opening the question: it does not ask
    whether an override behaves correctly, it asks that there be no override.
    """

    def test_the_sweep_found_twelve_implementers(self) -> None:
        """A guard that enumerates nothing passes vacuously.

        Twelve, not eight: an earlier count imported five backend modules by
        hand and missed ``sqlite_async``, ``s3``, ``s3_async`` and
        ``elasticsearch_async``.
        """
        assert len(_hook_implementers()) == 12, sorted(c.__name__ for c in _hook_implementers())

    @pytest.mark.parametrize("cls", BACKENDS, ids=lambda c: c.__name__)
    def test_no_backend_carries_its_own_vector_search(self, cls: type) -> None:
        """The template is the only implementation, by construction.

        This is the whole guard. Checking that an override *honours* the two
        parameters would pass a backend that honoured them differently ---
        over-fetching to refill ``k``, say, where the contract post-filters
        --- and the divergence this closes was never about any one backend
        being wrong.
        """
        assert "vector_search" not in cls.__dict__, (
            f"{cls.__name__} defines its own vector_search; the contract is "
            f"implemented once on the mixin, over the _vector_search hook"
        )

    @pytest.mark.parametrize("cls", _hook_implementers(), ids=lambda c: c.__name__)
    def test_the_hook_takes_exactly_the_hook_keywords(self, cls: type) -> None:
        named = _named_parameters(cls._vector_search)

        assert sorted(named) == sorted(("query_vector", *HOOK_KEYWORDS)), (
            f"{cls.__name__}._vector_search takes {sorted(named)}"
        )

    @pytest.mark.parametrize("cls", _hook_implementers(), ids=lambda c: c.__name__)
    def test_the_hook_swallows_nothing(self, cls: type) -> None:
        """``**kwargs`` is the mechanism, not a tidy-up.

        Without removing it the eight would have failed as loudly as Postgres
        did, and the defect would have been found the first time anyone
        passed ``score_threshold``. Left in place, the next
        declared-but-unimplemented keyword lands in exactly this position,
        just as silently.
        """
        kinds = inspect.signature(cls._vector_search).parameters.values()

        assert not any(p.kind is p.VAR_KEYWORD for p in kinds), (
            f"{cls.__name__}._vector_search still swallows unrecognised keywords"
        )

    @pytest.mark.parametrize("cls", _hook_implementers(), ids=lambda c: c.__name__)
    def test_everything_after_the_query_vector_is_keyword_only(self, cls: type) -> None:
        """``f3c1c99f`` fixed the call form at the declaration and left the
        implementations alone, which is why ten of them still spelled it
        ``(..., k, filter, metric)`` where the declaration says ``(..., k,
        metric, filter)``. The hook is new, so it can simply refuse the
        positional form that was never portable.
        """
        positional = [
            name
            for name, p in _named_parameters(cls._vector_search).items()
            if p.kind is p.POSITIONAL_OR_KEYWORD and name != "query_vector"
        ]

        assert positional == [], f"{cls.__name__}._vector_search takes {positional} positionally"

    @pytest.mark.parametrize("cls", _hook_implementers(), ids=lambda c: c.__name__)
    def test_the_declared_keywords_reach_every_implementer(self, cls: type) -> None:
        """Through the template, since none of them declares these itself."""
        named = _named_parameters(cls.vector_search)

        assert sorted(named) == sorted(("query_vector", *DECLARED_KEYWORDS)), (
            f"{cls.__name__}.vector_search takes {sorted(named)}"
        )


def _declaring_base(cls: type, method: str) -> type | None:
    """The mixin the method is declared on, if this class overrides one."""
    from dataknobs_data.vector import (
        AsyncVectorOperationsMixin,
        SyncVectorOperationsMixin,
    )

    for base in (SyncVectorOperationsMixin, AsyncVectorOperationsMixin):
        if issubclass(cls, base) and method in base.__dict__ and method in cls.__dict__:
            return base
    return None


def _override_cases() -> list[tuple[type, str]]:
    return [
        (cls, method)
        for cls in BACKENDS
        for method in LANE_METHODS
        if _declaring_base(cls, method) is not None
    ]


class TestAnOverrideAcceptsWhatTheBaseDeclares:
    """A call written against the mixin has to work on every backend.

    ``vector_search`` was the loud case, but the same shape sat on three of
    its neighbours. ``AsyncPostgresDatabase`` made ``vector_field`` a
    *required* argument on ``drop_vector_index`` and
    ``get_vector_index_stats`` where the mixin defaults it, and made both
    ``vector_field`` and ``dimensions`` required on ``create_vector_index``
    --- so ``db.get_vector_index_stats()``, which the mixin says is a
    complete call, raised ``TypeError`` there and answered on nine other
    backends.

    This generalises the two cells above: rather than naming the keywords
    one caller happens to pass, it compares each override against the
    declaration it overrides. A method added to the mixin is covered the day
    it lands, with no edit here.
    """

    def test_the_sweep_found_overrides(self) -> None:
        """Fourteen: the backends with native index and hybrid support.

        It was eleven while ``SyncPostgresDatabase`` took the mixin's
        ``create_vector_index``, ``drop_vector_index`` and
        ``get_vector_index_stats`` --- which return ``True``, ``True`` and an
        empty dict, so it reported that it had built an index and built
        nothing. Its three now restate the declaration and are swept with the
        rest.

        Every other backend takes the mixin's implementation unchanged, so
        it cannot disagree with a declaration it does not restate --- which
        is the whole argument for ``vector_search`` moving up here too.

        Plus the twelve ``_vector_search`` hooks: the hook is declared
        abstract on the mixin and implemented by every backend with a vector
        surface, so each one is an override this sweep can compare against
        its declaration. ``TestOneSearchOverTwelveHooks`` pins the hook's
        exact keyword set; this reaches it from the generic direction, which
        is what keeps the next method added to the mixin covered without an
        edit here.
        """
        assert len(_override_cases()) == 26, sorted(
            (cls.__name__, method) for cls, method in _override_cases()
        )

    @pytest.mark.parametrize(
        ("cls", "method"),
        _override_cases(),
        ids=lambda x: x if isinstance(x, str) else x.__name__,
    )
    def test_every_declared_parameter_is_reachable(self, cls: type, method: str) -> None:
        base = _declaring_base(cls, method)
        assert base is not None
        declared = _named_parameters(getattr(base, method))
        override = inspect.signature(cls.__dict__[method]).parameters
        swallows = any(p.kind is p.VAR_KEYWORD for p in override.values())

        missing = [name for name in declared if name not in override and not swallows]

        assert missing == [], (
            f"{cls.__name__}.{method} has no parameter named {missing} --- "
            f"the mixin declares them, so a caller written against it fails here"
        )

    @pytest.mark.parametrize(
        ("cls", "method"),
        _override_cases(),
        ids=lambda x: x if isinstance(x, str) else x.__name__,
    )
    def test_nothing_optional_becomes_required(self, cls: type, method: str) -> None:
        """The half a keyword check cannot see.

        An override may *name* every declared parameter and still refuse the
        call the declaration permits, by dropping the default. That is what
        all three Postgres index methods did.
        """
        base = _declaring_base(cls, method)
        assert base is not None
        declared = _named_parameters(getattr(base, method))
        override = _named_parameters(cls.__dict__[method])

        promoted = [
            name
            for name, parameter in declared.items()
            if parameter.default is not inspect.Parameter.empty
            and name in override
            and override[name].default is inspect.Parameter.empty
        ]

        assert promoted == [], (
            f"{cls.__name__}.{method} requires {promoted}, which the mixin defaults"
        )


def _python_search_helpers() -> list[Callable[..., object]]:
    """The two helpers the eight forwarding backends land in."""
    return [
        PythonVectorSearchMixin.python_vector_search_sync,
        PythonVectorSearchMixin.python_vector_search_async,
    ]


class TestTheSwallowIsGoneOneFrameDownToo:
    """The eight hooks stopped swallowing; what they call still did.

    ``python_vector_search_sync`` and its async twin each carried
    ``**kwargs`` that no line of either body read --- the same silent bind
    the hooks lost, one call deep, and unreachable from in-tree code only for
    as long as no caller adds a keyword. A keyword that lands here is dropped
    exactly as ``score_threshold`` was, and the backend above it now has a
    signature that promises it cannot happen.
    """

    @pytest.mark.parametrize("helper", _python_search_helpers(), ids=lambda f: f.__name__)
    def test_the_helper_swallows_nothing(self, helper: Callable[..., object]) -> None:
        kinds = inspect.signature(helper).parameters.values()

        assert not any(p.kind is p.VAR_KEYWORD for p in kinds), (
            f"{helper.__name__} still swallows unrecognised keywords"
        )

    @pytest.mark.parametrize("helper", _python_search_helpers(), ids=lambda f: f.__name__)
    def test_the_helper_refuses_a_keyword_it_does_not_declare(
        self, helper: Callable[..., object]
    ) -> None:
        """Pinned from the outside as well, because a signature is only half
        the claim: a body that re-collected keywords would satisfy the check
        above and swallow all the same.
        """
        db = SyncMemoryDatabase() if helper.__name__.endswith("_sync") else AsyncMemoryDatabase()

        with pytest.raises(TypeError, match="score_threshold"):
            helper(db, [1.0, 0.0], metric=DistanceMetric.COSINE, score_threshold=0.99)

    def test_the_sync_helper_resolves_its_metric_the_one_way(self) -> None:
        """Both bodies re-implemented ``resolve_metric`` inline.

        The ``None`` fallback to ``self.vector_metric`` and the ``str``
        coercion through ``DistanceMetric(...)``, copied into each twin ---
        so a name the one resolver accepts was refused here, which is a
        second answer to a question already settled one frame up. Asked by
        passing a name only the resolver knows, rather than by reading the
        source, because what matters is that the two vocabularies agree.
        """
        db = SyncMemoryDatabase()

        assert PythonVectorSearchMixin.python_vector_search_sync(db, [1.0, 0.0], metric="cos") == []

    @pytest.mark.asyncio
    async def test_the_async_helper_resolves_its_metric_the_one_way(self) -> None:
        db = AsyncMemoryDatabase()

        assert (
            await PythonVectorSearchMixin.python_vector_search_async(db, [1.0, 0.0], metric="cos")
            == []
        )


class TestTheTwinsSpellTheSameMethodTheSameWay:
    """Twin parity is not only laneness --- it is also the defaults.

    The sweep above compares the *names* a caller may pass and the lane the
    method runs in. Two twins can agree on both and still answer differently,
    because a default is part of what a call means: ``create_vector_index()``
    with no metric read the database's configuration on the async Postgres
    twin and did not on the sync one, and ``metric=None`` --- which the mixin
    declares --- worked on one and raised ``Unknown distance metric 'none'``
    on the other.

    The two were written in the same pass, one screen apart, and diverged in
    three separate ways: the default's spelling (``DistanceMetric.COSINE``
    against ``"cosine"``), whether the argument is resolved before use, and
    therefore what ``None`` means.
    """

    TWINS = (
        ("SyncPostgresDatabase", "AsyncPostgresDatabase", "postgres"),
        ("SyncElasticsearchDatabase", "AsyncElasticsearchDatabase", None),
    )

    def _pair(self, sync_name: str, async_name: str, module: str | None) -> tuple[type, type]:
        if module:
            found = importlib.import_module(f"dataknobs_data.backends.{module}")
            return getattr(found, sync_name), getattr(found, async_name)
        return (
            getattr(importlib.import_module("dataknobs_data.backends.elasticsearch"), sync_name),
            getattr(
                importlib.import_module("dataknobs_data.backends.elasticsearch_async"),
                async_name,
            ),
        )

    @pytest.mark.parametrize(("sync_name", "async_name", "module"), TWINS)
    @pytest.mark.parametrize("method", ["create_vector_index", "drop_vector_index"])
    def test_the_twins_agree_on_every_default(
        self, sync_name: str, async_name: str, module: str | None, method: str
    ) -> None:
        sync_cls, async_cls = self._pair(sync_name, async_name, module)
        if method not in sync_cls.__dict__ and method not in async_cls.__dict__:
            pytest.skip(f"neither twin overrides {method}")

        sync_defaults = {
            name: parameter.default
            for name, parameter in _named_parameters(getattr(sync_cls, method)).items()
        }
        async_defaults = {
            name: parameter.default
            for name, parameter in _named_parameters(getattr(async_cls, method)).items()
        }

        assert sync_defaults == async_defaults, (
            f"{sync_name}.{method} and {async_name}.{method} declare different "
            f"defaults, so the same call means different things on the two twins"
        )

    @pytest.mark.parametrize("lane", ["sync", "async"])
    def test_both_elasticsearch_twins_ask_for_the_source_the_same_way(self, lane: str) -> None:
        """One twin passed ``source=True`` and the other ``_source=True``.

        Both reach the transport --- elasticsearch-py rewrites body-field
        aliases before dispatch --- so no runtime check could tell them
        apart, which is why two spellings survived on two lines that do the
        same thing. Only one is a parameter of ``search`` though, and mypy
        says so: ``_source`` is a ``call-arg`` error wherever the client is
        typed. On the async twin it is not typed, which is the whole reason
        the wrong spelling was the one that looked established.
        """
        module = "elasticsearch" if lane == "sync" else "elasticsearch_async"
        source = inspect.getsource(
            getattr(
                importlib.import_module(f"dataknobs_data.backends.{module}"),
                "SyncElasticsearchDatabase" if lane == "sync" else "AsyncElasticsearchDatabase",
            )._vector_search
        )
        assert "source=True" in source and "_source=True" not in source, (
            "both twins pass `source=`, the spelling `search` declares; "
            "`_source=` works only because the client rewrites the alias"
        )
