# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The versioning layer's store, and the two implementations of it.

Almost every test here runs twice, against :class:`InMemoryVersionStore` and
against :class:`DatabaseVersionStore` over an ``AsyncMemoryDatabase`` --- both
real implementations, neither a mock nor a fake. A protocol with two
implementations is a promise that a consumer can develop against one and
deploy against the other, and the only way to keep that promise is to run one
set of assertions against both. Where they are *allowed* to differ, the test
says so out loud; at the time of writing there is no such test, because there
is no such difference.

What these replace was not a store. The three managers each took a
``storage: Any`` duck-typed for ``set``, ``append`` and ``delete``: two verbs
no DataKnobs backend has, and one it has by coincidence of name. So writes
vanished, reads came from an instance dictionary, and deletes fired against
ids nothing had written.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from dataknobs_data.backends.memory import AsyncMemoryDatabase

from dataknobs_llm.prompts.versioning import (
    DatabaseVersionStore,
    ExperimentStore,
    InMemoryVersionStore,
    MetricEvent,
    MetricsStore,
    PromptExperiment,
    PromptMetrics,
    PromptVariant,
    PromptVersion,
    VersioningStore,
    VersionStatus,
    VersionStore,
    require_store,
)


@pytest.fixture(params=["in-memory", "database"])
def store(request: pytest.FixtureRequest) -> VersioningStore:
    """One of the two shipped stores, so every test below runs against both."""
    if request.param == "in-memory":
        return InMemoryVersionStore()
    return DatabaseVersionStore(AsyncMemoryDatabase())


def a_version(
    version_id: str = "v1",
    name: str = "greeting",
    prompt_type: str = "system",
    version: str = "1.0.0",
) -> PromptVersion:
    """A version with every optional field populated, so nothing rides for free."""
    return PromptVersion(
        version_id=version_id,
        name=name,
        prompt_type=prompt_type,
        version=version,
        template="Hello {{name}}!",
        defaults={"name": "World"},
        validation={"level": "strict"},
        metadata={"author": "alice"},
        created_at=datetime.now(UTC),
        created_by="alice",
        parent_version=None,
        tags=["production"],
        status=VersionStatus.PRODUCTION,
    )


def an_experiment(experiment_id: str = "e1", name: str = "greeting") -> PromptExperiment:
    """A two-variant experiment."""
    return PromptExperiment(
        experiment_id=experiment_id,
        name=name,
        prompt_type="system",
        variants=[PromptVariant("1.0.0", 0.5, "Control"), PromptVariant("1.0.1", 0.5, "Treatment")],
        traffic_split={"1.0.0": 0.5, "1.0.1": 0.5},
    )


# ===== The protocols =====


@pytest.mark.parametrize("protocol", [VersionStore, ExperimentStore, MetricsStore, VersioningStore])
def test_both_shipped_stores_answer_for_every_protocol(
    store: VersioningStore, protocol: type
) -> None:
    """Three protocols, segregated by manager, satisfied by one object.

    Segregated so a manager declares only what it uses --- a test can hand
    ``VersionManager`` a five-method object without implementing seventeen ---
    and satisfied together so the library still passes one store to all three.
    """
    assert isinstance(store, protocol)


def test_a_store_that_cannot_answer_is_refused_by_name() -> None:
    """The old parameter, rejected with the list of what it is missing.

    ``set``/``append``/``delete`` is the vocabulary this replaced. An object
    written for it reaches a manager's constructor and fails there, rather than
    at the first write it would have dropped in silence.
    """

    class DuckTypedBackend:
        async def set(self, key: str, value: object) -> None: ...

        async def append(self, key: str, value: object) -> None: ...

        async def delete(self, key: str) -> None: ...

    with pytest.raises(TypeError) as caught:
        require_store(DuckTypedBackend(), VersioningStore, holder="VersionedPromptLibrary")

    message = str(caught.value)
    assert "VersionedPromptLibrary needs a VersioningStore" in message
    assert "save_version" in message and "append_event" in message and "load_names" in message
    assert "InMemoryVersionStore()" in message


def test_a_composed_protocol_names_every_member_it_is_missing() -> None:
    """``VersioningStore``'s own class body is empty; the message is not.

    It inherits all seventeen members and defines none of them, so a check
    that read one class body would refuse a store while reporting it missing
    *nothing*. The message walks the MRO instead. The case is real rather than
    contrived: a consumer who wrote a store for versions and experiments and
    stopped there is told which five methods remain.
    """

    class VersionsAndExperimentsOnly(InMemoryVersionStore):
        save_metrics = None  # type: ignore[assignment]
        load_metrics = None  # type: ignore[assignment]
        append_event = None  # type: ignore[assignment]
        load_events = None  # type: ignore[assignment]
        delete_metrics = None  # type: ignore[assignment]

    partial = VersionsAndExperimentsOnly()
    assert isinstance(partial, VersionStore) and isinstance(partial, ExperimentStore)

    with pytest.raises(TypeError) as caught:
        require_store(partial, VersioningStore, holder="VersionedPromptLibrary")

    message = str(caught.value)
    assert (
        "missing append_event, delete_metrics, load_events, load_metrics, save_metrics" in message
    )
    assert "save_version" not in message, "it has that one; only the gaps are named"


def test_a_store_that_satisfies_the_protocol_is_accepted(store: VersioningStore) -> None:
    """The positive control for the check above."""
    require_store(store, VersioningStore, holder="VersionedPromptLibrary")


# ===== Versions =====


@pytest.mark.asyncio
async def test_a_version_round_trips(store: VersioningStore) -> None:
    """Every field, including the ones that have to survive serialization."""
    version = a_version()
    await store.save_version(version)

    assert await store.load_version("v1") == version


@pytest.mark.asyncio
async def test_a_load_hands_back_a_value_not_a_handle(store: VersioningStore) -> None:
    """The one difference the two implementations could have had, and do not.

    A dictionary-backed store can cheaply hand out the object it holds. Doing
    so would make a consumer's in-place mutation stick in development and
    vanish in production --- the difference nobody discovers until it is
    expensive. ``AsyncMemoryDatabase`` deep-copies for the same reason, so
    both halves of this protocol agree with the rest of the workspace.
    """
    await store.save_version(a_version())

    loaded = await store.load_version("v1")
    assert loaded is not None
    loaded.tags.append("mutated")
    loaded.template = "changed"
    loaded.defaults["name"] = "changed"

    again = await store.load_version("v1")
    assert again is not None
    assert again.tags == ["production"]
    assert again.template == "Hello {{name}}!"
    assert again.defaults == {"name": "World"}


@pytest.mark.asyncio
async def test_versions_are_found_by_name_and_type(store: VersioningStore) -> None:
    """A prompt's versions, and nobody else's."""
    await store.save_version(a_version("v1", version="1.0.0"))
    await store.save_version(a_version("v2", version="1.0.1"))
    await store.save_version(a_version("v3", prompt_type="user"))
    await store.save_version(a_version("v4", name="signoff"))

    found = await store.load_versions("greeting", "system")

    assert sorted(v.version_id for v in found) == ["v1", "v2"]


@pytest.mark.asyncio
async def test_names_are_derived_from_the_versions(store: VersioningStore) -> None:
    """No index to drift, so a name cannot outlive its last version.

    The index that used to answer this was a second record of which names
    exist, and it had already drifted from the versions it indexed --- keys
    outliving their last version, and a split on the first colon that dropped
    any name containing one.
    """
    await store.save_version(a_version("v1"))
    await store.save_version(a_version("v2", name="sign:off:now"))
    await store.save_version(a_version("v3", name="farewell", prompt_type="user"))

    assert await store.load_names("system") == {"greeting", "sign:off:now"}
    assert await store.load_names("user") == {"farewell"}

    await store.delete_version("v2")
    assert await store.load_names("system") == {"greeting"}


@pytest.mark.asyncio
async def test_deleting_a_version_reports_whether_there_was_one(
    store: VersioningStore,
) -> None:
    """The answer comes from the store, which is the whole point.

    It used to come from an instance dictionary that a real backend had never
    been told about, so it said ``True`` for rows the backend had never held.
    """
    await store.save_version(a_version())

    assert await store.delete_version("v1") is True
    assert await store.delete_version("v1") is False
    assert await store.delete_version("never-existed") is False
    assert await store.load_version("v1") is None


@pytest.mark.asyncio
async def test_saving_the_same_id_twice_replaces(store: VersioningStore) -> None:
    """One version per id, so an update is not a second row."""
    await store.save_version(a_version())
    updated = a_version()
    updated.template = "Goodbye {{name}}!"
    await store.save_version(updated)

    found = await store.load_versions("greeting", "system")

    assert len(found) == 1
    assert found[0].template == "Goodbye {{name}}!"


# ===== Experiments and assignments =====


@pytest.mark.asyncio
async def test_an_experiment_round_trips(store: VersioningStore) -> None:
    """Variants are nested dataclasses, so they are the part that could be lost."""
    experiment = an_experiment()
    await store.save_experiment(experiment)

    loaded = await store.load_experiment("e1")

    assert loaded == experiment
    assert [v.description for v in loaded.variants] == ["Control", "Treatment"]


@pytest.mark.asyncio
async def test_experiments_match_every_filter_given(store: VersioningStore) -> None:
    """Each filter narrows; ``None`` does not filter."""
    running = an_experiment("e1")
    paused = an_experiment("e2", name="signoff")
    paused.status = "paused"
    await store.save_experiment(running)
    await store.save_experiment(paused)

    assert len(await store.load_experiments()) == 2
    assert [e.experiment_id for e in await store.load_experiments(status="paused")] == ["e2"]
    assert [e.experiment_id for e in await store.load_experiments(name="greeting")] == ["e1"]
    assert await store.load_experiments(name="greeting", status="paused") == []


@pytest.mark.asyncio
async def test_an_assignment_is_a_record_rather_than_a_bare_string(
    store: VersioningStore,
) -> None:
    """The shape the old vocabulary could not express.

    An assignment was persisted as a bare ``str`` under a composed key, which
    is not a record --- so no backend could hold it even if ``set`` had
    existed. It is a record now, found by its fields, which is also why a
    ``user_id`` containing a colon is an ordinary user id rather than a
    parsing bug waiting to happen.
    """
    await store.save_assignment("e1", "user:with:colons", "1.0.0")
    await store.save_assignment("e1", "u2", "1.0.1")
    await store.save_assignment("e2", "u2", "2.0.0")

    assert await store.load_assignment("e1", "user:with:colons") == "1.0.0"
    assert await store.load_assignment("e1", "nobody") is None
    assert await store.load_assignments("e1") == {"user:with:colons": "1.0.0", "u2": "1.0.1"}
    assert await store.load_assignments("e2") == {"u2": "2.0.0"}


@pytest.mark.asyncio
async def test_deleting_an_experiment_takes_its_assignments(store: VersioningStore) -> None:
    """An assignment outliving its experiment is a row nothing can interpret."""
    await store.save_experiment(an_experiment("e1"))
    await store.save_experiment(an_experiment("e2"))
    await store.save_assignment("e1", "u1", "1.0.0")
    await store.save_assignment("e2", "u1", "1.0.1")

    assert await store.delete_experiment("e1") is True

    assert await store.load_experiment("e1") is None
    assert await store.load_assignments("e1") == {}
    assert await store.load_assignments("e2") == {"u1": "1.0.1"}
    assert await store.delete_experiment("e1") is False


# ===== Metrics and events =====


@pytest.mark.asyncio
async def test_metrics_round_trip(store: VersioningStore) -> None:
    """``to_dict`` emits computed properties; ``from_dict`` recomputes them."""
    metrics = PromptMetrics(
        version_id="v1",
        total_uses=3,
        success_count=2,
        error_count=1,
        total_response_time=1.5,
        total_tokens=300,
        user_ratings=[4.0, 5.0],
        last_used=datetime.now(UTC),
    )
    await store.save_metrics(metrics)

    loaded = await store.load_metrics("v1")

    assert loaded == metrics
    assert loaded.avg_rating == 4.5
    assert await store.load_metrics("never-used") is None


@pytest.mark.asyncio
async def test_events_accumulate_rather_than_replace(store: VersioningStore) -> None:
    """The other shape the old vocabulary could not express.

    Events were ``append``-ed to a list under one key. A record store holds one
    record per event, so two identical events are two events and not one
    overwriting the other.
    """
    for _ in range(3):
        await store.append_event(MetricEvent(version_id="v1", success=True, tokens=10))
    await store.append_event(MetricEvent(version_id="v2", success=False))

    assert len(await store.load_events("v1")) == 3
    assert len(await store.load_events("v2")) == 1
    assert await store.load_events("never-used") == []


@pytest.mark.asyncio
async def test_an_aggregate_is_not_an_event(store: VersioningStore) -> None:
    """Both carry a ``version_id``, so only a discriminator keeps them apart."""
    await store.save_metrics(PromptMetrics(version_id="v1", total_uses=7))
    await store.append_event(MetricEvent(version_id="v1", success=True))

    events = await store.load_events("v1")

    assert len(events) == 1
    assert isinstance(events[0], MetricEvent)


@pytest.mark.asyncio
async def test_deleting_metrics_takes_the_events(store: VersioningStore) -> None:
    """An aggregate cannot outlive the events it was computed from."""
    await store.save_metrics(PromptMetrics(version_id="v1", total_uses=2))
    await store.append_event(MetricEvent(version_id="v1", success=True))
    await store.append_event(MetricEvent(version_id="v2", success=True))

    assert await store.delete_metrics("v1") is True

    assert await store.load_metrics("v1") is None
    assert await store.load_events("v1") == []
    assert len(await store.load_events("v2")) == 1
    assert await store.delete_metrics("v1") is False


@pytest.mark.asyncio
async def test_events_without_an_aggregate_are_still_something_to_delete(
    store: VersioningStore,
) -> None:
    """``False`` means nothing was there, not that one of the two was missing."""
    await store.append_event(MetricEvent(version_id="v1", success=True))

    assert await store.delete_metrics("v1") is True
    assert await store.delete_metrics("v1") is False


# ===== What the protocol check accepts and refuses =====


def test_a_store_that_forwards_its_methods_is_accepted() -> None:
    """A wrapper that delegates through ``__getattr__`` is a store.

    ``runtime_checkable`` decides with ``getattr_static``, which cannot see a
    forwarder, so the object every consumer eventually writes --- a logging,
    tracing or metrics wrapper around a real store --- was refused. It was
    refused with a message naming *nothing*, because the members it is asked
    about are all reachable: ``isinstance`` said no and ``missing`` came back
    empty.
    """

    class ForwardingStore:
        """Everything this does not define, the wrapped store answers."""

        def __init__(self, inner: VersioningStore) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    require_store(
        ForwardingStore(InMemoryVersionStore()),
        VersioningStore,
        holder="VersionedPromptLibrary",
    )


def test_a_member_that_is_not_callable_is_refused() -> None:
    """A store is its methods, and an attribute that cannot be called is not one.

    ``runtime_checkable`` treats only a ``None``-valued member as absent, so an
    attribute of any other type satisfied it and the failure arrived later, at
    the first call, as ``'int' object is not callable``.
    """

    class NotQuiteAStore(InMemoryVersionStore):
        save_version = 3  # type: ignore[assignment]

    with pytest.raises(TypeError) as caught:
        require_store(NotQuiteAStore(), VersioningStore, holder="VersionedPromptLibrary")

    assert "missing save_version" in str(caught.value)


def test_the_check_hands_back_the_store_it_accepted() -> None:
    """So a constructor is one statement rather than three."""
    store = InMemoryVersionStore()

    assert require_store(store, VersioningStore, holder="VersionedPromptLibrary") is store


# ===== A save copies too =====


@pytest.mark.asyncio
async def test_a_save_copies_what_it_was_given(store: VersioningStore) -> None:
    """The other half of the value/handle promise, and the untested half.

    ``test_a_load_hands_back_a_value_not_a_handle`` mutates what a load
    returned. Nothing mutated what a *save* was handed, so an implementation
    that stored the caller's object would keep the whole suite green while
    reintroducing exactly the divergence the copying exists to prevent:
    ``v = await create_version(...)`` followed by ``v.tags.append(...)`` would
    change an in-memory store and not a database one.
    """
    version = a_version()
    await store.save_version(version)

    version.tags.append("mutated")
    version.template = "changed"
    version.defaults["name"] = "changed"

    again = await store.load_version("v1")
    assert again is not None
    assert again.tags == ["production"]
    assert again.template == "Hello {{name}}!"
    assert again.defaults == {"name": "World"}


@pytest.mark.asyncio
async def test_saving_an_experiment_copies_what_it_was_given(store: VersioningStore) -> None:
    """The same promise, over the nested structure that carries the most."""
    experiment = an_experiment()
    await store.save_experiment(experiment)

    experiment.traffic_split["1.0.0"] = 0.9
    experiment.variants.clear()

    again = await store.load_experiment("e1")
    assert again is not None
    assert again.traffic_split == {"1.0.0": 0.5, "1.0.1": 0.5}
    assert len(again.variants) == 2


# ===== Assignments are found, not parsed =====


@pytest.mark.asyncio
async def test_two_assignments_cannot_collide_through_their_key(store: VersioningStore) -> None:
    """``(a:b, c)`` and ``(a, b:c)`` are two assignments, not one.

    The record id composes both halves with a colon between them, so these two
    pairs spell the same string. ``load_assignments`` and ``delete_experiment``
    were already immune --- they search by field --- but ``load_assignment``
    composed the key and read it straight back, which made one of these two
    overwrite the other and answer for it.

    The in-memory store nests one dictionary inside another and could never
    collide, so this is a divergence between the two halves of one protocol as
    much as it is a bug in either.
    """
    await store.save_assignment("a:b", "c", "1.0.0")
    await store.save_assignment("a", "b:c", "2.0.0")

    assert await store.load_assignment("a:b", "c") == "1.0.0"
    assert await store.load_assignment("a", "b:c") == "2.0.0"
    assert await store.load_assignments("a:b") == {"c": "1.0.0"}
    assert await store.load_assignments("a") == {"b:c": "2.0.0"}


# ===== Events are a stream, so reading one is bounded =====


@pytest.mark.asyncio
async def test_events_come_back_newest_first(store: VersioningStore) -> None:
    """An append-only stream has an order, and it is the one a reader wants.

    The protocol used to promise no order at all, which made a limit
    meaningless: the only way to get the most recent five was to materialize
    every event ever recorded and sort them in the caller.
    """
    for minute in range(5):
        await store.append_event(
            MetricEvent(
                version_id="v1",
                timestamp=datetime(2026, 1, 1, 0, minute, tzinfo=UTC),
                success=True,
                tokens=minute,
            )
        )

    events = await store.load_events("v1")

    assert [event.tokens for event in events] == [4, 3, 2, 1, 0]


@pytest.mark.asyncio
async def test_a_limit_takes_the_most_recent(store: VersioningStore) -> None:
    """And it is the store that applies it, so the rest is never loaded."""
    for minute in range(10):
        await store.append_event(
            MetricEvent(
                version_id="v1",
                timestamp=datetime(2026, 1, 1, 0, minute, tzinfo=UTC),
                success=True,
                tokens=minute,
            )
        )

    assert [event.tokens for event in await store.load_events("v1", limit=3)] == [9, 8, 7]
    assert await store.load_events("v1", limit=0) == []
    assert len(await store.load_events("v1", limit=99)) == 10


@pytest.mark.asyncio
async def test_deleting_metrics_takes_every_event_not_a_page_of_them(
    store: VersioningStore,
) -> None:
    """Enough events that a batched delete is not one call, and none survive."""
    for minute in range(25):
        await store.append_event(
            MetricEvent(
                version_id="v1",
                timestamp=datetime(2026, 1, 1, 0, minute, tzinfo=UTC),
                success=True,
            )
        )

    assert await store.delete_metrics("v1") is True

    assert await store.load_events("v1") == []


# ===== An event is folded into its aggregate atomically =====


@pytest.mark.asyncio
async def test_recording_an_event_appends_it_and_folds_it(store: VersioningStore) -> None:
    """One verb, because the two halves have to happen together.

    A caller that appends and then separately reads, folds and writes the
    aggregate has a suspension point in the middle of a read-modify-write.
    Against a real transport that is where a concurrent increment is lost.
    """
    first = await store.record_event(
        MetricEvent(version_id="v1", success=True, response_time=0.5, tokens=10, user_rating=4.0)
    )

    assert first.total_uses == 1
    assert first.success_count == 1
    assert first.user_ratings == [4.0]

    second = await store.record_event(MetricEvent(version_id="v1", success=False, tokens=5))

    assert second.total_uses == 2
    assert second.success_count == 1
    assert second.error_count == 1
    assert second.total_tokens == 15
    assert len(await store.load_events("v1")) == 2
    assert await store.load_metrics("v1") == second


@pytest.mark.asyncio
async def test_load_names_hands_back_a_set_the_caller_may_mutate(store: VersioningStore) -> None:
    """``VersionedPromptLibrary.list_system_prompts`` calls ``.update()`` on it.

    It merges the base library's names into whatever the manager returned, so a
    store handing back a set it keeps would have that set grow every listing.
    Both shipped stores build a fresh one; this is the protocol's half of that.
    """
    await store.save_version(a_version())

    names = await store.load_names("system")
    names.add("mutated")

    assert await store.load_names("system") == {"greeting"}
