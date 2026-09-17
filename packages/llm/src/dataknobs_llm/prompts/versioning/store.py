# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Persistence for the versioning layer.

Three protocols, one per manager, and two implementations that each satisfy
all three --- so one object serves :class:`~.version_manager.VersionManager`,
:class:`~.ab_testing.ABTestManager` and :class:`~.metrics.MetricsCollector`
exactly as one ``storage`` argument used to.

What stood here before was not persistence. Each manager took a
``storage: Any`` and duck-typed it for ``set``, ``append`` and ``delete`` --- a
key-value vocabulary, where the seven DataKnobs backends are a *record* store.
Against any of them ``set`` and ``append`` matched nothing, so every write was
dropped in silence, while ``delete`` matched **by name** and fired, against ids
that had therefore never been written. Every read came from an instance
dictionary the writes shadowed and nothing ever replayed. Handing that layer a
real database wrote nothing, read nothing, and issued live deletes.

It is also why none of these coroutines ever suspended: a method that never
reads has nothing to await. The ``async`` was decorative, and one of this
module's tests is the ``send(None)`` probe that says it no longer is.

**The dictionaries moved.** ``_versions``, ``_version_index``,
``_experiments``, ``_user_assignments``, ``_metrics`` and ``_events`` were six
instance dictionaries spread across three managers. Five of them are now
:class:`InMemoryVersionStore`, which is the default and holds what they held.
The sixth, ``_version_index``, is not carried over: which names exist is
derivable from the versions themselves, and a second record of it is a thing
to keep in step --- it had already drifted from the versions it indexed in two
ways at once.

**Both implementations copy.** A store hands back a value, not a handle:
mutate what :meth:`VersionStore.load_version` returned and the store is
unchanged until you save it. The in-memory one is the implementation that
could cheaply do otherwise and deliberately does not, because an
implementation that aliased would work in development and lose writes in
production --- the one difference between these two classes that a consumer
could not discover except in the place it hurts. ``AsyncMemoryDatabase``, this
workspace's other in-memory reference implementation, deep-copies for the same
reason.

**Errors propagate.** Neither implementation wraps what its backend raises:
``dataknobs_data`` raises from a typed hierarchy already, and a wrapper on one
implementation but not the other is a difference between two halves of one
protocol.

**Nothing here owns a database.** :class:`DatabaseVersionStore` is *handed* an
``AsyncDatabase``, so by the ownership convention it does not close it, and
this layer grows no teardown method to forget to call. Close the database you
opened; the store holds no other resource.
"""

from __future__ import annotations

import copy
import uuid
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from dataknobs_common.records import Record

from .types import (
    MetricEvent,
    PromptExperiment,
    PromptMetrics,
    PromptVersion,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, as elsewhere in this package
    from dataknobs_data.database import AsyncDatabase
    from dataknobs_data.query import Query

__all__ = [
    "DatabaseVersionStore",
    "ExperimentStore",
    "InMemoryVersionStore",
    "MetricsStore",
    "VersionStore",
    "VersioningStore",
    "require_store",
]


@runtime_checkable
class VersionStore(Protocol):
    """Where :class:`~.version_manager.VersionManager` keeps its versions.

    Four reads and two writes, and nothing about semantic versions: ordering,
    ``latest`` resolution, tag and status filtering are the manager's policy
    and stay there. A store answers what it holds.
    """

    async def save_version(self, version: PromptVersion) -> None:
        """Store ``version``, replacing any version with the same id."""
        ...

    async def load_version(self, version_id: str) -> PromptVersion | None:
        """Return the version with this id, or ``None``."""
        ...

    async def load_versions(self, name: str, prompt_type: str) -> list[PromptVersion]:
        """Return every version of one prompt, in no particular order."""
        ...

    async def load_names(self, prompt_type: str) -> set[str]:
        """Return the names holding at least one version of ``prompt_type``.

        Derived from the versions themselves, so a name whose last version was
        deleted is absent and a listing built from this cannot name a prompt
        the getters then answer ``None`` for.
        """
        ...

    async def delete_version(self, version_id: str) -> bool:
        """Delete one version. ``False`` if there was nothing to delete."""
        ...


@runtime_checkable
class ExperimentStore(Protocol):
    """Where :class:`~.ab_testing.ABTestManager` keeps experiments and assignments.

    An assignment is a user's variant for one experiment, and it is the
    entity the previous vocabulary could not express: it was persisted as a
    bare string under a composed key, which is not a record and therefore not
    something any of the seven backends could hold.
    """

    async def save_experiment(self, experiment: PromptExperiment) -> None:
        """Store ``experiment``, replacing any experiment with the same id."""
        ...

    async def load_experiment(self, experiment_id: str) -> PromptExperiment | None:
        """Return the experiment with this id, or ``None``."""
        ...

    async def load_experiments(
        self,
        name: str | None = None,
        prompt_type: str | None = None,
        status: str | None = None,
    ) -> list[PromptExperiment]:
        """Return experiments matching every filter given. ``None`` does not filter."""
        ...

    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment **and its assignments**.

        Both, because an assignment outliving its experiment is a row nothing
        can interpret, and the alternative --- making the caller enumerate
        assignments first --- is a round trip per user for a correctness
        property the store can guarantee.

        Returns:
            ``False`` if there was no such experiment.
        """
        ...

    async def save_assignment(self, experiment_id: str, user_id: str, version: str) -> None:
        """Record that ``user_id`` is on ``version`` for this experiment."""
        ...

    async def load_assignment(self, experiment_id: str, user_id: str) -> str | None:
        """Return one user's assigned version, or ``None``."""
        ...

    async def load_assignments(self, experiment_id: str) -> dict[str, str]:
        """Return every assignment for one experiment, as ``{user_id: version}``."""
        ...


@runtime_checkable
class MetricsStore(Protocol):
    """Where :class:`~.metrics.MetricsCollector` keeps aggregates and events.

    Two entities with one key each: an aggregate per version, replaced as it
    is recomputed, and an append-only event stream. The stream is the other
    entity the previous vocabulary could not express --- it was an ``append``
    to a list under a single key, where a record store holds one record per
    event.
    """

    async def save_metrics(self, metrics: PromptMetrics) -> None:
        """Store the aggregate for one version, replacing the previous one."""
        ...

    async def load_metrics(self, version_id: str) -> PromptMetrics | None:
        """Return the aggregate for one version, or ``None`` if it has none."""
        ...

    async def append_event(self, event: MetricEvent) -> None:
        """Add one event. Events are never replaced; two identical ones are two."""
        ...

    async def load_events(self, version_id: str) -> list[MetricEvent]:
        """Return every event for one version, in no particular order."""
        ...

    async def delete_metrics(self, version_id: str) -> bool:
        """Delete a version's aggregate **and** its events.

        Returns:
            ``False`` if there was neither to delete.
        """
        ...


@runtime_checkable
class VersioningStore(VersionStore, ExperimentStore, MetricsStore, Protocol):
    """All three at once --- what :class:`.VersionedPromptLibrary` needs.

    The library owns one store and hands the same object to all three
    managers, so it is the one caller that needs a name for the whole surface.
    A store written for a single manager does not need this: each manager
    declares only what it uses, which is why there are three protocols and not
    one.
    """


def require_store(store: object, protocol: type, *, holder: str) -> None:
    """Raise unless ``store`` satisfies ``protocol``, naming what is missing.

    Called from each manager's constructor, so that a store which cannot
    answer fails where the caller is still looking. The object this most often
    catches is the one that used to be passed: a duck-typed
    ``set``/``append``/``delete`` backend, whose failure mode was to write
    nothing at all and report success.

    Args:
        store: The candidate store.
        protocol: The runtime-checkable protocol it must satisfy.
        holder: The class name to quote in the message.

    Raises:
        TypeError: If any of the protocol's members is missing.
    """
    if isinstance(store, protocol):
        return
    # Walk the MRO rather than one class body, so a protocol composed of
    # others -- VersioningStore -- names every member it is missing rather
    # than none of them, its own body being empty.
    required = sorted(
        {name for klass in protocol.__mro__ for name in vars(klass) if not name.startswith("_")}
    )
    missing = [name for name in required if not callable(getattr(store, name, None))]
    raise TypeError(
        f"{holder} needs a {protocol.__name__}, and {type(store).__name__} is missing "
        f"{', '.join(missing)}. Pass InMemoryVersionStore() for in-memory behaviour, or "
        f"DatabaseVersionStore(db) for any dataknobs_data AsyncDatabase. A backend "
        f"duck-typed for set/append/delete is the previous parameter, not this one"
    )


class InMemoryVersionStore:
    """The default store, holding everything in dictionaries.

    Substituted for the ``storage=None`` default, so a manager constructed
    with no arguments behaves as it did --- with the aliasing removed: what
    comes out of a load is a copy, as it would be from any real backend.

    Example:
        ```python
        store = InMemoryVersionStore()
        manager = VersionManager(store)
        collector = MetricsCollector(store)  # one object, either protocol
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty store."""
        self._versions: dict[str, PromptVersion] = {}
        self._experiments: dict[str, PromptExperiment] = {}
        self._assignments: dict[str, dict[str, str]] = {}
        self._metrics: dict[str, PromptMetrics] = {}
        self._events: dict[str, list[MetricEvent]] = {}

    # ===== VersionStore =====

    async def save_version(self, version: PromptVersion) -> None:
        """Store ``version``, replacing any version with the same id."""
        self._versions[version.version_id] = copy.deepcopy(version)

    async def load_version(self, version_id: str) -> PromptVersion | None:
        """Return the version with this id, or ``None``."""
        version = self._versions.get(version_id)
        return copy.deepcopy(version) if version is not None else None

    async def load_versions(self, name: str, prompt_type: str) -> list[PromptVersion]:
        """Return every version of one prompt, in no particular order."""
        return [
            copy.deepcopy(version)
            for version in self._versions.values()
            if version.name == name and version.prompt_type == prompt_type
        ]

    async def load_names(self, prompt_type: str) -> set[str]:
        """Return the names holding at least one version of ``prompt_type``."""
        return {
            version.name
            for version in self._versions.values()
            if version.prompt_type == prompt_type
        }

    async def delete_version(self, version_id: str) -> bool:
        """Delete one version. ``False`` if there was nothing to delete."""
        return self._versions.pop(version_id, None) is not None

    # ===== ExperimentStore =====

    async def save_experiment(self, experiment: PromptExperiment) -> None:
        """Store ``experiment``, replacing any experiment with the same id."""
        self._experiments[experiment.experiment_id] = copy.deepcopy(experiment)

    async def load_experiment(self, experiment_id: str) -> PromptExperiment | None:
        """Return the experiment with this id, or ``None``."""
        experiment = self._experiments.get(experiment_id)
        return copy.deepcopy(experiment) if experiment is not None else None

    async def load_experiments(
        self,
        name: str | None = None,
        prompt_type: str | None = None,
        status: str | None = None,
    ) -> list[PromptExperiment]:
        """Return experiments matching every filter given."""
        return [
            copy.deepcopy(experiment)
            for experiment in self._experiments.values()
            if (name is None or experiment.name == name)
            and (prompt_type is None or experiment.prompt_type == prompt_type)
            and (status is None or experiment.status == status)
        ]

    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and its assignments."""
        self._assignments.pop(experiment_id, None)
        return self._experiments.pop(experiment_id, None) is not None

    async def save_assignment(self, experiment_id: str, user_id: str, version: str) -> None:
        """Record that ``user_id`` is on ``version`` for this experiment."""
        self._assignments.setdefault(experiment_id, {})[user_id] = version

    async def load_assignment(self, experiment_id: str, user_id: str) -> str | None:
        """Return one user's assigned version, or ``None``."""
        return self._assignments.get(experiment_id, {}).get(user_id)

    async def load_assignments(self, experiment_id: str) -> dict[str, str]:
        """Return every assignment for one experiment."""
        return dict(self._assignments.get(experiment_id, {}))

    # ===== MetricsStore =====

    async def save_metrics(self, metrics: PromptMetrics) -> None:
        """Store the aggregate for one version, replacing the previous one."""
        self._metrics[metrics.version_id] = copy.deepcopy(metrics)

    async def load_metrics(self, version_id: str) -> PromptMetrics | None:
        """Return the aggregate for one version, or ``None``."""
        metrics = self._metrics.get(version_id)
        return copy.deepcopy(metrics) if metrics is not None else None

    async def append_event(self, event: MetricEvent) -> None:
        """Add one event."""
        self._events.setdefault(event.version_id, []).append(copy.deepcopy(event))

    async def load_events(self, version_id: str) -> list[MetricEvent]:
        """Return every event for one version."""
        return [copy.deepcopy(event) for event in self._events.get(version_id, [])]

    async def delete_metrics(self, version_id: str) -> bool:
        """Delete a version's aggregate and its events."""
        had_metrics = self._metrics.pop(version_id, None) is not None
        had_events = self._events.pop(version_id, None) is not None
        return had_metrics or had_events


# Record kinds. One field distinguishes the five entities in a store they all
# share, which is what lets a search for events not find the aggregate that
# carries the same ``version_id``.
_VERSION = "version"
_EXPERIMENT = "experiment"
_ASSIGNMENT = "assignment"
_METRICS = "metrics"
_EVENT = "event"

_KIND_FIELD = "kind"


class DatabaseVersionStore:
    """A store over any ``dataknobs_data`` ``AsyncDatabase``.

    Brings all seven backends --- memory, file, SQLite, PostgreSQL, S3, DuckDB,
    Elasticsearch --- to a layer that previously persisted to none of them.
    The vocabulary translation lives here and nowhere else: the managers speak
    versions, experiments and events, and only this class knows about
    ``Record``, ``upsert`` and ``Query``.

    Each entity is one record, keyed ``{kind}:{id}`` and carrying a ``kind``
    field so that a search can tell them apart. An assignment is keyed
    ``assignment:{experiment_id}:{user_id}`` but is never read back by parsing
    that key --- it is found by its fields, so a ``user_id`` containing a colon
    is an ordinary user id rather than a bug.

    The database is the caller's. This class does not close it.

    Example:
        ```python
        from dataknobs_data import async_database_factory
        from dataknobs_llm.prompts import VersionedPromptLibrary
        from dataknobs_llm.prompts.versioning import DatabaseVersionStore

        db = async_database_factory.create(backend="sqlite", path="./prompts.db")
        await db.connect()
        library = VersionedPromptLibrary(store=DatabaseVersionStore(db))

        await library.create_version(
            name="greeting", prompt_type="system",
            template="Hello {{name}}!", version="1.0.0",
        )
        # A second library over the same database reads it back, including
        # the next time this program runs.
        await db.close()
        ```
    """

    def __init__(self, database: AsyncDatabase) -> None:
        """Initialize the store.

        Args:
            database: Any ``dataknobs_data`` ``AsyncDatabase``. Not owned: the
                caller opened it and the caller closes it.
        """
        self._db = database

    # ===== VersionStore =====

    async def save_version(self, version: PromptVersion) -> None:
        """Store ``version``, replacing any version with the same id."""
        await self._save(_VERSION, version.version_id, version.to_dict())

    async def load_version(self, version_id: str) -> PromptVersion | None:
        """Return the version with this id, or ``None``."""
        data = await self._read(_VERSION, version_id)
        return PromptVersion.from_dict(data) if data is not None else None

    async def load_versions(self, name: str, prompt_type: str) -> list[PromptVersion]:
        """Return every version of one prompt, in no particular order."""
        records = await self._search(_VERSION, name=name, prompt_type=prompt_type)
        return [PromptVersion.from_dict(data) for data in records]

    async def load_names(self, prompt_type: str) -> set[str]:
        """Return the names holding at least one version of ``prompt_type``."""
        records = await self._search(_VERSION, prompt_type=prompt_type)
        return {str(data["name"]) for data in records}

    async def delete_version(self, version_id: str) -> bool:
        """Delete one version. ``False`` if there was nothing to delete."""
        return await self._db.delete(_key(_VERSION, version_id))

    # ===== ExperimentStore =====

    async def save_experiment(self, experiment: PromptExperiment) -> None:
        """Store ``experiment``, replacing any experiment with the same id."""
        await self._save(_EXPERIMENT, experiment.experiment_id, experiment.to_dict())

    async def load_experiment(self, experiment_id: str) -> PromptExperiment | None:
        """Return the experiment with this id, or ``None``."""
        data = await self._read(_EXPERIMENT, experiment_id)
        return PromptExperiment.from_dict(data) if data is not None else None

    async def load_experiments(
        self,
        name: str | None = None,
        prompt_type: str | None = None,
        status: str | None = None,
    ) -> list[PromptExperiment]:
        """Return experiments matching every filter given."""
        filters = {"name": name, "prompt_type": prompt_type, "status": status}
        records = await self._search(
            _EXPERIMENT, **{k: v for k, v in filters.items() if v is not None}
        )
        return [PromptExperiment.from_dict(data) for data in records]

    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and its assignments."""
        deleted = await self._db.delete(_key(_EXPERIMENT, experiment_id))
        for data in await self._search(_ASSIGNMENT, experiment_id=experiment_id):
            await self._db.delete(_assignment_key(experiment_id, str(data["user_id"])))
        return deleted

    async def save_assignment(self, experiment_id: str, user_id: str, version: str) -> None:
        """Record that ``user_id`` is on ``version`` for this experiment."""
        await self._db.upsert(
            _assignment_key(experiment_id, user_id),
            Record(
                {
                    _KIND_FIELD: _ASSIGNMENT,
                    "experiment_id": experiment_id,
                    "user_id": user_id,
                    "version": version,
                }
            ),
        )

    async def load_assignment(self, experiment_id: str, user_id: str) -> str | None:
        """Return one user's assigned version, or ``None``."""
        record = await self._db.read(_assignment_key(experiment_id, user_id))
        return None if record is None else str(_payload(record)["version"])

    async def load_assignments(self, experiment_id: str) -> dict[str, str]:
        """Return every assignment for one experiment."""
        records = await self._search(_ASSIGNMENT, experiment_id=experiment_id)
        return {str(data["user_id"]): str(data["version"]) for data in records}

    # ===== MetricsStore =====

    async def save_metrics(self, metrics: PromptMetrics) -> None:
        """Store the aggregate for one version, replacing the previous one."""
        await self._save(_METRICS, metrics.version_id, metrics.to_dict())

    async def load_metrics(self, version_id: str) -> PromptMetrics | None:
        """Return the aggregate for one version, or ``None``."""
        data = await self._read(_METRICS, version_id)
        return PromptMetrics.from_dict(data) if data is not None else None

    async def append_event(self, event: MetricEvent) -> None:
        """Add one event, under an id of its own."""
        await self._save(_EVENT, str(uuid.uuid4()), event.to_dict())

    async def load_events(self, version_id: str) -> list[MetricEvent]:
        """Return every event for one version, in no particular order."""
        records = await self._search(_EVENT, version_id=version_id)
        return [MetricEvent.from_dict(data) for data in records]

    async def delete_metrics(self, version_id: str) -> bool:
        """Delete a version's aggregate and its events."""
        deleted = await self._db.delete(_key(_METRICS, version_id))
        events = await self._db.search(self._query(_EVENT, version_id=version_id))
        for record in events:
            if record.id:
                await self._db.delete(record.id)
        return deleted or bool(events)

    # ===== Helper Methods =====

    async def _save(self, kind: str, entity_id: str, data: dict[str, Any]) -> None:
        """Write one entity's dictionary as a record of ``kind``."""
        await self._db.upsert(_key(kind, entity_id), Record({**data, _KIND_FIELD: kind}))

    async def _read(self, kind: str, entity_id: str) -> dict[str, Any] | None:
        """Read one entity's dictionary back, or ``None``."""
        record = await self._db.read(_key(kind, entity_id))
        return None if record is None else _payload(record)

    async def _search(self, kind: str, **fields: Any) -> list[dict[str, Any]]:
        """Read back every entity of ``kind`` whose fields match."""
        records = await self._db.search(self._query(kind, **fields))
        return [_payload(record) for record in records]

    def _query(self, kind: str, **fields: Any) -> Query:
        """Build an equality query over ``kind`` and the given fields.

        The one deferred import, following this package's existing backend
        adapter: ``dataknobs_data`` is a declared dependency, so this is a cost
        deferred rather than a dependency avoided.
        """
        from dataknobs_data.query import Filter, Operator, Query

        filters = [Filter(_KIND_FIELD, Operator.EQ, kind)]
        filters += [Filter(name, Operator.EQ, value) for name, value in fields.items()]
        return Query(filters=filters)


def _key(kind: str, entity_id: str) -> str:
    """The record id for one entity."""
    return f"{kind}:{entity_id}"


def _assignment_key(experiment_id: str, user_id: str) -> str:
    """The record id for one user's assignment in one experiment."""
    return f"{_ASSIGNMENT}:{experiment_id}:{user_id}"


def _payload(record: Record) -> dict[str, Any]:
    """The entity dictionary a record carries, without the discriminator."""
    return {name: field.value for name, field in record.fields.items() if name != _KIND_FIELD}
