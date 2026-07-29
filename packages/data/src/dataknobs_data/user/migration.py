"""Per-section schema migration for the per-user state coordinator.

Each user-state section carries a schema ``version`` (stamped onto every
written record as ``_section_version``). When a section's schema evolves, the
consumer registers an ordered set of **pure upgraders** — one per source
version — that rewrite a record's payload from ``v_n`` to ``v_{n+1}``. The
coordinator applies the registered chain **lazily on read**: a record stamped
behind its section's current version is upgraded in memory before it is
returned (and, when the store opts into ``persist_migrations``, written back
with a compare-and-set guard).

Upgraders are declared through a process-global :data:`section_migrators`
registry keyed by section name — mirroring the ``stage_synthesizer_backends`` /
``intent_classifier_backends`` registry pattern — rather than on the frozen
:class:`~dataknobs_data.user.config.UserStateSectionSpec`, because a live
``Callable`` cannot round-trip a frozen config by value. A consumer wires its
migration chain once at import time::

    from dataknobs_data.user import register_section_migrator

    def _v1_to_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(payload)
        out["theme"] = out.pop("color", "light")
        return out

    register_section_migrator("preferences", 1, _v1_to_v2)

An upgrader is a pure ``Callable[[Mapping[str, Any]], Mapping[str, Any]]``: it
receives the record's *consumer payload* (never the coordinator-owned scope
stamps) and returns the next version's payload. It must not mutate its input.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.registry import Registry

#: A pure per-version upgrader: given a ``v_n`` consumer payload, return the
#: ``v_{n+1}`` payload. Must not mutate its input.
SectionUpgrader = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class SectionMigrator:
    """Ordered schema upgraders for one section, keyed by source version.

    Immutable: :meth:`with_step` returns a new migrator with an added step
    rather than mutating in place, so a migrator handed out of the registry
    can never be changed underneath a concurrent reader. The registry stores
    one migrator per section name; :func:`register_section_migrator` composes
    steps onto it.
    """

    section: str
    upgraders: Mapping[int, SectionUpgrader] = field(default_factory=dict)

    def with_step(
        self, from_version: int, fn: SectionUpgrader
    ) -> SectionMigrator:
        """Return a new migrator with a ``from_version -> from_version+1`` step.

        A later registration of the same ``from_version`` replaces the earlier
        upgrader (last wins), matching the registry's ``allow_overwrite`` shape.
        A non-positive ``from_version`` is rejected — section versions start at
        ``1``, so there is no ``v0`` to upgrade from.
        """
        if from_version < 1:
            raise ConfigurationError(
                f"Section {self.section!r} migration step has "
                f"from_version={from_version}: schema versions start at 1, so "
                "a migration step must upgrade from version 1 or higher.",
                context={"section": self.section, "from_version": from_version},
            )
        merged = dict(self.upgraders)
        merged[from_version] = fn
        return replace(self, upgraders=merged)

    def chain(
        self, from_version: int, to_version: int
    ) -> list[SectionUpgrader]:
        """Return the ordered upgraders taking ``from_version`` to ``to_version``.

        An empty chain is returned for a no-op or backwards window
        (``to_version <= from_version``). A missing intermediate step raises
        :class:`~dataknobs_common.exceptions.ConfigurationError`: a declared
        version bump with no path to it is a consumer wiring bug, surfaced at
        read time rather than silently returning a partially-upgraded record.
        """
        if to_version <= from_version:
            return []
        steps: list[SectionUpgrader] = []
        for version in range(from_version, to_version):
            fn = self.upgraders.get(version)
            if fn is None:
                raise ConfigurationError(
                    f"Section {self.section!r} has no migration step from "
                    f"version {version} to {version + 1}; the requested chain "
                    f"{from_version}→{to_version} has a gap. Register "
                    "every intermediate step with register_section_migrator.",
                    context={
                        "section": self.section,
                        "missing_from_version": version,
                    },
                )
            steps.append(fn)
        return steps


#: Process-global registry of per-section migrators, keyed by section name.
#: Mirrors the ``stage_synthesizer_backends`` / ``intent_classifier_backends``
#: registries: a consumer registers its upgraders once at import time and every
#: store built for that section applies them.
section_migrators: Registry[SectionMigrator] = Registry(
    name="user_state_section_migrators"
)


def register_section_migrator(
    section: str, from_version: int, fn: SectionUpgrader
) -> None:
    """Register a ``v_n -> v_{n+1}`` upgrader for ``section``.

    Accumulates onto the section's existing migrator (creating one on first
    use); a repeat registration of the same ``from_version`` replaces the
    previous upgrader. ``fn`` is a pure
    :data:`SectionUpgrader` — it receives a version-``from_version`` consumer
    payload and returns the version-``from_version + 1`` payload, and must not
    mutate its input.
    """
    existing = section_migrators.get_optional(section)
    migrator = existing if existing is not None else SectionMigrator(section)
    section_migrators.register(
        section, migrator.with_step(from_version, fn), allow_overwrite=True
    )


def resolve_chain(
    section: str, from_version: int, to_version: int
) -> list[SectionUpgrader]:
    """Return the upgrader chain for ``section`` from ``from_version`` on.

    A no-op or backwards window (``to_version <= from_version``) returns an
    empty chain without consulting the registry. Otherwise the section's
    migrator is resolved and its :meth:`SectionMigrator.chain` returned; a
    missing migrator (a record needs upgrading but the section has none
    registered) or a gap within the chain raises
    :class:`~dataknobs_common.exceptions.ConfigurationError` — a consumer
    wiring bug surfaced at read time.
    """
    if to_version <= from_version:
        return []
    migrator = section_migrators.get_optional(section)
    if migrator is None:
        raise ConfigurationError(
            f"Section {section!r} has a record at version {from_version} but "
            f"its current schema version is {to_version}, and no migrator is "
            "registered for the section. Register the upgrade chain with "
            "register_section_migrator so the record can be migrated on read.",
            context={
                "section": section,
                "from_version": from_version,
                "to_version": to_version,
            },
        )
    return migrator.chain(from_version, to_version)
