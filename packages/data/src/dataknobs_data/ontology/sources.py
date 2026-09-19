# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A vocabulary over a table someone else owns.

The projection is **data**, never a callable. That is the decision write-through
rests on: ``name: display_name`` inverts, ``lambda row: Entity(...)`` does not,
and choosing the callable looks like a convenience while foreclosing the return
path permanently. So the declarative form is the published surface and every
column a projection names is validated at load against a schema the binding
**declares** -- there is no introspection member on
:class:`~dataknobs_data.database.AsyncDatabase`, and an overlay over foreign
data is the case least likely to have declared one elsewhere, which is exactly
why silence here would validate nothing and report green.

A column name from configuration reaches a query builder and is never
interpolated into SQL: :class:`~dataknobs_data.query.Filter` is the only path.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import batched
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from dataknobs_common.capabilities import (
    Capability,
    CapabilityLike,
    CapabilityNotSupportedError,
    DynamicCapabilityMixin,
)
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import Entity, SourceDescription, SourceRef
from dataknobs_common.text import default_normalizer

from dataknobs_data.query import Filter, Operator, Query

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from dataknobs_common.records import Record

    from dataknobs_data.database import AsyncDatabase
    from dataknobs_data.schema import DatabaseSchema

#: The source kind this module binds. One value, spelled the way the
#: published ``sources:`` grammar spells it, and read by the registry's
#: dispatch --
#: which computes its refusal from the declared kind alone, so a document
#: means the same thing whether or not this package has been imported.
RECORD_SOURCE_KIND = "record"

#: How many ids one bulk read asks about at once.
#:
#: A bulk member sends one ``IN`` filter because a round trip per id is the
#: cost it exists to avoid -- but one filter is one read, and a read is a
#: thing a backend bounds. Two bounds are in reach and neither is this
#: package's to set: ``AsyncElasticsearchDatabase.search`` answers an
#: unbounded query with ``size=10000``, and the SQL backends bind one
#: parameter per element of an ``IN`` list against a server ceiling --
#: 65535 on Postgres, and 999 on a SQLite built before 3.32. So a batch
#: larger than this is split, and each read stays under both.
#:
#: 1000 because it is already this package's answer to *how many rows per
#: read*: :attr:`~dataknobs_data.streaming.StreamConfig.batch_size` has
#: carried that default since streaming shipped, and a second number here
#: would be a second answer to one question.
READ_BATCH_SIZE = 1000


def _optional_positive_int(
    row: Mapping[str, Any], key: str, *, binding: str, section: str
) -> int | None:
    """An optional whole number of at least one, or None where the key is absent.

    Below one is refused rather than clamped, for the reason
    ``_checked_max_window`` refuses it one package along: a bound of zero
    builds a scan that probes nothing and reports the empty set for every
    query, which is indistinguishable from a vocabulary that matches nothing.
    """
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(
            f"binding {binding!r}: `{section}.{key}:` must be a whole number, "
            f"got {type(value).__name__}",
            context={"source_id": binding, "key": key, "value": value},
        )
    if value < 1:
        raise ValidationError(
            f"binding {binding!r}: `{section}.{key}:` must be at least 1, got "
            f"{value}. A bound of zero probes no window and reports nothing for "
            f"every query, which is what a vocabulary matching nothing looks like",
            context={"source_id": binding, "key": key, "value": value},
        )
    return value


@dataclass(frozen=True)
class SurfaceFormLookup:
    """The declared ``(folded form, entity)`` table ``by_surface_form`` reads.

    A **table** rather than a column, because the lookup is many-to-many in
    both directions: an authored index folds an entity's id, its name and
    every one of its aliases into one map and answers a set, so one folded
    form reaches many entities and one entity is reached by many forms. A
    ``name_folded:`` column would answer for names and silently miss every
    alias.

    Attributes:
        table: The side table, one row per ``(folded form, entity)`` pair
        form: The column holding ``normalizer(form)`` -- the source's own fold,
            applied when the row was written
        entity: The column holding the entity id, in the space the
            projection's ``id:`` names
        longest_form_tokens: How many tokens the longest form in this table
            occupies, or None where the binding declares no bound. Declared
            rather than measured: an authored index counts its own keys, and a
            live table cannot be counted without reading it -- a count taken at
            load is stale as soon as a row is written. A scanning rung needs
            this number to stay linear in the caller's input, and is refused at
            load over a binding that declares none
    """

    table: str
    form: str
    entity: str
    longest_form_tokens: int | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, binding: str) -> SurfaceFormLookup:
        """Parse a ``surface_forms:`` block, naming the binding on a refusal."""
        return cls(
            table=str(_required(row, "table", binding=binding, section="surface_forms")),
            form=str(_required(row, "form", binding=binding, section="surface_forms")),
            entity=str(_required(row, "entity", binding=binding, section="surface_forms")),
            longest_form_tokens=_optional_positive_int(
                row, "longest_form_tokens", binding=binding, section="surface_forms"
            ),
        )


@dataclass(frozen=True)
class EntityProjection:
    """The column-to-field map, as configuration.

    Every bare scalar names a column -- ``id: sku``, ``name: title`` -- and the
    braced forms are the ones that do something else: ``{const: Product}`` is a
    literal where the table has no type column, ``{column: alt_names, split:
    ","}`` is a delimited list, ``{columns: [...]}`` gathers metadata. A column
    may be a **dotted path** into a JSON value, and exactly one thing is
    promised about that: the *root segment* is validated at load and the path
    within the value is not, because
    :class:`~dataknobs_data.schema.FieldSchema` is flat and a declared schema
    can prove ``payload`` exists while saying nothing about ``address.city``
    inside it. A projection reaching into a blob fails on first read.

    Attributes:
        table: The table the entity rows live in
        id: The column holding the local id -- the round-trip key
        name: The column holding the display name, or None to use the id
        const_type: The one entity type every projected row carries
        aliases: The column holding alias forms, or None
        alias_split: The delimiter that column is split on, or None for one
            alias per row value
        description: The column holding a description, or None
        metadata_columns: Columns gathered into the entity's ``metadata``
        expose_origin: Whether ``fetch_origin`` may return unprojected columns.
            Defaults to ``True``: the projection is a *mapping*, not an access
            boundary, and a consumer needing narrowing enforced binds a
            source-side view or sets this false
        surface_forms: The declared folded lookup, or None where the binding
            declares none
    """

    table: str
    id: str
    const_type: str
    name: str | None = None
    aliases: str | None = None
    alias_split: str | None = None
    description: str | None = None
    metadata_columns: tuple[str, ...] = ()
    expose_origin: bool = True
    surface_forms: SurfaceFormLookup | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, binding: str) -> EntityProjection:
        """Parse an ``entity_projection:`` block, naming the binding on a refusal.

        Args:
            row: The block as written
            binding: The source id, so every refusal names what to go and fix

        Returns:
            The parsed projection

        Raises:
            ValidationError: On a missing ``table:`` or ``id:``; on a ``type:``
                that is absent or names a column rather than declaring a
                constant; on a malformed ``aliases:`` or ``metadata:`` block
        """
        aliases_spec = row.get("aliases")
        aliases: str | None = None
        alias_split: str | None = None
        if isinstance(aliases_spec, Mapping):
            aliases = str(_required(aliases_spec, "column", binding=binding, section="aliases"))
            split = aliases_spec.get("split")
            alias_split = None if split is None else str(split)
        elif aliases_spec is not None:
            aliases = str(aliases_spec)

        metadata_spec = row.get("metadata")
        metadata_columns: tuple[str, ...] = ()
        if isinstance(metadata_spec, Mapping):
            columns = metadata_spec.get("columns", ())
            if isinstance(columns, str) or not isinstance(columns, Iterable):
                raise ValidationError(
                    f"binding {binding!r}: `metadata:` takes `{{columns: [...]}}`, got {columns!r}",
                    context={"source_id": binding, "metadata": metadata_spec},
                )
            metadata_columns = tuple(str(column) for column in columns)
        elif metadata_spec is not None:
            raise ValidationError(
                f"binding {binding!r}: `metadata:` takes `{{columns: [...]}}`, "
                f"got {metadata_spec!r}",
                context={"source_id": binding, "metadata": metadata_spec},
            )

        surface_forms_spec = row.get("surface_forms")
        surface_forms: SurfaceFormLookup | None = None
        if isinstance(surface_forms_spec, Mapping):
            surface_forms = SurfaceFormLookup.from_mapping(surface_forms_spec, binding=binding)
        elif surface_forms_spec is not None:
            raise ValidationError(
                f"binding {binding!r}: `surface_forms:` takes a block naming "
                f"`table:`, `form:` and `entity:`, got {surface_forms_spec!r}",
                context={"source_id": binding, "surface_forms": surface_forms_spec},
            )

        name = row.get("name")
        description = row.get("description")
        return cls(
            table=str(_required(row, "table", binding=binding, section="entity_projection")),
            id=str(_required(row, "id", binding=binding, section="entity_projection")),
            const_type=_const_type(row.get("type"), binding=binding),
            name=None if name is None else str(name),
            aliases=aliases,
            alias_split=alias_split,
            description=None if description is None else str(description),
            metadata_columns=metadata_columns,
            expose_origin=bool(row.get("expose_origin", True)),
            surface_forms=surface_forms,
        )

    def entity_columns(self) -> tuple[str, ...]:
        """Every column this projection names in the **entity** table."""
        columns = [self.id]
        for optional in (self.name, self.aliases, self.description):
            if optional:
                columns.append(optional)
        columns.extend(self.metadata_columns)
        return tuple(columns)

    def form_columns(self) -> tuple[str, ...]:
        """Every column this projection names in the **surface-form** table."""
        if self.surface_forms is None:
            return ()
        return (self.surface_forms.form, self.surface_forms.entity)

    def to_mapping(self) -> dict[str, Any]:
        """The projection as ``describe()`` reports it -- configuration, not state."""
        reported: dict[str, Any] = {
            "table": self.table,
            "id": self.id,
            "type": {"const": self.const_type},
            "expose_origin": self.expose_origin,
        }
        if self.name:
            reported["name"] = self.name
        if self.aliases:
            reported["aliases"] = (
                {"column": self.aliases, "split": self.alias_split}
                if self.alias_split is not None
                else self.aliases
            )
        if self.description:
            reported["description"] = self.description
        if self.metadata_columns:
            reported["metadata"] = {"columns": list(self.metadata_columns)}
        if self.surface_forms is not None:
            reported["surface_forms"] = {
                "table": self.surface_forms.table,
                "form": self.surface_forms.form,
                "entity": self.surface_forms.entity,
            }
        return reported


def _required(row: Mapping[str, Any], key: str, *, binding: str, section: str) -> Any:
    value = row.get(key)
    if value is None or value == "":
        raise ValidationError(
            f"binding {binding!r}: `{section}:` requires `{key}:`, and this one "
            f"declares none. Declared: {sorted(row)}",
            context={"source_id": binding, "section": section, "key": key},
        )
    return value


def _const_type(declared: Any, *, binding: str) -> str:
    """The one declared entity type, or a refusal naming why a column is not one yet.

    ``type: {const: Product}`` is the form this version binds.

    ``type: kind`` -- the column form -- is refused, and the refusal is about
    what a source can *say*, not about what it can read.
    :attr:`~dataknobs_common.ontology.SourceDescription.declares` is the field
    an index enumerates a vocabulary from, and over a column the honest answer
    is *I cannot enumerate my types without a scan*. That answer is spelled
    ``None`` and the field does not admit it yet, so a source over a type
    column would have to answer ``frozenset()`` instead -- which already means
    *this source declares no types at all*, and is read as a complete
    enumeration by everything downstream. Refusing costs a consumer a
    ``{const: ...}`` per table; answering would cost them entities that quietly
    stop being indexed.
    """
    if declared is None:
        raise ValidationError(
            f"binding {binding!r}: `entity_projection:` requires `type:`, either "
            f"as `{{const: <type id>}}` or, once a source may report that it "
            f"cannot enumerate its types, as a column name",
            context={"source_id": binding, "key": "type"},
        )
    if isinstance(declared, Mapping):
        const = declared.get("const")
        if const is None:
            raise ValidationError(
                f"binding {binding!r}: `type:` takes `{{const: <type id>}}`, got {declared!r}",
                context={"source_id": binding, "type": declared},
            )
        return str(const)
    raise ValidationError(
        f"binding {binding!r}: `type: {declared!r}` names a column, and this "
        f"version binds only `type: {{const: <type id>}}`. A source over a type "
        f"column cannot report that its declared types are unenumerable, and "
        f"reporting an empty set instead would read as a complete enumeration",
        context={"source_id": binding, "type": declared},
    )


def validate_against_schema(
    projection: EntityProjection,
    schema: DatabaseSchema | None,
    *,
    binding: str,
) -> None:
    """Refuse a binding with no declared schema, and a column that schema lacks.

    Two refusals, both at load, both naming what to fix. The first: a
    by-reference binding declares its schema or it is rejected, because no
    ``Database`` exposes introspection and the case least likely to have a
    schema declared elsewhere is precisely this one. The second is what the
    first buys -- a column name from a config file, checked against a
    declaration before it ever reaches a query builder.

    **The check reaches the root segment only.** A dotted path names a value
    inside a JSON column, ``FieldSchema`` is flat, and a declared schema can
    prove ``payload`` exists while saying nothing about ``address.city``. That
    is stated rather than discovered: promising more would be the quiet failure
    the declaration requirement exists to prevent.

    **One schema covers both of the binding's tables.** ``FieldSchema`` carries
    no table, so the entity table's columns and the surface-form table's are
    checked against one list, and two columns of the same name in the two
    tables are one entry. That is a naming constraint on the consumer rather
    than a validation hole -- a name declared once is checked once and both
    uses are checked against it.

    Args:
        projection: The parsed projection
        schema: The schema the binding declared, or None
        binding: The source id, so a refusal names what to go and fix

    Raises:
        ValidationError: When the binding declares no schema, or names a column
            the declared schema does not carry
    """
    if schema is None or not schema.fields:
        raise ValidationError(
            f"binding {binding!r} declares no `schema:`. A by-reference binding "
            f"must declare the columns it projects: no database in this package "
            f"exposes schema introspection, so a binding without one would "
            f"validate nothing and report success",
            context={"source_id": binding},
        )
    declared = set(schema.fields)
    for column in (*projection.entity_columns(), *projection.form_columns()):
        root = column.split(".", 1)[0]
        if root not in declared:
            raise ValidationError(
                f"binding {binding!r}: the projection names column {column!r}, "
                f"which this binding's `schema:` does not declare. Declared: "
                f"{sorted(declared)}",
                context={"source_id": binding, "column": column},
            )


class RecordEntitySource(DynamicCapabilityMixin):
    """An :class:`~dataknobs_common.ontology.AsyncEntitySource` over a table.

    Every member that reaches for data is ``async`` because the protocol says
    so, and :meth:`describe` and :meth:`longest_form_tokens` stay synchronous
    because they touch no backend.

    **``get`` is a query, not a read.** A record's storage id is not the
    projection's ``id:`` column -- over a store holding ``{"sku": "sku-4471"}``
    a read by ``"sku-4471"`` answers None while a filter on ``sku`` answers the
    row -- so :meth:`get` searches and :meth:`get_many` sends one ``IN`` filter
    rather than N reads.

    Args:
        database: The handle the entity rows are read through. Built by the
            registry from a resolved ``$resource``, or handed in already built
        projection: The column-to-field map
        source_id: The binding's id, carried on every ``SourceRef`` this
            source mints
        backend: What to report as ``describe().backend``. Defaults to the
            handle's class name
        normalizer: How a surface form is folded. Defaults to
            :func:`~dataknobs_common.text.default_normalizer`, and it is the
            *same* parameter :class:`MappingEntitySource` carries so a consumer
            who built their lookup table with their own fold hands one callable
            to both halves
        forms_database: The handle the surface-form table is read through.
            Defaults to ``database``, which is what an injected handle means:
            one store holding rows of both kinds. That arrangement is
            supported rather than merely tolerated, and
            :meth:`_entity_filters` is what makes the entity side of it
            answer with entity rows
    """

    #: Everything a ``kind: record`` binding **can** declare -- the ceiling,
    #: answerable without an instance, which is what the classmethod half of
    #: :class:`~dataknobs_common.capabilities.CapabilityContract` is for: a
    #: caller choosing a source kind has a kind and no source yet.
    #:
    #: Both are conditional on the projection, so what any one binding has is
    #: :meth:`_compute_instance_capabilities`'s answer and is a subset of
    #: this. That is the whole reason the *dynamic* mixin rather than the
    #: plain one.
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]] = frozenset(
        {Capability.ORIGIN_FETCH, Capability.SURFACE_FORM_LOOKUP}
    )

    def __init__(
        self,
        database: AsyncDatabase,
        projection: EntityProjection,
        *,
        source_id: str,
        backend: str | None = None,
        normalizer: Callable[[str], str] | None = None,
        forms_database: AsyncDatabase | None = None,
    ) -> None:
        self._db = database
        self._projection = projection
        self._source_id = source_id
        self._backend = backend or type(database).__name__
        self._normalizer = normalizer or default_normalizer
        self._forms_db = forms_database if forms_database is not None else database
        # Computed once: every one of these is a property of the projection
        # and the handles, which are configuration and do not change after
        # construction.
        self._shared_store = self._forms_db is self._db
        self._capabilities = self._declared_capabilities()
        self._init_capability_cache()

    @property
    def projection(self) -> EntityProjection:
        """The map this source projects with -- configuration, and read-only."""
        return self._projection

    def _entity_filters(self) -> list[Filter]:
        """What narrows a read to the **entity** table, on this pair of handles.

        Empty where the two tables have handles of their own: the three SQL
        backends declare a ``table`` on their config, so the registry opens
        one handle per table and a query through either reaches only its own
        rows.

        Not empty where the handle *is* the store. ``memory``, ``file``,
        ``s3`` and ``elasticsearch`` declare no table, so one ``$resource``
        is one store and rows of both kinds live in it -- and a form row
        carries the projection's id column, because
        :attr:`SurfaceFormLookup.entity` names the space the projection's
        ``id:`` names. An id filter alone therefore reaches both kinds, and
        which one a read answers with is decided by insertion order: over the
        guide's own example ``get`` can return a form row projected as the
        entity, with an empty name and no aliases, and report nothing.

        The discriminator is the form column, which is what the *other*
        direction already relies on: :meth:`by_surface_form` filters on it
        and reaches only form rows because an entity row does not carry it.
        This is that same invariant read the other way round, so a shared
        store needs no assumption it did not already need.
        """
        if not self._shared_store or self._projection.surface_forms is None:
            return []
        return [Filter(self._projection.surface_forms.form, Operator.NOT_EXISTS)]

    async def get(self, entity_id: str) -> Entity[str] | None:
        """The entity this local id names, or None."""
        found = await self._db.search(
            Query(
                filters=[
                    Filter(self._projection.id, Operator.EQ, entity_id),
                    *self._entity_filters(),
                ]
            ).limit(1)
        )
        return self._projected(found[0]) if found else None

    async def get_many(self, entity_ids: Sequence[str]) -> dict[str, Entity[str]]:
        """Those of these ids that name a row. Misses are absent.

        One ``IN`` filter rather than N reads: the bulk member exists because a
        round trip per id is the cost it was added to avoid.
        """
        if not entity_ids:
            return {}
        found = await self._read_ids(list(entity_ids))
        projected = (self._projected(record) for record in found)
        return {entity.id: entity for entity in projected}

    async def fetch_origin(self, ref: SourceRef) -> Record | None:
        """The backing row, when this binding exposes origins and the ref is ours.

        None when ``expose_origin: false`` -- and ``describe()`` withholds
        :attr:`~dataknobs_common.capabilities.Capability.ORIGIN_FETCH` so a
        caller learns that without making the call.
        """
        local_id = self._origin_key(ref)
        if local_id is None:
            return None
        found = await self._db.search(
            Query(
                filters=[
                    Filter(self._projection.id, Operator.EQ, local_id),
                    *self._entity_filters(),
                ]
            ).limit(1)
        )
        return found[0] if found else None

    async def fetch_origins(self, refs: Sequence[SourceRef]) -> list[Record | None]:
        """Every ref's backing row, in one read -- one slot per ref, in order.

        **This is the member the protocol change bought.** It used to refuse,
        because its declared answer was ``dict[SourceRef, Record]`` and
        :class:`~dataknobs_common.ontology.SourceRef` is deliberately
        unhashable, so no non-empty value of that type could be built. Every
        implementation that existed answered ``{}`` and satisfied the type
        only by being empty; this was the first source that could actually
        reach an origin, so it was the first that had to build the value and
        the first to find that nothing could. The declaration was the half
        that was wrong -- see
        :meth:`~dataknobs_common.ontology.sources.EntitySource.fetch_origins`.

        **One ``search`` for the whole batch**, which is the only thing that
        distinguishes this member from a loop over :meth:`fetch_origin`: the
        rows it returns are the same rows. The ids are collected, deduplicated
        and sent as a single ``IN`` filter alongside the same narrowing
        :meth:`fetch_origin` applies, so a shared store still cannot answer
        with a surface-form row.

        ``IN`` rather than a per-ref ``EQ`` disjunction because every backend
        in this package applies it: :class:`~dataknobs_data.query.Filter`
        evaluates it directly for the in-process stores, ``sql_base`` compiles
        it, and the Elasticsearch builder maps it to ``terms``. It goes
        through ``search`` for the reason
        :meth:`~dataknobs_data.ontology.sources.RecordEntitySource.by_type`
        keeps its narrowed read there.

        Refs are answered ``None`` in place -- one that belongs to another
        source, one carrying no id under the projection's id column, one whose
        row is gone, and every ref when the binding sets ``expose_origin:
        false``. A caller reads the miss in the slot it asked about, which a
        mapping keyed by ref could not have told it.
        """
        if not refs:
            return []
        wanted: list[Any] = [self._origin_key(ref) for ref in refs]
        distinct = {key for key in wanted if key is not None}
        if not distinct:
            return [None] * len(refs)
        found = await self._read_ids(sorted(distinct, key=str))
        by_key = {record.get_value(self._projection.id): record for record in found}
        return [None if key is None else by_key.get(key) for key in wanted]

    async def _read_ids(self, keys: Sequence[Any]) -> list[Record]:
        """Every entity row whose id is one of these, in reads a backend answers.

        The shared half of :meth:`get_many` and :meth:`fetch_origins`, for the
        reason :meth:`_origin_key` is the shared half of the origin pair: the
        two build the same query over the same two filters, and a difference
        between them would be a bulk read answering rows the other declines.

        **One ``IN`` per batch rather than one for the whole list.** The single
        filter is what makes a bulk member worth having, but one filter is one
        read, and a read is a thing a backend bounds -- by result size on
        Elasticsearch, by bind parameters on the SQL backends. Splitting at
        :data:`READ_BATCH_SIZE` keeps each read under both, and the caller
        cannot tell: the rows are concatenated and both callers index them by
        the projection's id column afterwards, so order across batches carries
        nothing.
        """
        found: list[Record] = []
        for chunk in batched(keys, READ_BATCH_SIZE):
            found.extend(
                await self._db.search(
                    Query(
                        filters=[
                            Filter(self._projection.id, Operator.IN, list(chunk)),
                            *self._entity_filters(),
                        ]
                    )
                )
            )
        return found

    def _origin_key(self, ref: SourceRef) -> Any:
        """The local id this ref names, or ``None`` when it names none of ours.

        The shared half of :meth:`fetch_origin` and :meth:`fetch_origins`, so
        the two cannot drift on what counts as a ref this source will answer
        -- a difference there would be a bulk read that returns rows the
        singular member declines, which is the kind of divergence only a
        consumer finds.
        """
        if not self._projection.expose_origin or ref.source_id != self._source_id:
            return None
        return ref.locator.get(self._projection.id)

    def describe(self) -> SourceDescription:
        """What this source is, without touching the backend.

        ``declares`` is a set of one: the projection's ``type:`` is a constant,
        so the vocabulary's declared types are known from configuration and no
        scan is needed to say so.
        """
        return SourceDescription(
            source_id=self._source_id,
            backend=self._backend,
            table=self._projection.table,
            projection=self._projection.to_mapping(),
            capabilities=self._capabilities,
            declares=frozenset({self._projection.const_type}),
        )

    def _declared_capabilities(self) -> frozenset[Capability]:
        """The two this projection declares, read off the projection.

        Narrowly typed, and that is why it exists beside the mixin's hook:
        :attr:`~dataknobs_common.ontology.SourceDescription.capabilities` is
        ``frozenset[Capability]`` while the hook's is
        ``frozenset[CapabilityLike]``, which also admits a consumer's own
        capability string. This class declares enum members only, so the
        narrow type is the true one and the hook widens it rather than
        :meth:`describe` narrowing.
        """
        declared: set[Capability] = set()
        if self._projection.expose_origin:
            declared.add(Capability.ORIGIN_FETCH)
        if self._projection.surface_forms is not None:
            declared.add(Capability.SURFACE_FORM_LOOKUP)
        return frozenset(declared)

    def _compute_instance_capabilities(self) -> frozenset[CapabilityLike]:
        """The mixin's hook, over the set :meth:`_declared_capabilities` computed.

        Which makes ``supports()``, ``instance_capabilities()`` and
        ``supported_capabilities()`` this class's too, rather than one of the
        three hand-rolled here and the other two absent -- so a caller
        enumerating capabilities *through the contract* sees a contract host.
        """
        return self._capabilities

    async def by_surface_form(self, form: str) -> frozenset[str]:
        """The ids of entities carrying this form, folded the way this source folds.

        ``EQ`` against the declared ``(folded form, entity)`` table, with the
        query folded by this source's own normalizer -- a byte comparison,
        which is the one operation that means the same thing on every backend.
        There is no query-time fold available: the contract's fold is
        :meth:`str.casefold`, every engine primitive is simple lowercasing, and
        the two differ on the first European catalogue anyone binds.

        **A binding that declared no lookup refuses rather than answering.**
        ``frozenset()`` already means *ran and matched nothing*, and a cascade
        falls through to a guessing rung on exactly that reading -- so a source
        answering over the unfolded column would report a vocabulary gap that
        is not there for every query whose case differs by one letter.

        Raises:
            CapabilityNotSupportedError: When this binding declares no
                ``surface_forms:`` lookup
        """
        lookup = self._projection.surface_forms
        if lookup is None:
            raise CapabilityNotSupportedError(Capability.SURFACE_FORM_LOOKUP, self)
        # async-dispatch-exempt: the fold is the same parameter
        # `MappingEntitySource` takes, and the synchronous twin calls it from a
        # plain `def` with nowhere to await. A normalizer that returned a
        # coroutine would already be unusable by half the sources that accept
        # one, so `Callable[[str], str]` is the contract rather than a narrowing
        # this call site chose.
        folded = self._normalizer(form)
        found = await self._forms_db.search(
            Query(filters=[Filter(lookup.form, Operator.EQ, folded)])
        )
        return frozenset(
            str(entity_id)
            for entity_id in (record.get_value(lookup.entity) for record in found)
            if entity_id is not None
        )

    async def by_type(self, type_id: str) -> frozenset[str]:
        """The ids of every entity of this type.

        The projection declares one constant type, so this is *every row* for
        that one id and empty for any other. A scan, and it is the projection's
        shape rather than this implementation that makes it one: a table with a
        type column answers with a filter, and binding one is the leg that also
        teaches ``declares`` to say *I cannot enumerate*.

        **It costs one pass over the table, every call.** The answer is every
        id, so nothing bounds what the *caller* receives -- on a million-row
        binding this returns a million-element set and there is no ``limit:``
        that would make it something else. What ``stream_read`` bounds is the
        peak in between: a ``Record`` carries the whole row, and streaming
        keeps one batch of them alive rather than all of them, which on the
        backends that page for real -- Postgres by cursor, DuckDB, SQLite and
        Elasticsearch by batch -- is the difference between a bounded read and
        one whose footprint is the table's.

        **Both branches stream, and the narrowed one could not always.** It
        used to take ``search``, because ``stream_read`` and ``search`` are
        separate implementations on every backend and they did not agree:
        Postgres's ``stream_read`` open-coded its WHERE clause and *silently
        dropped* non-EQ filters, which is exactly what :meth:`_entity_filters`
        emits -- ``NOT_EXISTS`` on the form column. Streaming the narrowed
        branch would have answered with form rows projected as entities, on
        one backend, with no error.

        That is fixed: the async Postgres ``stream_read`` now builds its WHERE
        clause through the same ``SQLQueryBuilder`` its ``search`` uses, so
        every async backend's streaming door honours the full operator set --
        memory and file delegate to ``search``, SQLite and DuckDB page it,
        Postgres shares the builder, and Elasticsearch shares the translator.
        The asymmetry was a workaround for that defect and went with it.

        **Keeping it would now lose rows rather than save them.** An unbounded
        ``search`` is the read a backend is free to cap, and one does:
        ``AsyncElasticsearchDatabase`` answers a query carrying no ``limit``
        with ``size=10000``. Elasticsearch declares ``index`` rather than
        ``table``, so the registry gives a binding one handle, so the store is
        shared, so a declared ``surface_forms:`` put the narrowed branch on
        exactly that read -- and a binding of more rows than the cap answered
        with the cap's worth and reported nothing. ``stream_read`` goes through
        the scroll API there, which the cap does not reach. Paging ``search``
        would not have served: past ``index.max_result_window`` a ``from``/
        ``size`` page *errors* rather than truncating, which is why that
        backend's streaming door does not use one.
        """
        if type_id != self._projection.const_type:
            return frozenset()
        narrowing = self._entity_filters()
        query = Query(filters=narrowing) if narrowing else None
        return frozenset(
            {
                str(local_id)
                async for record in self._db.stream_read(query)
                if (local_id := record.get_value(self._projection.id)) is not None
            }
        )

    def longest_form_tokens(self) -> int | None:
        """What the binding declared, or None where it declared nothing.

        **Declared rather than measured, because nothing here can measure
        it.** An authored index counts its own keys at build time. A live
        table's longest form needs a read to know, and a number taken at load
        is stale the moment a row is written -- so a count would be a bound
        that silently stops finding a form added after it, which trades a cost
        problem for a correctness one.

        ``None`` remains this member's published spelling of *I cannot bound
        this*, and a rung told nothing enumerates every window: *n(n+1)/2*
        probes for *n* tokens, each one a round trip on this source. That is
        why the combination is refused where the composition is read rather
        than survived here -- see the registry's refusal, which names the key
        above. A binding no scanning rung is declared over needs no bound and
        keeps answering None.
        """
        lookup = self._projection.surface_forms
        return None if lookup is None else lookup.longest_form_tokens

    def _projected(self, record: Record) -> Entity[str]:
        """One row to one entity -- the single place a column name is spent."""
        projection = self._projection
        local_id = str(record.get_value(projection.id))
        name = record.get_value(projection.name) if projection.name else None
        description = record.get_value(projection.description) if projection.description else None
        return Entity(
            id=local_id,
            type=projection.const_type,
            name="" if name is None else str(name),
            aliases=self._aliases(record),
            description=None if description is None else str(description),
            metadata={
                column: record.get_value(column)
                for column in projection.metadata_columns
                if record.get_value(column) is not None
            },
            source=SourceRef(
                source_id=self._source_id,
                kind=RECORD_SOURCE_KIND,
                locator={"table": projection.table, projection.id: local_id},
            ),
        )

    def _aliases(self, record: Record) -> list[str]:
        """The alias forms this row carries, split where the projection says to."""
        projection = self._projection
        if not projection.aliases:
            return []
        raw = record.get_value(projection.aliases)
        if raw is None:
            return []
        if isinstance(raw, str):
            if projection.alias_split:
                return [part.strip() for part in raw.split(projection.alias_split) if part.strip()]
            return [raw] if raw.strip() else []
        if isinstance(raw, Iterable):
            return [str(item) for item in raw if str(item).strip()]
        return [str(raw)]


__all__ = [
    "RECORD_SOURCE_KIND",
    "EntityProjection",
    "RecordEntitySource",
    "SurfaceFormLookup",
    "validate_against_schema",
]
