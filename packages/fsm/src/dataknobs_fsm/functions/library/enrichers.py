"""Record-enrichment substrate for FSM patterns.

This module is the enrichment counterpart to :mod:`validators`: one shared,
type-dispatching :func:`build_record_enricher` front door normalizes any
supported enrichment spec to a ``(record, context) -> dict`` per-record step, so
a pattern (today the ETL pattern) — or any consumer rolling their own — can add
fields to a record from a computed value, a reference-table lookup, a pre-built
library construct, or a callable, without re-implementing the merge policy.

Four spec forms are accepted:

- a **field→value mapping** (static or callable values) — backed by the shipped
  :class:`~dataknobs_fsm.functions.library.transformers.DataEnricher`; pure data,
  no resource;
- a **reference-table lookup** keyed under ``resource`` (a registered async
  database resource name) with a ``match`` join spec — :class:`LookupMergeEnricher`
  resolves the resource from ``FunctionContext.resources``, reads the matching
  reference record via a dataknobs :class:`~dataknobs_data.Query`, and merges the
  looked-up ``fields`` under the same ``overwrite`` policy as the field-map form;
- any **:class:`~dataknobs_fsm.functions.base.ITransformFunction`** instance — used
  directly (its ``transform`` may be sync or async);
- a plain **callable** ``record -> dict`` or ``(record, context) -> dict`` (sync
  or async) — arity/await-normalized to the engine's call shape.

The lookup form deliberately compiles to a dataknobs ``Query`` rather than raw
SQL: the async database resource adapter reads through ``Query`` +
``stream_read`` (it rejects raw SQL strings), so a ``Query`` lookup is both the
supported path and backend-agnostic (file, memory, sqlite, postgres).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from dataknobs_data import Filter, Operator, Query

from dataknobs_fsm.functions.base import ITransformFunction, TransformError
from dataknobs_fsm.functions.library._callables import normalize_record_callable
from dataknobs_fsm.functions.library.database import _require_resource
from dataknobs_fsm.functions.library.transformers import (
    DataEnricher,
    merge_enrichment_field,
)

#: Accepted ``enrichment_on_missing`` policies for a reference lookup.
ENRICHMENT_ON_MISSING = ("ignore", "null", "error")

# Spec key naming a registered resource for a reference lookup (library form).
_LOOKUP_RESOURCE_KEY = "resource"
# ETL-level convenience keys (a backend config, not a registered name); the ETL
# pattern registers the config as an FSM resource and rewrites to ``resource``.
_ETL_SOURCE_KEYS = ("database", "api")


class LookupMergeEnricher(ITransformFunction):
    """Enrich a record from a reference table by key, then merge fields.

    Resolves an injected async database resource (declared on the enriching
    state and bound into ``FunctionContext.resources``), reads the single
    reference record whose columns equal the record's ``match`` values, and
    merges the requested ``fields`` into the record under the ``overwrite``
    policy. A record with no matching reference row is handled per
    ``on_missing``.

    The reference read goes through a dataknobs :class:`~dataknobs_data.Query`
    (``execute_query(query, fetch_one=True)``) — the primitive the async
    database resource adapter supports — so the lookup is backend-agnostic. A
    multi-field ``match`` compiles to AND-combined equality filters (every join
    column must match). When more than one reference row satisfies the join only
    the first (in backend ``stream_read`` order) is merged — author ``match`` on a
    unique key; multi-row fan-out is not supported. A record missing a ``match``
    source field queries that join column against ``None`` (so it matches a row
    whose reference column is null, if any), rather than raising.
    """

    def __init__(
        self,
        resource_name: str,
        match: Mapping[str, str],
        fields: list[str] | None = None,
        *,
        overwrite: bool = False,
        on_missing: str = "ignore",
    ) -> None:
        """Initialize the lookup-merge enricher.

        Args:
            resource_name: Name of the (async database) resource to look up in.
            match: ``{record_field: reference_field}`` equality join — the
                reference row whose ``reference_field`` columns equal the
                record's ``record_field`` values is the match.
            fields: Reference columns to merge into the record. When omitted,
                every reference column except the ``match`` reference fields is
                merged — including the reference table's own ``id``/storage
                columns, so name ``fields`` explicitly when the reference shares
                column names with the record. Required when ``overwrite`` is
                ``True`` or ``on_missing`` is ``"null"`` (see ``Raises``).
            overwrite: When ``False`` a looked-up field that collides with an
                existing record key is dropped; when ``True`` it replaces it
                (requires explicit ``fields``).
            on_missing: Policy when no reference row matches — ``"ignore"`` (pass
                the record through unchanged), ``"null"`` (set the listed
                ``fields`` to ``None``, subject to the ``overwrite`` policy: a
                field the record already carries is left untouched when
                ``overwrite`` is ``False`` and nulled only when it is ``True``),
                or ``"error"`` (raise :class:`TransformError`, so the record is a
                counted error).

        Raises:
            ValueError: ``match`` empty; unknown ``on_missing``; ``overwrite``
                without explicit ``fields``; or ``on_missing="null"`` without
                explicit ``fields``.
        """
        if not match:
            raise ValueError(
                "LookupMergeEnricher requires a non-empty 'match' mapping "
                "{record_field: reference_field}"
            )
        if on_missing not in ENRICHMENT_ON_MISSING:
            raise ValueError(
                f"unknown on_missing '{on_missing}'; expected one of "
                f"{', '.join(ENRICHMENT_ON_MISSING)}"
            )
        # A blanket "merge every reference column" (``fields`` omitted) combined
        # with ``overwrite`` can replace the record's own key/identity columns
        # (the reference table's ``id`` clobbers the record's ``id``), collapsing
        # rows onto the wrong storage id at load. Demand explicit ``fields`` when
        # overwriting so the author names exactly what is replaced.
        if overwrite and fields is None:
            raise ValueError(
                "overwrite=True requires an explicit 'fields' list — a blanket "
                "merge of every reference column with overwrite can clobber the "
                "record's own key columns. Name the fields to overwrite."
            )
        # ``on_missing="null"`` nulls the named ``fields``; with no ``fields`` it
        # would null nothing — silently identical to ``"ignore"``. Reject the
        # no-op rather than let it masquerade as a miss policy.
        if on_missing == "null" and fields is None:
            raise ValueError(
                "on_missing='null' requires an explicit 'fields' list — there "
                "are no named fields to set to None."
            )
        self.resource_name = resource_name
        self.match = dict(match)
        self.fields = list(fields) if fields is not None else None
        self.overwrite = overwrite
        self.on_missing = on_missing

    async def transform(
        self, data: dict, context: Any = None
    ) -> dict:
        resource = _require_resource(self.resource_name, context)
        query = Query(
            filters=[
                Filter(ref_field, Operator.EQ, data.get(rec_field))
                for rec_field, ref_field in self.match.items()
            ]
        )
        try:
            row = await resource.execute_query(
                query, fetch_one=True, as_dict=True
            )
        except Exception as e:
            raise TransformError(f"Enrichment lookup failed: {e}") from e

        result = dict(data)
        if not row:
            return self._apply_on_missing(result)

        if self.fields is not None:
            merge_fields = self.fields
        else:
            ref_columns = set(self.match.values())
            merge_fields = [k for k in row if k not in ref_columns]
        for field in merge_fields:
            if field in row:
                merge_enrichment_field(
                    result, field, row[field], overwrite=self.overwrite
                )
        return result

    def _apply_on_missing(self, result: dict) -> dict:
        if self.on_missing == "error":
            raise TransformError(
                f"No reference row matched {self.match} for enrichment "
                f"from resource '{self.resource_name}'"
            )
        if self.on_missing == "null":
            for field in self.fields or ():
                merge_enrichment_field(
                    result, field, None, overwrite=self.overwrite
                )
        # "ignore": pass the record through unchanged.
        return result

    def get_transform_description(self) -> str:
        return (
            f"Enrich from '{self.resource_name}' matching {self.match} "
            f"(on_missing={self.on_missing})"
        )


def build_record_enricher(
    spec: Mapping[str, Any] | ITransformFunction | Callable[..., Any],
    *,
    on_missing: str = "ignore",
) -> Callable[..., Any]:
    """Normalize any supported enrichment spec to a per-record enrich step.

    Returns a callable the enrich step invokes as ``enrich(record, context)``
    and which returns the enriched record dict (a coroutine when the underlying
    enricher is async). See the module docstring for the four accepted forms.

    Args:
        spec: The enrichment specification (mapping / ``ITransformFunction`` /
            callable).
        on_missing: Lookup-miss policy forwarded to a reference-lookup spec
            (ignored by the other forms).

    Returns:
        A ``(record, context) -> dict`` enricher (async when the underlying
        enricher is async).

    Raises:
        TypeError: ``spec`` is none of the supported forms, or a mapping carries
            an ETL-level ``database`` / ``api`` source key (use ``resource`` with
            a registered resource name), or a mapping carries a ``match`` join
            spec but no ``resource`` source (a malformed reference lookup that
            would otherwise be silently mis-read as a field→value map).
        ValueError: a ``resource`` lookup spec lacks a non-empty ``match``.
    """
    if isinstance(spec, ITransformFunction):
        return normalize_record_callable(spec.transform)
    if isinstance(spec, Mapping):
        if _LOOKUP_RESOURCE_KEY in spec:
            match = spec.get("match")
            if not isinstance(match, Mapping) or not match:
                raise ValueError(
                    "a 'resource' lookup enrichment requires a non-empty 'match' "
                    "mapping {record_field: reference_field}"
                )
            enricher = LookupMergeEnricher(
                resource_name=spec[_LOOKUP_RESOURCE_KEY],
                match=match,
                fields=spec.get("fields"),
                overwrite=bool(spec.get("overwrite", False)),
                on_missing=on_missing,
            )
            return normalize_record_callable(enricher.transform)
        conflict = next((k for k in _ETL_SOURCE_KEYS if k in spec), None)
        if conflict is not None:
            raise TypeError(
                f"enrichment spec key '{conflict}' is an ETL-level convenience "
                "(a backend config, not a registered name); the library builder "
                "takes a registered resource name under the 'resource' key. Pass "
                "{'resource': <name>, 'match': {...}}, or let the ETL pattern "
                "register the backend and rewrite it for you."
            )
        # A ``match`` join spec with no source key is a reference lookup missing
        # its ``resource`` — NOT a field→value map. Reinterpreting it as a
        # field-map would add literal ``match`` / ``fields`` columns to every
        # record and silently mis-enrich (the exact silent-no-op this builder
        # exists to prevent). Reject it pointing at the missing source key.
        if isinstance(spec.get("match"), Mapping):
            raise TypeError(
                "enrichment spec has a 'match' join spec but no 'resource' "
                "source key — a reference lookup needs {'resource': <name>, "
                "'match': {...}}. Add the resource name, or remove 'match' for a "
                "computed field→value map."
            )
        # No source key: a field->value map (static or callable values).
        return normalize_record_callable(DataEnricher(dict(spec)).transform)
    if callable(spec):
        return normalize_record_callable(spec)
    raise TypeError(
        "enrichment spec must be a mapping (field map or reference lookup), an "
        "ITransformFunction, or a callable; got " + type(spec).__name__
    )
