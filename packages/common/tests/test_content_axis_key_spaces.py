"""The content axis's key spaces, classified and checked against the tree.

**Two different ``str``\u200bs live in these files and a diff cannot tell them
apart.** One became ``K`` when the content axis went generic; one stayed
``str``. After the pass they look identical, so what makes the change reviewable
is not reading the diff -- it is this table, declared by hand and checked
against what the source actually says.

**It catches the widening in both directions**, and the second is the half that
makes it worth writing:

* a ``text`` or ``payload`` row that became ``K`` -- a mechanical pass that
  widened the wrong thing, and the one everybody expects;
* a **``key`` row that stayed ``str``** -- a row the pass *missed*, which pins
  ``K = str`` for that call and is **invisible in a green suite**, because
  every test in this repository binds ``K`` to ``str`` anyway.

**And a third direction the first two cannot see.** Once the value types are
themselves generic, a row that should read ``Assertion[K]`` and reads
``Assertion`` mentions neither ``str`` nor ``K`` -- so a sweep over
``str``-mentioning annotations goes past it in silence while the default binds
it to ``str``. That direction is :func:`test_no_generic_is_named_bare_inside_the_population`,
and it is the reason this guard has three tests rather than two.

**The population is computed, not listed**: every ``Protocol`` in the two
subpackages, plus every concrete type reachable from one of their annotations,
transitively. A row in the tree that no line below claims fails the run -- which
is the whole difference between a table checked against the source and a list of
the annotations somebody remembered.
"""

from __future__ import annotations

import ast
import pathlib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

#: The two subpackages the content axis lives in.
_ROOTS = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "dataknobs_common" / "ontology",
    pathlib.Path(__file__).resolve().parents[1] / "src" / "dataknobs_common" / "entity_resolution",
)

#: What a ``str`` in one of these annotations is, and whether it moved.
#:
#: ``key`` is the only verdict that carries ``K``; the other four are four
#: different reasons for staying ``str``, and they are four rather than the
#: three originally ruled because the population grew past what three could
#: describe. The distinction they draw is not decorative: it is the difference
#: between an annotation a later reader may widen and one they must not.
#:
#: =========== ====================================================
#: ``key``     an entity key -- becomes ``K``
#: ``text``    surface text a person typed, or wrote in a document
#: ``payload`` a dict key in an opaque blob, or an open label
#: ``schema``  an id in the vocabulary's own declaration space --
#:             entity types, relations, assertions, sources. Authored,
#:             and therefore ``str`` whatever the instances are keyed by
#: ``door``    a key **rendered**, at the boundary where one leaves.
#:             Two rows, both on the codec, which is what renders it
#: =========== ====================================================
_CARRIES_THE_KEY = "key"

#: Every ``str``- or ``K``-mentioning annotation in the population, with its
#: verdict. Hand-declared: a table derived from the tree would agree with the
#: tree by construction, which is the one thing it must not do.
_VERDICTS: tuple[tuple[str, str, str, str], ...] = (
    ("AliasFormSource", "by_alias_form", "form", "text"),
    ("AliasFormSource", "by_alias_form", "->", "key"),
    ("AssertionSource", "get", "assertion_id", "schema"),
    ("AssertionSource", "get", "->", "key"),
    ("AssertionSource", "find", "subject", "key"),
    ("AssertionSource", "find", "object", "key"),
    ("AssertionSource", "find", "->", "key"),
    ("AssertionSource", "find_many", "subjects", "key"),
    ("AssertionSource", "find_many", "objects", "key"),
    ("AssertionSource", "find_many", "->", "key"),
    ("AsyncAliasFormSource", "by_alias_form", "form", "text"),
    ("AsyncAliasFormSource", "by_alias_form", "->", "key"),
    ("AsyncAssertionSource", "get", "assertion_id", "schema"),
    ("AsyncAssertionSource", "get", "->", "key"),
    ("AsyncAssertionSource", "find", "subject", "key"),
    ("AsyncAssertionSource", "find", "object", "key"),
    ("AsyncAssertionSource", "find", "->", "key"),
    ("AsyncAssertionSource", "find_many", "subjects", "key"),
    ("AsyncAssertionSource", "find_many", "objects", "key"),
    ("AsyncAssertionSource", "find_many", "->", "key"),
    ("AsyncEntityResolver", "resolve", "name", "text"),
    ("AsyncEntityResolver", "resolve", "->", "key"),
    ("AsyncEntityResolver", "resolve_many", "names", "text"),
    ("AsyncEntityResolver", "resolve_many", "->", "key"),
    ("AsyncEntitySource", "get", "entity_id", "key"),
    ("AsyncEntitySource", "get", "->", "key"),
    ("AsyncEntitySource", "get_many", "entity_ids", "key"),
    ("AsyncEntitySource", "get_many", "->", "key"),
    ("AsyncEntitySource", "by_surface_form", "form", "text"),
    ("AsyncEntitySource", "by_surface_form", "->", "key"),
    ("AsyncEntitySource", "by_type", "type_id", "schema"),
    ("AsyncEntitySource", "by_type", "->", "key"),
    ("AsyncMatchSignal", "name", "->", "payload"),
    ("AsyncMatchSignal", "candidates", "query", "text"),
    ("AsyncMatchSignal", "candidates", "filter", "payload"),
    ("AsyncMatchSignal", "candidates", "->", "key"),
    ("AsyncMatchSignal", "candidates_many", "queries", "text"),
    ("AsyncMatchSignal", "candidates_many", "filter", "payload"),
    ("AsyncMatchSignal", "candidates_many", "->", "key"),
    ("EntityResolver", "resolve", "name", "text"),
    ("EntityResolver", "resolve", "->", "key"),
    ("EntityResolver", "resolve_many", "names", "text"),
    ("EntityResolver", "resolve_many", "->", "key"),
    ("EntitySource", "get", "entity_id", "key"),
    ("EntitySource", "get", "->", "key"),
    ("EntitySource", "get_many", "entity_ids", "key"),
    ("EntitySource", "get_many", "->", "key"),
    ("EntitySource", "by_surface_form", "form", "text"),
    ("EntitySource", "by_surface_form", "->", "key"),
    ("EntitySource", "by_type", "type_id", "schema"),
    ("EntitySource", "by_type", "->", "key"),
    ("KeyCodec", "to_id", "key", "key"),
    ("KeyCodec", "to_id", "->", "door"),
    ("KeyCodec", "from_id", "rendered", "door"),
    ("KeyCodec", "from_id", "->", "key"),
    ("MatchSignal", "name", "->", "payload"),
    ("MatchSignal", "candidates", "query", "text"),
    ("MatchSignal", "candidates", "filter", "payload"),
    ("MatchSignal", "candidates", "->", "key"),
    ("MatchSignal", "candidates_many", "queries", "text"),
    ("MatchSignal", "candidates_many", "filter", "payload"),
    ("MatchSignal", "candidates_many", "->", "key"),
    ("MembershipOracle", "memberships", "entity", "key"),
    ("MembershipOracle", "memberships", "->", "payload"),
    ("MembershipOracle", "axes", "->", "payload"),
    ("ParentChoice", "choose", "node_id", "key"),
    ("ParentChoice", "choose", "parents", "key"),
    ("ParentChoice", "choose", "ctx", "key"),
    ("ParentChoice", "choose", "->", "key"),
    ("Assertion", "<field>", "id", "schema"),
    ("Assertion", "<field>", "subject", "key"),
    ("Assertion", "<field>", "object", "key"),
    ("Assertion", "<field>", "metadata", "payload"),
    ("Assertion", "<field>", "provenance", "key"),
    ("Assertion", "<field>", "derived_from", "schema"),
    ("Coverage", "<field>", "beyond_authority", "key"),
    ("Entity", "<field>", "id", "key"),
    ("Entity", "<field>", "type", "schema"),
    ("Entity", "<field>", "name", "text"),
    ("Entity", "<field>", "aliases", "text"),
    ("Entity", "<field>", "description", "text"),
    ("Entity", "<field>", "metadata", "payload"),
    ("EntityCandidate", "<field>", "entity_id", "key"),
    ("EntityRef", "<field>", "entity_id", "key"),
    ("Literal", "<field>", "unit", "payload"),
    ("Literal", "<field>", "metadata", "payload"),
    ("Literal", "as_field", "name", "payload"),
    ("MatchEvidence", "<field>", "signal", "payload"),
    ("MatchEvidence", "<field>", "matched_text", "text"),
    ("ProjectionContext", "<field>", "taxonomy_id", "schema"),
    ("ProjectionContext", "<field>", "roots", "key"),
    ("ProjectionContext", "<field>", "depths", "key"),
    ("ProjectionContext", "<field>", "types", "key"),
    ("Provenance", "<field>", "resolution", "key"),
    ("Provenance", "<field>", "asserted_by", "payload"),
    ("Provenance", "<field>", "derivation", "payload"),
    ("RelationType", "<field>", "type", "schema"),
    ("RelationType", "<field>", "domain", "schema"),
    ("RelationType", "<field>", "range", "schema"),
    ("RelationType", "<field>", "inverse_of", "schema"),
    ("ResolutionRef", "<field>", "query", "text"),
    ("ResolutionRef", "<field>", "entity_id", "key"),
    ("ResolutionRef", "<field>", "signal", "payload"),
    ("ResolutionRef", "<field>", "corpus", "payload"),
    ("ResolutionRef", "<field>", "runners_up", "key"),
    ("ResolutionResult", "<field>", "candidates", "key"),
    ("ResolutionResult", "<field>", "query", "text"),
    ("ResolutionResult", "<field>", "coverage", "key"),
    ("ResolutionResult", "ranked", "->", "key"),
    ("ResolutionResult", "matched_text", "->", "text"),
    ("ResolutionResult", "unmatched_text", "->", "text"),
    ("ResolutionResult", "as_distribution", "->", "key"),
    ("ResolutionResult", "explain", "entity_id", "key"),
    ("RunnerUp", "<field>", "entity_id", "key"),
    ("SourceDescription", "<field>", "source_id", "schema"),
    ("SourceDescription", "<field>", "backend", "payload"),
    ("SourceDescription", "<field>", "table", "payload"),
    ("SourceDescription", "<field>", "projection", "payload"),
    ("SourceDescription", "<field>", "declares", "schema"),
    ("SourceRef", "<field>", "source_id", "schema"),
    ("SourceRef", "<field>", "kind", "payload"),
    ("SourceRef", "<field>", "locator", "payload"),
    ("SourceRef", "<field>", "projection_id", "payload"),
)


#: What the population measures, pinned so that it moving is visible.
#:
#: Not a target and not a budget -- a tripwire. A protocol added to either
#: subpackage, or a value type newly reachable from one, changes these numbers,
#: and the pass that makes that change is the pass that should be reading the
#: rows it brings with it.
_PROTOCOLS = 13
_REACHABLE_VALUE_TYPES = 20
_ROWS = 123


def _modules() -> Iterator[ast.Module]:
    for root in _ROOTS:
        for path in sorted(root.glob("*.py")):
            yield ast.parse(path.read_text(encoding="utf-8"))


def _named(annotation: str) -> set[str]:
    """Every bare name an annotation mentions."""
    return {
        node.id
        for node in ast.walk(ast.parse(annotation, mode="eval").body)
        if isinstance(node, ast.Name)
    }


def _rows_of(node: ast.ClassDef) -> list[tuple[str, str, str]]:
    """Every annotated position on a class: parameters, returns and fields.

    One row per *annotation* rather than per ``str`` in it. A row may still be
    mixed -- ``Mapping[K, str]`` on ``ProjectionContext.types`` is a node key
    and a type id in one annotation -- and its verdict describes what moved.
    """
    rows: list[tuple[str, str, str]] = []
    for item in node.body:
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
            arguments = item.args
            for arg in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]:
                if arg.annotation is not None:
                    rows.append((item.name, arg.arg, ast.unparse(arg.annotation)))
            if item.returns is not None:
                rows.append((item.name, "->", ast.unparse(item.returns)))
        elif isinstance(item, ast.AnnAssign):
            rows.append(("<field>", ast.unparse(item.target), ast.unparse(item.annotation)))
    return rows


def _population() -> tuple[dict[str, ast.ClassDef], list[str], list[str]]:
    """The protocols, and the concrete types their annotations reach.

    Computed rather than listed, so a protocol added to either subpackage
    arrives in the population without anyone remembering to add it -- and its
    rows then fail as unclassified rather than going unchecked.
    """
    classes: dict[str, ast.ClassDef] = {}
    bases: dict[str, list[str]] = {}
    aliases: dict[str, str] = {}
    for module in _modules():
        for node in module.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node
                bases[node.name] = [ast.unparse(base) for base in node.bases]
            elif isinstance(node, ast.TypeAlias):
                aliases[ast.unparse(node.name)] = ast.unparse(node.value)
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                aliases[node.targets[0].id] = ast.unparse(node.value)

    protocols = sorted(n for n in classes if any(b.startswith("Protocol") for b in bases[n]))

    def reached(annotation: str) -> set[str]:
        """Classes an annotation names, following a module-level alias once.

        One hop rather than a closure, because the aliases here are unions of
        classes -- ``Term`` and ``ScopeAuthority`` -- and an alias of an alias
        would be a shape nothing in these files has.
        """
        found: set[str] = set()
        for name in _named(annotation):
            if name in classes:
                found.add(name)
            elif name in aliases:
                found |= {inner for inner in _named(aliases[name]) if inner in classes}
        return found

    concrete = {n for n in classes if n not in protocols}
    frontier = [
        c
        for p in protocols
        for _, _, annotation in _rows_of(classes[p])
        for c in reached(annotation)
        if c in concrete
    ]
    closure = set(frontier)
    while frontier:
        for _, _, annotation in _rows_of(classes[frontier.pop()]):
            for c in reached(annotation):
                if c in concrete and c not in closure:
                    closure.add(c)
                    frontier.append(c)
    return classes, protocols, sorted(closure)


def _measured() -> dict[tuple[str, str, str], str]:
    """Every row in the tree, mapped to the annotation the source carries."""
    classes, protocols, values = _population()
    return {
        (owner, member, position): annotation
        for owner in [*protocols, *values]
        for member, position, annotation in _rows_of(classes[owner])
        if {"str", "K", "K_co"} & _named(annotation)
    }


def test_the_population_is_what_the_table_was_written_against() -> None:
    """The three counts, so that the population moving is a failure rather than a surprise."""
    _, protocols, values = _population()

    assert len(protocols) == _PROTOCOLS, sorted(protocols)
    assert len(values) == _REACHABLE_VALUE_TYPES, sorted(values)
    assert len(_VERDICTS) == _ROWS


def test_every_annotation_in_the_population_has_a_verdict() -> None:
    """A row nobody classified fails the run.

    The direction that makes this a check rather than a record: a member added
    to one of these surfaces, or a field added to a value type one of them
    carries, arrives here unclassified and stops the suite until somebody has
    decided which space its ``str`` is in.
    """
    measured = _measured()
    declared = {(owner, member, position) for owner, member, position, _ in _VERDICTS}

    assert set(measured) - declared == set(), "rows in the tree that no verdict claims"
    assert declared - set(measured) == set(), "verdicts for rows the tree does not have"


@pytest.mark.parametrize(
    ("owner", "member", "position", "verdict"),
    _VERDICTS,
    ids=[f"{o}.{m}.{p}" for o, m, p, _ in _VERDICTS],
)
def test_each_row_is_spelled_the_way_its_verdict_says(
    owner: str, member: str, position: str, verdict: str
) -> None:
    """Both directions, one assertion each, per row.

    Parametrised rather than looped so that a failure names the row. A loop
    over a hundred and twenty rows reports the first one and says nothing about
    how many others are wrong, which is the report a reviewer of a mechanical
    pass most needs.
    """
    annotation = _measured()[owner, member, position]
    carries = "K" in _named(annotation) or "K_co" in _named(annotation)

    if verdict == _CARRIES_THE_KEY:
        assert carries, (
            f"{owner}.{member} {position} is declared a key and is spelled "
            f"{annotation!r}. A key left as `str` pins K = str for this call and "
            f"breaks nothing until a consumer binds something else"
        )
    else:
        assert not carries, (
            f"{owner}.{member} {position} is declared {verdict!r} -- it stays "
            f"`str` -- and is spelled {annotation!r}"
        )


def test_no_generic_is_named_bare_inside_the_population() -> None:
    """The third direction: a generic named without its parameter binds ``str`` silently.

    ``Assertion`` where ``Assertion[K]`` was meant mentions neither ``str`` nor
    ``K``, so the two tests above walk straight past it -- and the default on
    the key parameter then makes it ``Assertion[str]``, inside a surface whose
    whole point is that it is not. Nothing reports it and every test passes,
    because every test binds ``str``.

    **Scoped to the population deliberately.** A bare ``Entity`` inside
    ``MappingEntitySource`` is *correct*: a concrete that binds ``K = str`` is
    right whichever way the axis goes, which is what makes the concrete half of
    the widening free. It is only inside the polymorphic surfaces that a bare
    generic is a pin.
    """
    classes, protocols, values = _population()
    generic = {
        name
        for name, node in classes.items()
        if any(ast.unparse(b).startswith(("Protocol[", "Generic[")) for b in node.bases)
    }

    bare: list[str] = []
    for owner in [*protocols, *values]:
        for member, position, annotation in _rows_of(classes[owner]):
            tree = ast.parse(annotation, mode="eval").body
            parameterised = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Subscript)}
            bare += [
                f"{owner}.{member} {position}: {annotation}"
                for n in ast.walk(tree)
                if isinstance(n, ast.Name) and n.id in generic and id(n) not in parameterised
            ]

    assert bare == [], (
        "a generic named without its parameter inside the key-polymorphic "
        f"surfaces, which binds it to `str`: {bare}"
    )
