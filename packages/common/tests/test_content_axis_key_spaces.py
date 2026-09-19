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
import functools
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
#:             Six rows: the two on the codec, which is what renders
#:             it, and the four doors it is spent at -- ``qualify``
#:             out and ``localize`` back, on each flavour of ontology
#: ``bound``   a key parameter **bound** to ``str`` by a door that
#:             knows what the keys are. A document's ids are the
#:             strings its author typed, and the shipped cascade is
#:             ``str``-keyed, so both loader doors and both resolver
#:             doors say so in the annotation rather than leaving the
#:             default to say it silently
#: =========== ====================================================
_CARRIES_THE_KEY = "key"

#: Every ``str``- or ``K``-mentioning annotation in the population, with its
#: verdict. Hand-declared: a table derived from the tree would agree with the
#: tree by construction, which is the one thing it must not do.
_VERDICTS: tuple[tuple[str, str, str, str], ...] = (
    ("<module>", "async_build_resolver", "config", "payload"),
    ("<module>", "async_build_resolver", "ontology", "bound"),
    ("<module>", "async_build_resolver", "->", "bound"),
    ("<module>", "async_load_ontology", "normalizer", "text"),
    ("<module>", "async_load_ontology", "source", "payload"),
    ("<module>", "async_load_ontology", "->", "bound"),
    # The assembly the three doors share. `bound` throughout, for the doors'
    # reason and by the same route: the sources handed in are the ones a door
    # bound over a document, and a document's ids are the strings its author
    # typed. The predicate these two call -- which axes to copy -- was briefly
    # published for the third door and is private again now that the door
    # calls the assembly instead, so its two rows are gone from here.
    #
    # `structures` is the same reading for a third collaborator: it is keyed by
    # the *name* an axis is reached under, which is a taxonomy id and so
    # `schema`-space, and its values are axes over the keys the sources beside
    # it speak. Both halves are the document's, so `bound` covers the row.
    ("<module>", "assemble_async_ontology", "assertions", "bound"),
    ("<module>", "assemble_async_ontology", "entities", "bound"),
    ("<module>", "assemble_async_ontology", "structures", "bound"),
    ("<module>", "assemble_async_ontology", "->", "bound"),
    ("<module>", "assemble_ontology", "assertions", "bound"),
    ("<module>", "assemble_ontology", "entities", "bound"),
    ("<module>", "assemble_ontology", "structures", "bound"),
    ("<module>", "assemble_ontology", "->", "bound"),
    ("<module>", "build_resolver", "config", "payload"),
    ("<module>", "build_resolver", "ontology", "bound"),
    ("<module>", "build_resolver", "->", "bound"),
    ("<module>", "declared_candidates", "found", "key"),
    ("<module>", "declared_candidates", "signal", "schema"),
    ("<module>", "declared_candidates", "query", "text"),
    ("<module>", "declared_candidates", "admitted", "key"),
    ("<module>", "declared_candidates", "->", "key"),
    # A registration's metadata: the `str` is the key a door looks a fact up
    # by -- "flavour", "needs_io", "reads_surface_forms", or one a consumer
    # coins for their own rung. An open label, which is what every other
    # `metadata` row here is classified as.
    ("<module>", "declared_signal_metadata", "base", "payload"),
    ("<module>", "declared_signal_metadata", "->", "payload"),
    ("<module>", "finish", "->", "bound"),
    ("<module>", "load_ontology", "normalizer", "text"),
    ("<module>", "load_ontology", "source", "payload"),
    ("<module>", "load_ontology", "->", "bound"),
    ("<module>", "merge_rung", "produced", "bound"),
    ("<module>", "merge_rung", "signal", "schema"),
    ("<module>", "object_entity_id", "term", "key"),
    ("<module>", "object_entity_id", "->", "key"),
    ("<module>", "qualify", "local_id", "door"),
    ("<module>", "qualify", "ontology_id", "schema"),
    ("<module>", "qualify", "source_id", "schema"),
    ("<module>", "qualify", "->", "door"),
    ("<module>", "relation_id", "->", "schema"),
    ("<module>", "split_qualified", "qualified_id", "door"),
    ("<module>", "split_qualified", "source_ids", "schema"),
    ("<module>", "within_admits", "axes", "schema"),
    ("<module>", "within_admits", "memberships", "schema"),
    ("<module>", "within_axes", "->", "schema"),
    ("<module>", "within_axis_names", "->", "schema"),
    ("<module>", "within_memberships", "entity", "key"),
    ("<module>", "within_memberships", "source", "key"),
    ("<module>", "within_memberships", "->", "schema"),
    ("<module>", "refuse_unknown_axes", "axes", "schema"),
    ("AliasFormSource", "by_alias_form", "->", "key"),
    ("AliasFormSource", "by_alias_form", "form", "text"),
    ("SurfaceFormCatalog", "surface_forms", "->", "text"),
    ("Assertion", "<field>", "derived_from", "schema"),
    ("Assertion", "<field>", "id", "schema"),
    ("Assertion", "<field>", "metadata", "payload"),
    ("Assertion", "<field>", "object", "key"),
    ("Assertion", "<field>", "provenance", "key"),
    ("Assertion", "<field>", "subject", "key"),
    ("AssertionHierarchy", "<field>", "source", "key"),
    ("AssertionHierarchy", "_find", "->", "key"),
    ("AssertionHierarchy", "_find", "object", "key"),
    ("AssertionHierarchy", "_find", "subject", "key"),
    ("AssertionHierarchy", "_find_many", "->", "key"),
    ("AssertionHierarchy", "_find_many", "objects", "key"),
    ("AssertionHierarchy", "_find_many", "subjects", "key"),
    ("AssertionHierarchy", "children", "->", "key"),
    ("AssertionHierarchy", "children", "node_id", "key"),
    ("AssertionHierarchy", "children_many", "->", "key"),
    ("AssertionHierarchy", "children_many", "node_ids", "key"),
    ("AssertionHierarchy", "contains", "node_id", "key"),
    ("AssertionHierarchy", "parent_edges", "->", "key"),
    ("AssertionHierarchy", "parents", "->", "key"),
    ("AssertionHierarchy", "parents", "node_id", "key"),
    ("AssertionHierarchy", "parents_many", "->", "key"),
    ("AssertionHierarchy", "parents_many", "node_ids", "key"),
    ("AssertionHierarchy", "roots", "->", "key"),
    ("AssertionSource", "find", "->", "key"),
    ("AssertionSource", "find", "object", "key"),
    ("AssertionSource", "find", "subject", "key"),
    ("AssertionSource", "find_many", "->", "key"),
    ("AssertionSource", "find_many", "objects", "key"),
    ("AssertionSource", "find_many", "subjects", "key"),
    ("AssertionSource", "get", "->", "key"),
    ("AssertionSource", "get", "assertion_id", "schema"),
    ("AsyncAliasFormSource", "by_alias_form", "->", "key"),
    ("AsyncAliasFormSource", "by_alias_form", "form", "text"),
    # The two catalogue protocols answer with **forms**, so their one member
    # mentions no key at all -- which is why neither is generic where the two
    # alias-form protocols above are. `text` rather than `key`: a surface form
    # is what a person typed or wrote, and it stays `str` however the entities
    # it resolves to are keyed.
    ("AsyncSurfaceFormCatalog", "surface_forms", "->", "text"),
    ("AsyncAssertionHierarchy", "<field>", "source", "key"),
    ("AsyncAssertionHierarchy", "_find", "->", "key"),
    ("AsyncAssertionHierarchy", "_find", "object", "key"),
    ("AsyncAssertionHierarchy", "_find", "subject", "key"),
    ("AsyncAssertionHierarchy", "_find_many", "->", "key"),
    ("AsyncAssertionHierarchy", "_find_many", "objects", "key"),
    ("AsyncAssertionHierarchy", "_find_many", "subjects", "key"),
    ("AsyncAssertionHierarchy", "children", "->", "key"),
    ("AsyncAssertionHierarchy", "children", "node_id", "key"),
    ("AsyncAssertionHierarchy", "children_many", "->", "key"),
    ("AsyncAssertionHierarchy", "children_many", "node_ids", "key"),
    ("AsyncAssertionHierarchy", "contains", "node_id", "key"),
    ("AsyncAssertionHierarchy", "parent_edges", "->", "key"),
    ("AsyncAssertionHierarchy", "parents", "->", "key"),
    ("AsyncAssertionHierarchy", "parents", "node_id", "key"),
    ("AsyncAssertionHierarchy", "parents_many", "->", "key"),
    ("AsyncAssertionHierarchy", "parents_many", "node_ids", "key"),
    ("AsyncAssertionHierarchy", "roots", "->", "key"),
    ("AsyncAssertionSource", "find", "->", "key"),
    ("AsyncAssertionSource", "find", "object", "key"),
    ("AsyncAssertionSource", "find", "subject", "key"),
    ("AsyncAssertionSource", "find_many", "->", "key"),
    ("AsyncAssertionSource", "find_many", "objects", "key"),
    ("AsyncAssertionSource", "find_many", "subjects", "key"),
    ("AsyncAssertionSource", "get", "->", "key"),
    ("AsyncAssertionSource", "get", "assertion_id", "schema"),
    ("AsyncEntityResolver", "resolve", "->", "key"),
    ("AsyncEntityResolver", "resolve", "name", "text"),
    ("AsyncEntityResolver", "resolve_many", "->", "key"),
    ("AsyncEntityResolver", "resolve_many", "names", "text"),
    ("AsyncEntitySource", "by_surface_form", "->", "key"),
    ("AsyncEntitySource", "by_surface_form", "form", "text"),
    ("AsyncEntitySource", "by_type", "->", "key"),
    ("AsyncEntitySource", "by_type", "type_id", "schema"),
    ("AsyncEntitySource", "get", "->", "key"),
    ("AsyncEntitySource", "get", "entity_id", "key"),
    ("AsyncEntitySource", "get_many", "->", "key"),
    ("AsyncEntitySource", "get_many", "entity_ids", "key"),
    ("AsyncMatchSignal", "candidates", "->", "key"),
    ("AsyncMatchSignal", "candidates", "filter", "payload"),
    ("AsyncMatchSignal", "candidates", "query", "text"),
    ("AsyncMatchSignal", "candidates_many", "->", "key"),
    ("AsyncMatchSignal", "candidates_many", "filter", "payload"),
    ("AsyncMatchSignal", "candidates_many", "queries", "text"),
    ("AsyncMatchSignal", "name", "->", "payload"),
    ("AsyncOntology", "<field>", "assertions", "key"),
    ("AsyncOntology", "<field>", "codec", "key"),
    ("AsyncOntology", "<field>", "entities", "key"),
    ("AsyncOntology", "<field>", "entity_types", "schema"),
    ("AsyncOntology", "<field>", "id", "schema"),
    ("AsyncOntology", "<field>", "imports", "schema"),
    ("AsyncOntology", "<field>", "relation_types", "schema"),
    ("AsyncOntology", "<field>", "structures", "key"),
    ("AsyncOntology", "<field>", "taxonomies", "schema"),
    ("AsyncOntology", "<field>", "version", "payload"),
    ("AsyncOntology", "by_surface_form", "->", "key"),
    ("AsyncOntology", "by_surface_form", "form", "text"),
    ("AsyncOntology", "entity", "->", "key"),
    ("AsyncOntology", "entity", "entity_id", "key"),
    ("AsyncOntology", "inherited_attributes", "entity_type", "schema"),
    ("AsyncOntology", "localize", "->", "key"),
    ("AsyncOntology", "localize", "qualified_id", "door"),
    ("AsyncOntology", "qualify", "->", "door"),
    ("AsyncOntology", "qualify", "local_id", "key"),
    ("AsyncOntology", "taxonomy", "->", "key"),
    ("AsyncOntology", "taxonomy", "name", "schema"),
    ("AsyncTaxonomy", "<field>", "assertions", "key"),
    ("AsyncTaxonomy", "<field>", "entities", "key"),
    ("AsyncTaxonomy", "<field>", "entity_types", "schema"),
    ("AsyncTaxonomy", "<field>", "structure", "key"),
    ("AsyncTaxonomy", "at", "->", "key"),
    ("AsyncTaxonomy", "at", "node_id", "key"),
    ("AsyncTaxonomy", "inherited_attributes", "entity_type", "schema"),
    ("AsyncTaxonomy", "subtree_keys", "->", "key"),
    ("AsyncTaxonomy", "subtree_keys", "root_id", "key"),
    ("AsyncTaxonomy", "walk", "->", "key"),
    ("AsyncTaxonomy", "walk", "from_id", "key"),
    ("AsyncTaxonomyView", "<field>", "node", "key"),
    ("AsyncTaxonomyView", "<field>", "taxonomy", "key"),
    ("AsyncTaxonomyView", "_structural", "->", "key"),
    ("AsyncTaxonomyView", "_wrap", "->", "key"),
    ("AsyncTaxonomyView", "_wrap", "views", "key"),
    ("AsyncTaxonomyView", "ancestors", "->", "key"),
    ("AsyncTaxonomyView", "at", "->", "key"),
    ("AsyncTaxonomyView", "at", "node_id", "key"),
    ("AsyncTaxonomyView", "child_edges", "->", "key"),
    ("AsyncTaxonomyView", "children", "->", "key"),
    ("AsyncTaxonomyView", "descendants", "->", "key"),
    ("AsyncTaxonomyView", "children_at_depth", "->", "key"),
    ("AsyncTaxonomyView", "descendants_to_depth", "->", "key"),
    ("AsyncTaxonomyView", "entity", "->", "key"),
    ("AsyncTaxonomyView", "parent_edges", "->", "key"),
    ("AsyncTaxonomyView", "parents", "->", "key"),
    ("AsyncTaxonomyView", "paths_to_root", "->", "key"),
    ("AttributeDef", "<field>", "description", "text"),
    ("AttributeDef", "<field>", "entity_type", "schema"),
    ("AttributeDef", "<field>", "enum_values", "payload"),
    ("AttributeDef", "<field>", "name", "schema"),
    ("AttributeDef", "<field>", "value_type", "schema"),
    ("Coverage", "<field>", "beyond_authority", "key"),
    ("Entity", "<field>", "aliases", "text"),
    ("Entity", "<field>", "description", "text"),
    ("Entity", "<field>", "id", "key"),
    ("Entity", "<field>", "metadata", "payload"),
    ("Entity", "<field>", "name", "text"),
    ("Entity", "<field>", "type", "schema"),
    ("EntityCandidate", "<field>", "entity_id", "key"),
    ("EntityRef", "<field>", "entity_id", "key"),
    ("EntityResolver", "resolve", "->", "key"),
    ("EntityResolver", "resolve", "name", "text"),
    ("EntityResolver", "resolve_many", "->", "key"),
    ("EntityResolver", "resolve_many", "names", "text"),
    ("EntitySource", "by_surface_form", "->", "key"),
    ("EntitySource", "by_surface_form", "form", "text"),
    ("EntitySource", "by_type", "->", "key"),
    ("EntitySource", "by_type", "type_id", "schema"),
    ("EntitySource", "get", "->", "key"),
    ("EntitySource", "get", "entity_id", "key"),
    ("EntitySource", "get_many", "->", "key"),
    ("EntitySource", "get_many", "entity_ids", "key"),
    ("EntityType", "<field>", "isa", "schema"),
    ("EntityType", "<field>", "type", "schema"),
    ("FormHit", "<field>", "entity_id", "key"),
    ("KeyCodec", "from_id", "->", "key"),
    ("KeyCodec", "from_id", "rendered", "door"),
    ("KeyCodec", "to_id", "->", "door"),
    ("KeyCodec", "to_id", "key", "key"),
    ("Literal", "<field>", "metadata", "payload"),
    ("Literal", "<field>", "unit", "payload"),
    ("Literal", "as_field", "name", "payload"),
    ("MatchEvidence", "<field>", "matched_text", "text"),
    ("MatchEvidence", "<field>", "signal", "payload"),
    ("MatchSignal", "candidates", "->", "key"),
    ("MatchSignal", "candidates", "filter", "payload"),
    ("MatchSignal", "candidates", "query", "text"),
    ("MatchSignal", "candidates_many", "->", "key"),
    ("MatchSignal", "candidates_many", "filter", "payload"),
    ("MatchSignal", "candidates_many", "queries", "text"),
    ("MatchSignal", "name", "->", "payload"),
    ("MembershipOracle", "axes", "->", "payload"),
    ("MembershipOracle", "memberships", "->", "payload"),
    ("MembershipOracle", "memberships", "entity", "key"),
    ("Ontology", "<field>", "assertions", "key"),
    ("Ontology", "<field>", "codec", "key"),
    ("Ontology", "<field>", "entities", "key"),
    ("Ontology", "<field>", "entity_types", "schema"),
    ("Ontology", "<field>", "id", "schema"),
    ("Ontology", "<field>", "imports", "schema"),
    ("Ontology", "<field>", "relation_types", "schema"),
    ("Ontology", "<field>", "structures", "key"),
    ("Ontology", "<field>", "taxonomies", "schema"),
    ("Ontology", "<field>", "version", "payload"),
    ("Ontology", "by_surface_form", "->", "key"),
    ("Ontology", "by_surface_form", "form", "text"),
    ("Ontology", "entity", "->", "key"),
    ("Ontology", "entity", "entity_id", "key"),
    ("Ontology", "inherited_attributes", "entity_type", "schema"),
    ("Ontology", "localize", "->", "key"),
    ("Ontology", "localize", "qualified_id", "door"),
    ("Ontology", "qualify", "->", "door"),
    ("Ontology", "qualify", "local_id", "key"),
    ("Ontology", "taxonomy", "->", "key"),
    ("Ontology", "taxonomy", "name", "schema"),
    ("ParentChoice", "choose", "->", "key"),
    ("ParentChoice", "choose", "ctx", "key"),
    ("ParentChoice", "choose", "node_id", "key"),
    ("ParentChoice", "choose", "parents", "key"),
    ("ProjectionContext", "<field>", "depths", "key"),
    ("ProjectionContext", "<field>", "roots", "key"),
    ("ProjectionContext", "<field>", "taxonomy_id", "schema"),
    ("ProjectionContext", "<field>", "types", "key"),
    ("Provenance", "<field>", "asserted_by", "payload"),
    ("Provenance", "<field>", "derivation", "payload"),
    ("Provenance", "<field>", "resolution", "key"),
    ("RelationType", "<field>", "domain", "schema"),
    ("RelationType", "<field>", "inverse_of", "schema"),
    ("RelationType", "<field>", "range", "schema"),
    ("RelationType", "<field>", "type", "schema"),
    ("ResolutionRef", "<field>", "corpus", "payload"),
    ("ResolutionRef", "<field>", "entity_id", "key"),
    ("ResolutionRef", "<field>", "query", "text"),
    ("ResolutionRef", "<field>", "runners_up", "key"),
    ("ResolutionRef", "<field>", "signal", "payload"),
    ("ResolutionResult", "<field>", "candidates", "key"),
    ("ResolutionResult", "<field>", "coverage", "key"),
    ("ResolutionResult", "<field>", "query", "text"),
    ("ResolutionResult", "as_distribution", "->", "key"),
    ("ResolutionResult", "explain", "entity_id", "key"),
    ("ResolutionResult", "matched_text", "->", "text"),
    ("ResolutionResult", "ranked", "->", "key"),
    ("ResolutionResult", "unmatched_text", "->", "text"),
    ("RunnerUp", "<field>", "entity_id", "key"),
    ("SourceDescription", "<field>", "backend", "payload"),
    ("SourceDescription", "<field>", "declares", "schema"),
    ("SourceDescription", "<field>", "projection", "payload"),
    ("SourceDescription", "<field>", "source_id", "schema"),
    ("SourceDescription", "<field>", "table", "payload"),
    ("SourceRef", "<field>", "kind", "payload"),
    ("SourceRef", "<field>", "locator", "payload"),
    ("SourceRef", "<field>", "projection_id", "payload"),
    ("SourceRef", "<field>", "source_id", "schema"),
    ("Taxonomy", "<field>", "assertions", "key"),
    ("Taxonomy", "<field>", "entities", "key"),
    ("Taxonomy", "<field>", "entity_types", "schema"),
    ("Taxonomy", "<field>", "structure", "key"),
    ("Taxonomy", "at", "->", "key"),
    ("Taxonomy", "at", "node_id", "key"),
    ("Taxonomy", "inherited_attributes", "entity_type", "schema"),
    ("Taxonomy", "subtree_keys", "->", "key"),
    ("Taxonomy", "subtree_keys", "root_id", "key"),
    ("Taxonomy", "walk", "->", "key"),
    ("Taxonomy", "walk", "from_id", "key"),
    ("TaxonomyDefinition", "<field>", "description", "text"),
    ("TaxonomyDefinition", "<field>", "id", "schema"),
    ("TaxonomyDefinition", "<field>", "metadata", "payload"),
    ("TaxonomyDefinition", "<field>", "name", "text"),
    ("TaxonomyView", "<field>", "node", "key"),
    ("TaxonomyView", "<field>", "taxonomy", "key"),
    ("TaxonomyView", "_structural", "->", "key"),
    ("TaxonomyView", "_wrap", "->", "key"),
    ("TaxonomyView", "_wrap", "views", "key"),
    ("TaxonomyView", "ancestors", "->", "key"),
    ("TaxonomyView", "at", "->", "key"),
    ("TaxonomyView", "at", "node_id", "key"),
    ("TaxonomyView", "child_edges", "->", "key"),
    ("TaxonomyView", "children", "->", "key"),
    ("TaxonomyView", "descendants", "->", "key"),
    ("TaxonomyView", "children_at_depth", "->", "key"),
    ("TaxonomyView", "descendants_to_depth", "->", "key"),
    ("TaxonomyView", "entity", "->", "key"),
    ("TaxonomyView", "parent_edges", "->", "key"),
    ("TaxonomyView", "parents", "->", "key"),
    ("TaxonomyView", "paths_to_root", "->", "key"),
    ("TreeProjection", "<field>", "choice", "key"),
    ("TreeProjection", "<field>", "order_key", "payload"),
)


#: What the population measures, pinned so that it moving is visible.
#:
#: Not a target and not a budget -- a tripwire. A protocol added to either
#: subpackage, or a value type newly reachable from one, changes these numbers,
#: and the pass that makes that change is the pass that should be reading the
#: rows it brings with it.
_PROTOCOLS = 15
_REACHABLE_VALUE_TYPES = 36

#: Class rows plus the published module-level ones -- see :func:`_module_rows`
#: for why a function belonging to no class is in the population at all.
_ROWS = 320


def _modules() -> Iterator[ast.Module]:
    """Every module under the two roots, at any depth.

    ``rglob`` rather than ``glob``: both subpackages are flat today, so the
    two read alike -- and the day one grows a directory, the flat form would
    quietly stop measuring what is in it while every test here still passed.
    A guard that shrinks in silence is the failure this file exists to catch.
    """
    for root in _ROOTS:
        for path in sorted(root.rglob("*.py")):
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


def _alias_value(node: ast.expr) -> str:
    """An alias's right-hand side, seeing through an explicit ``TypeAliasType``.

    Three spellings reach this module and two of them are a bare expression.
    The third -- ``X = TypeAliasType("X", "<value>", type_params=(K,))`` -- is a
    *call*, so unparsing it yields the call rather than the union, and every
    name the alias reaches would drop out of the population in silence. The
    spelling exists because it is the only one that carries a PEP 696 default
    on a lazily-evaluated alias before 3.13, so it is the one a key-defaulted
    alias has to use, which makes it exactly the spelling this guard must not
    be blind to.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TypeAliasType"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return node.args[1].value
    return ast.unparse(node)


def _module_rows() -> list[tuple[str, str, str]]:
    """Every annotated position on a **published module-level function**.

    The other half of the surface, and the half the frontier could not reach:
    it is seeded from class annotations, so a function that belongs to no class
    was outside the population however many keys it carried.
    ``object_entity_id(term: Term[K]) -> K | None`` and
    ``within_memberships(entity: Entity[K], source: ScopeAuthority[K])`` are
    key-carrying surfaces a consumer calls directly, and a regression narrowing
    either return to ``str`` was invisible to all three tests here.

    **Published only, and the boundary is an argument rather than a
    convenience.** A module-private helper is not a surface: it is reached
    through the published function that calls it, whose row *is* in the table,
    so a helper narrowed to ``str`` shows up at the door above it. A private
    *method* stays in, because the class it hangs on is in the population by
    declaration -- the two halves are drawn by the same rule, which is whether
    something outside these files can name it.

    The owner is ``"<module>"`` rather than a file name: these two subpackages
    export from their package door, so which module a function is written in is
    not a fact a caller knows, and keying on it would make a row move when a
    function did.
    """
    rows: list[tuple[str, str, str]] = []
    for module in _modules():
        for node in module.body:
            if isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef
            ) and not node.name.startswith("_"):
                arguments = node.args
                for arg in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]:
                    if arg.annotation is not None:
                        rows.append((node.name, arg.arg, ast.unparse(arg.annotation)))
                if node.returns is not None:
                    rows.append((node.name, "->", ast.unparse(node.returns)))
    return rows


def _generic_aliases() -> set[str]:
    """Alias names that take a key parameter, in any of the three spellings.

    The blind spot :func:`test_no_generic_is_named_bare_inside_the_population`
    had, and the one that let a real regression through: that sweep built its
    set from class *bases*, so an alias was never in it, and a bare ``Term``
    or ``ScopeAuthority`` inside a polymorphic surface bound ``str`` with
    nothing to report it -- the same defect the test exists for, one node type
    away.
    """
    found: set[str] = set()
    for module in _modules():
        for node in module.body:
            if isinstance(node, ast.TypeAlias):
                if node.type_params:
                    found.add(ast.unparse(node.name))
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and {"K", "K_co"} & _named(_alias_value(node.value))
            ):
                found.add(node.targets[0].id)
    return found


@functools.cache  # the two subpackages' ASTs, parsed once: the row test below is parametrised 271 times
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
                # A name defined twice across the two roots would overwrite the
                # first silently, and the lost class's rows would vanish from
                # the table with nothing to report it. There are none today.
                assert node.name not in classes, (
                    f"{node.name} is declared twice across these subpackages; "
                    f"the population keys on the bare name and one would be lost"
                )
                classes[node.name] = node
                bases[node.name] = [ast.unparse(base) for base in node.bases]
            elif isinstance(node, ast.TypeAlias):
                aliases[ast.unparse(node.name)] = ast.unparse(node.value)
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                aliases[node.targets[0].id] = _alias_value(node.value)

    protocols = sorted(n for n in classes if any(b.startswith("Protocol") for b in bases[n]))

    #: Every class that declares a key parameter of its own, protocol or not.
    #:
    #: **The frontier was seeded from protocol annotations alone**, which made
    #: the population everything a *protocol* could reach and nothing else.
    #: ``Ontology``, ``Taxonomy``, both cursors and both assertion axes are
    #: named by no protocol member, so the nine classes carrying the codec and
    #: the cursor walks -- most of what the key parameter was added for -- sat
    #: outside the table while the docstring above claimed them. A generic
    #: class is polymorphic in the key by declaration, so it belongs in the
    #: population by the same rule a protocol does.
    #:
    #: Concrete ``str``-binding classes stay out, deliberately and for
    #: :func:`test_no_generic_is_named_bare_inside_the_population`'s reason: a
    #: bare ``Entity`` inside ``MappingEntitySource`` is *correct*, so sweeping
    #: those in would turn the guard's own rule into a hundred false rows.
    polymorphic = sorted(
        n
        for n in classes
        if any(b.startswith(("Protocol[", "Generic[")) for b in bases[n]) and n not in protocols
    )

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
        for p in [*protocols, *polymorphic]
        for _, _, annotation in _rows_of(classes[p])
        for c in reached(annotation)
        if c in concrete
    ]
    closure = set(frontier) | set(polymorphic)
    while frontier:
        for _, _, annotation in _rows_of(classes[frontier.pop()]):
            for c in reached(annotation):
                if c in concrete and c not in closure:
                    closure.add(c)
                    frontier.append(c)
    return classes, protocols, sorted(closure)


@functools.cache  # and the rows read off them once: the row test below is parametrised 271 times
def _measured() -> dict[tuple[str, str, str], str]:
    """Every row in the tree, mapped to the annotation the source carries."""
    classes, protocols, values = _population()
    measured = {
        (owner, member, position): annotation
        for owner in [*protocols, *values]
        for member, position, annotation in _rows_of(classes[owner])
        if {"str", "K", "K_co"} & _named(annotation)
    }
    measured.update(
        {
            ("<module>", member, position): annotation
            for member, position, annotation in _module_rows()
            if {"str", "K", "K_co"} & _named(annotation)
        }
    )
    return measured


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
    } | _generic_aliases()

    surfaces = [(owner, _rows_of(classes[owner])) for owner in [*protocols, *values]]
    surfaces.append(("<module>", _module_rows()))

    bare: list[str] = []
    for owner, rows in surfaces:
        for member, position, annotation in rows:
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
