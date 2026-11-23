"""Two source guards over every package: who reads a cause, and who vouches for text.

A ``PluginRegistry`` bounds a factory's error and keeps it on ``__cause__``.
Helpers in front of registries read it back by type, seven of them, and handed
a factory's text -- a connection URL included -- to their callers. The one
answer now is ``dataknobs_common.registry.surfaced_cause``, which decides by a
private mark saying dataknobs wrote the text. Each half needs a guard:

* **Nothing else reads ``__cause__``.** A new reader is how an eighth helper
  would reopen what the registry closed, and it would report green
  everywhere: a leak is only visible to a test that puts a secret in a
  factory's message.
* **The mark is set only where it was reviewed.** It vouches that an error's
  text names nothing a factory, driver or service supplied. A site setting it
  over text it did not write would turn every guard above into a pass-through,
  so the population is a list someone reads, not a property of the tree.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from dataknobs_common.testing import assert_no_cause_read
from tests._workspace import ROOT

SOURCES = sorted((ROOT / "packages").glob("*/src"))

#: The readers of ``__cause__`` outside the registry module, each with what it
#: does with the cause and why that shows a caller no factory's text.
ALLOWED_CAUSE_READS = {
    # The registry module is where the decision is made, one function per
    # entry so a new reader there is reviewed like any other.
    # The answer itself: returns the cause only when dataknobs wrote it.
    "dataknobs_common/registry.py::surfaced_cause",
    # The create_surfacing body: asks surfaced_cause, and names the cause's
    # type in a bounded message.
    "dataknobs_common/registry.py::_raise_surfaced",
    # Tests only that a miss has no cause.
    "dataknobs_common/registry.py::_is_this_registrys_miss",
    # The doors' question: a refusal with nothing behind it, else surfaced_cause.
    "dataknobs_common/registry.py::_shown_failure",
    # The walk for a value-free config refusal. Reads marks and returns
    # only the words a mark recorded, never an instance's text.
    "dataknobs_common/exceptions.py::_value_free_refusal_behind",
    # The guard's own vocabulary: the attribute names it looks for, as strings.
    "dataknobs_common/testing/cause_reads.py::<module>",
    # Appends the cause to a server-side log line. The response body is built
    # from the policy row, never from this.
    "dataknobs_bots/api/exceptions.py::_log_handled_error",
    # Finds a TypeError in the chain and tests its text for one phrase, then
    # emits a hint dataknobs authored; the cause's text is never shown.
    "dataknobs_bots/reasoning/wizard_response.py::_maybe_strict_signature_hint",
}


def test_nothing_reads_a_cause_but_the_reviewed_readers() -> None:
    assert_no_cause_read(*SOURCES, allowed=ALLOWED_CAUSE_READS)


#: Every place that marks an error as text dataknobs wrote, by file and
#: enclosing function, with how many marks each holds. A new marking site is a
#: claim that its text names only what a config's author wrote or what
#: dataknobs knows, and it needs reading before it joins this list.
MARK_SITES: dict[tuple[str, str], int] = {
    # The one amender that marks again, and only an error marked on entry.
    ("dataknobs_common/exceptions.py", "naming_refusals"): 1,
    # Every error a registry raises: names, keys and types, all its own.
    ("dataknobs_common/registry.py", "_ours"): 1,
    # A helper's own "Unknown <subject> type" text, built from names.
    ("dataknobs_common/registry.py", "_raise_surfaced"): 1,
    # Dotted-path refusals: a module path, a name, and a type.
    ("dataknobs_common/imports.py", "_split"): 2,
    ("dataknobs_common/imports.py", "resolve_dotted"): 5,
    ("dataknobs_common/imports.py", "resolve_callable"): 1,
    ("dataknobs_common/imports.py", "resolve_class"): 1,
    ("dataknobs_common/imports.py", "resolve_optional_callable"): 1,
    # The rungs' construction-time refusals of a document.
    ("dataknobs_common/entity_resolution/signals.py", "_checked_callable"): 1,
    ("dataknobs_common/entity_resolution/signals.py", "_checked_count"): 2,
    ("dataknobs_common/entity_resolution/signals.py", "_checked_threshold"): 2,
    ("dataknobs_common/entity_resolution/signals.py", "_refuse_catalogue"): 1,
    ("dataknobs_common/entity_resolution/signals.py", "_refuse_unfolded"): 1,
    ("dataknobs_common/entity_resolution/signals.py", "_require_flavour"): 1,
    ("dataknobs_common/entity_resolution/signals.py", "refuse_negative_k"): 1,
    # Tokenizer and variant-rule doors and their built-in factories.
    ("dataknobs_common/text/tokenizers.py", "_default_factory"): 1,
    ("dataknobs_common/text/tokenizers.py", "_words_factory"): 2,
    ("dataknobs_common/text/tokenizers.py", "tokenizer_from_config"): 5,
    ("dataknobs_common/text/variants.py", "_hyphen_factory"): 1,
    ("dataknobs_common/text/variants.py", "_inflect_factory"): 1,
    ("dataknobs_common/text/variants.py", "_parameterless.factory"): 1,
    ("dataknobs_common/text/variants.py", "_parameters"): 2,
    ("dataknobs_common/text/variants.py", "_strip_prefix_factory"): 1,
    ("dataknobs_common/text/variants.py", "rules_from_config"): 6,
    ("dataknobs_common/text/variants.py", "split"): 1,
    ("dataknobs_common/text/variants.py", "strip_prefix"): 1,
    # Backend selection: an unknown or uninstalled backend name.
    ("dataknobs_data/backend_selection.py", "select_backend"): 3,
    # The shared `schema:` reader's refusals of a declaration.
    ("dataknobs_data/database.py", "extract_schema_from_config"): 1,
    ("dataknobs_data/schema.py", "DatabaseSchema.from_dict"): 3,
    ("dataknobs_data/schema.py", "_declared_name"): 3,
    ("dataknobs_data/schema.py", "_field_schema"): 6,
    ("dataknobs_data/schema.py", "read_field_declarations"): 4,
    # The semantic rung's refusal of its handles, and its index mismatch.
    ("dataknobs_data/entity_resolution.py", "SemanticSignal.__init__"): 1,
    ("dataknobs_data/entity_resolution.py", "_make_semantic"): 2,
    # A data backend's refusals of its own config, raised while it is built:
    # the native layout's columns, key and scope, the layout reader, the
    # config classes' checks, and a registered object with no from_config.
    ("dataknobs_data/backend_selection.py", "build_backend"): 1,
    ("dataknobs_data/backends/column_layout.py", "NativeColumnLayout.__init__"): 4,
    ("dataknobs_data/backends/column_layout.py", "NativeColumnLayout._check_scope_filter"): 2,
    ("dataknobs_data/backends/column_layout.py", "NativeColumnLayout._column"): 1,
    ("dataknobs_data/backends/column_layout.py", "_resolve_column"): 3,
    ("dataknobs_data/backends/column_layout.py", "read_layout_config.refuse"): 1,
    ("dataknobs_data/backends/column_layout.py", "resolve_layout"): 1,
    ("dataknobs_data/backends/config.py", "ColumnLayoutConfig._refuse_under_native"): 1,
    ("dataknobs_data/backends/config.py", "ElasticsearchDatabaseConfigBase.__post_init__"): 2,
    ("dataknobs_data/backends/config.py", "PostgresDatabaseConfig.__post_init__"): 1,
    ("dataknobs_data/backends/config.py", "S3DatabaseConfigBase.__post_init__"): 1,
    ("dataknobs_data/backends/postgres_mixins.py", "validate_pg_identifier"): 2,
    # The binding resolver's refusal of an async factory on its sync path,
    # naming the factory's type and the resource; and the async-only factory's
    # own "use resolve_async()" text.
    ("dataknobs_config/binding_resolver.py", "_refuse_awaitable"): 1,
    ("dataknobs_config/binding_resolver.py", "AsyncCallableFactory.create"): 1,
    # An unknown LLM provider name.
    ("dataknobs_llm/llm/providers/__init__.py", "LLMProviderFactory.create"): 1,
    # The composite intent classifier's refusals of its child specs.
    ("dataknobs_llm/intent/composite.py", "CompositeIntentClassifier.__init__"): 1,
    ("dataknobs_llm/intent/registry.py", "_composite_factory"): 2,
    # Memory and source backends' construction refusals.
    ("dataknobs_bots/memory/summary.py", "_resolve_summary_llm"): 1,
    ("dataknobs_bots/memory/composite.py", "CompositeMemory._validate_and_bind"): 2,
    ("dataknobs_bots/memory/composite.py", "CompositeMemory._ainit"): 1,
    ("dataknobs_bots/knowledge/sources/factory.py", "create_source_from_config"): 1,
    ("dataknobs_bots/knowledge/sources/factory.py", "_create_vector_kb_source"): 1,
    ("dataknobs_bots/knowledge/sources/factory.py", "_create_database_source"): 3,
}

SETTER = "_written_by_dataknobs"
ATTRIBUTE = "_WRITTEN_BY_DATAKNOBS"
#: The two marks' attribute names on an instance, each owned by one module.
#: Spelled out here rather than imported, so the guard reads what is written.
MARK_VALUES = {
    "_dataknobs_written_by_dataknobs": "dataknobs_common/exceptions.py",
    "_dataknobs_raised_by_a_registry": "dataknobs_common/registry.py",
}


def _owners(tree: ast.AST) -> dict[ast.AST, str]:
    owner: dict[ast.AST, str] = {}

    def visit(node: ast.AST, scope: list[str], function: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, [*scope, child.name], function)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                visit(child, [*scope, child.name], ".".join([*scope, child.name]))
            else:
                owner[child] = function
                visit(child, scope, function)

    visit(tree, [], "<module>")
    return owner


def _module_path(path: Path) -> str:
    return path.as_posix().split("/src/", 1)[1]


def test_the_mark_is_set_only_at_the_reviewed_sites() -> None:
    found: Counter[tuple[str, str]] = Counter()
    for source in SOURCES:
        for path in sorted(source.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            owners = _owners(tree)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name | ast.Attribute)
                    and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr)
                    == SETTER
                ):
                    found[(_module_path(path), owners.get(node, "<module>"))] += 1

    assert dict(found) == MARK_SITES


def test_the_marks_attribute_is_written_only_by_its_setter() -> None:
    """A direct ``setattr`` would vouch without the snapshot the reader compares."""
    writers: list[str] = []
    for source in SOURCES:
        for path in sorted(source.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            owners = _owners(tree)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "setattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Name)
                    and node.args[1].id == ATTRIBUTE
                ):
                    writers.append(f"{_module_path(path)}::{owners.get(node, '<module>')}")

    assert writers == ["dataknobs_common/exceptions.py::_written_by_dataknobs"]


def test_neither_mark_is_set_by_a_route_the_census_cannot_see() -> None:
    """The census counts calls by name; these would set a mark without one.

    An aliased import or the setter passed as a value makes a call the count
    misses; a string or attribute naming a mark's attribute outside its own
    module is a direct write; ``_ours`` sets both marks, so it is called in
    the registry module and nowhere else.
    """
    evasions: list[str] = []
    for source in SOURCES:
        for path in sorted(source.rglob("*.py")):
            module = _module_path(path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
            for node in ast.walk(tree):
                where = f"{module}:{getattr(node, 'lineno', 0)}"
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name in {SETTER, "_ours"} and alias.asname:
                            evasions.append(f"{where} imports {alias.name} as {alias.asname}")
                        if alias.name == "_ours" and module != "dataknobs_common/registry.py":
                            evasions.append(f"{where} imports _ours")
                named = (
                    node.id
                    if isinstance(node, ast.Name)
                    else node.attr
                    if isinstance(node, ast.Attribute)
                    else None
                )
                if named == SETTER:
                    parent = parents.get(node)
                    if not (isinstance(parent, ast.Call) and parent.func is node):
                        evasions.append(f"{where} uses {SETTER} other than by calling it")
                if named == "_ours" and module != "dataknobs_common/registry.py":
                    evasions.append(f"{where} uses _ours")
                if isinstance(node, ast.Constant) and node.value in MARK_VALUES:
                    if module != MARK_VALUES[node.value]:
                        evasions.append(f"{where} spells the mark attribute {node.value!r}")
                if isinstance(node, ast.Attribute) and node.attr in MARK_VALUES:
                    evasions.append(f"{where} reaches the mark attribute {node.attr!r} directly")

    assert evasions == []
