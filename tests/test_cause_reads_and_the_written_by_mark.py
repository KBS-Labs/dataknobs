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

The mark is one of three ways to vouch, so there is a census of each: the
written-by mark, the private namer that re-marks an error it amends, and the
value-free mark on a config refusal. A census counts calls by name, so
``PRIVATE_NAMES`` also says where each name involved may appear at all; any
other mention is a route the counts would miss.

**Not covered:** a name assembled at run time (``getattr(e, "a" + "b")``), or
read out of a mapping by a computed key. No source scan sees one. These guards
stop an unreviewed site, not an author set on evading them.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path

import pytest

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
    # only text a mark vouches for: the outermost written-by link above the
    # refusal while it still says what it said when marked, else the words
    # the refusal's own mark recorded -- never an unvouched instance's text.
    "dataknobs_common/exceptions.py::_carried_refusal_words",
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
    # The one amender that marks again, and only an error marked on entry:
    # the private core, when its in-tree caller asks it to vouch for an
    # origin naming only what a config's author wrote (a source, a binding).
    # The public `naming_refusals` never vouches.
    ("dataknobs_common/exceptions.py", "_naming_refusals"): 1,
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
    ("dataknobs_data/backends/config.py", "S3DatabaseConfigBase.__post_init__"): 2,
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

#: Every caller of ``_naming_refusals_as_dataknobs``, by file and enclosing
#: function. The marking site above is one line inside the private core; the
#: vouching *decision* is made here, by whoever chooses the origin text. A new
#: caller is a claim that its origin and ``context`` name only what a config's
#: author wrote -- never a value a factory, driver or service supplied -- and
#: it needs reading before it joins this list.
VOUCHING_NAMERS: dict[tuple[str, str], int] = {
    # Origin: the grounded source's name, from the bot config.
    ("dataknobs_bots/knowledge/sources/factory.py", "_create_database_source"): 1,
    # Origin: the ontology's and the binding's ids, from the ontology document.
    ("dataknobs_data/ontology/registry.py", "OntologyRegistry._bind_record_source"): 1,
}

#: Every site that marks a config refusal value-free. The mark vouches that the
#: refusal's words name fields and types and no value, and a registry shows
#: them to its caller on that word alone.
VALUE_FREE_SITES: dict[tuple[str, str], int] = {
    # Every refusal ``StructuredConfig.from_dict`` writes is built here.
    ("dataknobs_common/structured_config.py", "_config_field_error"): 1,
}

#: The two callers of the private core. ``vouch=True`` there is the namer's
#: decision without the namer's census, so a third caller would be one.
NAMING_CORE_CALLERS: dict[tuple[str, str], int] = {
    ("dataknobs_common/exceptions.py", "naming_refusals"): 1,
    ("dataknobs_common/exceptions.py", "_naming_refusals_as_dataknobs"): 1,
}

#: Each censused function, with every call of it.
CENSUSES: dict[str, dict[tuple[str, str], int]] = {
    "_written_by_dataknobs": MARK_SITES,
    "_naming_refusals_as_dataknobs": VOUCHING_NAMERS,
    "_value_free": VALUE_FREE_SITES,
    "_naming_refusals": NAMING_CORE_CALLERS,
}

EXCEPTIONS = "dataknobs_common/exceptions.py"
REGISTRY = "dataknobs_common/registry.py"

#: Every private name that can set a mark or vouch for a text, and the one
#: module that may mention it. ``None`` means any module may, but only by
#: importing it unaliased and calling it: the forms a census counts. Spelled
#: out here rather than imported, so the guard reads what is written.
PRIVATE_NAMES: dict[str, str | None] = {
    "_written_by_dataknobs": None,
    "_naming_refusals_as_dataknobs": None,
    "_value_free": None,
    # The private core both namers share: ``vouch=True`` is the decision.
    "_naming_refusals": EXCEPTIONS,
    # Sets both of a registry's marks.
    "_ours": REGISTRY,
    # The constants holding the marks' attribute names.
    "_WRITTEN_BY_DATAKNOBS": EXCEPTIONS,
    "_VALUE_FREE_REFUSAL": EXCEPTIONS,
    "_RAISED_BY_A_REGISTRY": REGISTRY,
}

#: The marks' attribute names on an instance, each spelled once, by the module
#: that owns its constant, and reached only through that constant.
MARK_VALUES: dict[str, str] = {
    "_dataknobs_written_by_dataknobs": EXCEPTIONS,
    "_dataknobs_value_free_refusal": EXCEPTIONS,
    "_dataknobs_raised_by_a_registry": REGISTRY,
}

MARK_CONSTANTS = [name for name in PRIVATE_NAMES if name.isupper()]

#: Every use of a mark's constant and every spelling of its attribute name,
#: reads included, by module, function and what was written. Counted whole
#: because a write need not look like one: a constant bound to a local, or a
#: key into ``__dict__``, sets the mark as surely as ``setattr``.
MARK_USES: dict[tuple[str, str, str], int] = {
    # Each constant, defined once.
    (EXCEPTIONS, "<module>", "_WRITTEN_BY_DATAKNOBS"): 1,
    (EXCEPTIONS, "<module>", "_dataknobs_written_by_dataknobs"): 1,
    (EXCEPTIONS, "<module>", "_VALUE_FREE_REFUSAL"): 1,
    (EXCEPTIONS, "<module>", "_dataknobs_value_free_refusal"): 1,
    (REGISTRY, "<module>", "_RAISED_BY_A_REGISTRY"): 1,
    (REGISTRY, "<module>", "_dataknobs_raised_by_a_registry"): 1,
    # The one function that writes each.
    (EXCEPTIONS, "_written_by_dataknobs", "_WRITTEN_BY_DATAKNOBS"): 1,
    (EXCEPTIONS, "_value_free", "_VALUE_FREE_REFUSAL"): 1,
    (REGISTRY, "_ours", "_RAISED_BY_A_REGISTRY"): 1,
    # The readers: each mark's own, and the walk that passes a registry's own
    # miss through unwrapped.
    (EXCEPTIONS, "_is_written_by_dataknobs", "_WRITTEN_BY_DATAKNOBS"): 1,
    (EXCEPTIONS, "_value_free_words", "_VALUE_FREE_REFUSAL"): 1,
    (REGISTRY, "_raised_by_a_registry", "_RAISED_BY_A_REGISTRY"): 1,
    (REGISTRY, "PluginRegistry._raise_factory_failure", "_RAISED_BY_A_REGISTRY"): 1,
}

#: Strings that spell a private name without reaching it.
ALLOWED_SPELLINGS = {
    # A scanner's vocabulary: it reads source, and matches calls by name.
    ("dataknobs_common/testing/error_text.py", "_written_by_dataknobs"),
    ("dataknobs_common/testing/error_text.py", "_ours"),
}


def _module_path(path: Path) -> str:
    return path.as_posix().split("/src/", 1)[1]


@cache
def _parsed(source: Path) -> tuple[tuple[str, ast.Module], ...]:
    return tuple(
        (_module_path(path), ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        for path in sorted(source.rglob("*.py"))
    )


def _modules(sources: Iterable[Path]) -> Iterator[tuple[str, ast.Module]]:
    for source in sources:
        yield from _parsed(source)


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


def _named(node: ast.AST) -> str | None:
    """The bare name a ``Name`` or ``Attribute`` node mentions."""
    if isinstance(node, ast.Name):
        return node.id
    return node.attr if isinstance(node, ast.Attribute) else None


def _calls(name: str, sources: Iterable[Path]) -> dict[tuple[str, str], int]:
    """Every call of ``name``, by file and enclosing function."""
    found: Counter[tuple[str, str]] = Counter()
    for module, tree in _modules(sources):
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _named(n.func) == name]
        if calls:
            owners = _owners(tree)
            found.update((module, owners[call]) for call in calls)
    return dict(found)


def _mark_uses(sources: Iterable[Path]) -> dict[tuple[str, str, str], int]:
    """Every mention of a mark's constant or attribute string, by module and function."""
    found: Counter[tuple[str, str, str]] = Counter()
    for module, tree in _modules(sources):
        uses = [
            (node, written)
            for node in ast.walk(tree)
            if (written := _named(node)) in MARK_CONSTANTS
            or (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and (written := node.value) in MARK_VALUES
            )
        ]
        if uses:
            owners = _owners(tree)
            found.update((module, owners[node], str(written)) for node, written in uses)
    return dict(found)


def _bound_names(node: ast.AST) -> Iterator[str | None]:
    """The names a node gives to something, other than through an attribute."""
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, ast.keyword | ast.arg):
        yield node.arg
    elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        yield node.name
    elif isinstance(node, ast.alias):
        yield node.name
        yield node.asname


def _evasions(sources: Iterable[Path]) -> list[str]:
    """Every mention of a private name that a census could not count."""
    found: list[str] = []
    for module, tree in _modules(sources):
        called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        for node in ast.walk(tree):
            where = f"{module}:{getattr(node, 'lineno', 0)}"
            found.extend(
                f"{where} binds the mark attribute {name!r}"
                for name in _bound_names(node)
                if name in MARK_VALUES
            )
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name not in PRIVATE_NAMES:
                        continue
                    home = PRIVATE_NAMES[alias.name]
                    if home is not None and module != home:
                        found.append(f"{where} imports {alias.name}")
                    elif alias.asname not in (None, alias.name):
                        found.append(f"{where} imports {alias.name} as {alias.asname}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in MARK_VALUES:
                    if module != MARK_VALUES[node.value]:
                        found.append(f"{where} spells the mark attribute {node.value!r}")
                elif node.value in PRIVATE_NAMES and (module, node.value) not in ALLOWED_SPELLINGS:
                    found.append(f"{where} spells {node.value!r} in a string")
            elif (named := _named(node)) in PRIVATE_NAMES:
                home = PRIVATE_NAMES[named]
                if home is not None and module != home:
                    found.append(f"{where} uses {named}")
                elif named in CENSUSES and id(node) not in called:
                    found.append(f"{where} uses {named} other than by calling it")
            elif isinstance(node, ast.Attribute) and node.attr in MARK_VALUES:
                found.append(f"{where} reaches the mark attribute {node.attr!r} directly")
    return found


@pytest.mark.parametrize("name", CENSUSES)
def test_every_call_that_vouches_is_at_a_reviewed_site(name: str) -> None:
    """A new site is a claim about text, and needs reading before it joins its list."""
    assert _calls(name, SOURCES) == CENSUSES[name]


def test_each_mark_is_used_only_where_it_was_reviewed() -> None:
    """A write elsewhere would vouch without the snapshot the reader compares."""
    assert _mark_uses(SOURCES) == MARK_USES


def test_no_private_name_is_mentioned_where_a_census_cannot_count_it() -> None:
    assert _evasions(SOURCES) == []


# ---------------------------------------------------------------------------
# The guards themselves: each route planted alone in a tree of its own
# ---------------------------------------------------------------------------

FROM_EXCEPTIONS = "from dataknobs_common.exceptions import "
FROM_REGISTRY = "from dataknobs_common.registry import "
AS_MODULE = "from dataknobs_common import exceptions as e\n"
CENSUSED_ANYWHERE = [name for name, home in PRIVATE_NAMES.items() if home is None]
OWNED = [(name, home) for name, home in PRIVATE_NAMES.items() if home is not None]


def _planted(tmp_path: Path, text: str, module: str = "planted/m.py") -> list[Path]:
    path = tmp_path / "src" / module
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return [tmp_path / "src"]


@pytest.mark.parametrize("name", CENSUSES)
@pytest.mark.parametrize(
    "template",
    [
        FROM_EXCEPTIONS + "{0}\ndef f(x):\n    return {0}(x)\n",
        AS_MODULE + "def f(x):\n    return e.{0}(x)\n",
        FROM_EXCEPTIONS + "{0}\ndef f(x):\n    with {0}(x):\n        pass\n",
    ],
    ids=["call", "attribute-call", "with"],
)
def test_a_planted_call_is_counted(tmp_path: Path, name: str, template: str) -> None:
    sources = _planted(tmp_path, template.format(name))

    assert _calls(name, sources) == {("planted/m.py", "f"): 1}


@pytest.mark.parametrize("name", CENSUSED_ANYWHERE)
def test_an_unaliased_import_and_a_direct_call_is_the_allowed_form(
    tmp_path: Path, name: str
) -> None:
    text = FROM_EXCEPTIONS + f"{name}\ndef f(x):\n    return {name}(x)\n"

    assert _evasions(_planted(tmp_path, text)) == []


@pytest.mark.parametrize("name", PRIVATE_NAMES)
@pytest.mark.parametrize(
    "template",
    [
        FROM_EXCEPTIONS + "{0} as n\ndef f(x):\n    return n(x)\n",
        FROM_EXCEPTIONS + "{0}\nn = {0}\n",
        "import functools\n" + FROM_EXCEPTIONS + "{0}\nn = functools.partial({0}, 'x')\n",
        AS_MODULE + "n = e.{0}\n",
        AS_MODULE + "e.{0} = lambda x: x\n",
        AS_MODULE + "n = getattr(e, '{0}')\n",
        AS_MODULE + "n = vars(e)['{0}']\n",
    ],
    ids=["alias", "binding", "partial", "attribute-value", "rebound", "getattr", "vars"],
)
def test_a_planted_mention_is_reported(tmp_path: Path, name: str, template: str) -> None:
    assert _evasions(_planted(tmp_path, template.format(name))) != []


@pytest.mark.parametrize(("name", "home"), OWNED)
def test_an_owned_name_is_reported_outside_its_module_and_not_inside(
    tmp_path: Path, name: str, home: str
) -> None:
    text = f"def f(x):\n    return {name}(x)\n"

    assert _evasions(_planted(tmp_path / "out", text)) != []
    assert _evasions(_planted(tmp_path / "in", text, module=home)) == []


@pytest.mark.parametrize("name", [name for name, _ in OWNED])
def test_importing_an_owned_name_is_reported(tmp_path: Path, name: str) -> None:
    assert _evasions(_planted(tmp_path, FROM_REGISTRY + f"{name}\n")) != []


@pytest.mark.parametrize("value", MARK_VALUES)
@pytest.mark.parametrize(
    "template",
    [
        "def f(x):\n    x.{0} = 1\n",
        "def f(x):\n    setattr(x, '{0}', 1)\n",
        "class X(Exception):\n    {0} = 1\n",
        "class X(Exception):\n    {0}: int = 1\n",
        "def f(x):\n    x.__dict__.update({0}=1)\n",
        "class X(Exception):\n    def {0}(self):\n        pass\n",
        "class X(Exception):\n    class {0}:\n        pass\n",
        "class X(Exception):\n    from os import sep as {0}\n",
    ],
    ids=["attribute", "string", "class", "annotated", "keyword", "method", "nested", "import"],
)
def test_a_planted_mark_attribute_is_reported(tmp_path: Path, value: str, template: str) -> None:
    assert _evasions(_planted(tmp_path, template.format(value))) != []


@pytest.mark.parametrize("constant", MARK_CONSTANTS)
@pytest.mark.parametrize(
    "template",
    [
        "def f(x):\n    setattr(x, {0}, 1)\n",
        "def f(x):\n    x.__setattr__({0}, 1)\n",
        "def f(x):\n    object.__setattr__(x, {0}, 1)\n",
        "def f(x):\n    k = {0}\n    setattr(x, k, 1)\n",
        "def f(x):\n    x.__dict__[{0}] = 1\n",
    ],
    ids=["setattr", "method", "object", "bound", "dict"],
)
def test_a_second_use_in_a_marks_own_module_is_counted(
    tmp_path: Path, constant: str, template: str
) -> None:
    home = PRIVATE_NAMES[constant]
    assert home is not None
    sources = _planted(tmp_path, template.format(constant), module=home)

    assert _mark_uses(sources) == {(home, "f", constant): 1}


@pytest.mark.parametrize(("value", "home"), MARK_VALUES.items())
def test_a_second_spelling_in_a_marks_own_module_is_counted(
    tmp_path: Path, value: str, home: str
) -> None:
    text = f"def f(x):\n    x.__dict__['{value}'] = 1\n"

    assert _mark_uses(_planted(tmp_path, text, module=home)) == {(home, "f", value): 1}


@pytest.mark.parametrize(
    "template",
    [
        "n = {0}\n",
        "import functools\nn = functools.partial({0}, 'x')\n",
        "def f(x):\n    return {0}\n",
        "from dataknobs_common.other import {0} as n\n",
    ],
    ids=["binding", "partial", "returned", "alias"],
)
def test_the_private_core_is_only_called_even_in_its_own_module(
    tmp_path: Path, template: str
) -> None:
    sources = _planted(tmp_path, template.format("_naming_refusals"), module=EXCEPTIONS)

    assert _evasions(sources) != []


@pytest.mark.parametrize("name", [name for name, _ in OWNED])
@pytest.mark.parametrize(
    "template",
    ["n = {0}\n", "import functools\nn = functools.partial({0}, 'x')\n"],
    ids=["binding", "partial"],
)
def test_an_owned_names_value_use_is_reported_without_an_import(
    tmp_path: Path, name: str, template: str
) -> None:
    """With no import to report first, the mention itself is what is caught."""
    assert _evasions(_planted(tmp_path, template.format(name))) != []
