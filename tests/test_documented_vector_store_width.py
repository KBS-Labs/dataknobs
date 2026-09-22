# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A documented vector store declares its width under the key the store reads.

``VectorStoreConfig`` spells the width ``dimensions``. Nine documented
``vector_store:`` blocks spelled it ``dimension``, and ``from_dict`` projects a
dict onto the declared fields and drops the rest -- so every one of them
declared nothing while appearing to declare 384. Two more named a backend and
no width at all. The number was inert, which is why the misspelling survived:
nothing compared a stored vector to the field, so a store that took its width
from the vectors behaved identically either way.

Two changes on this branch ended that. ``VectorStoreBase._check_batch_width``
compares a write to the declaration, and ``_validate_dimensions`` -- dead from
the commit that introduced it -- now runs at construction, where a backend that
cannot take the width from the vectors refuses an undeclared one outright. All
eleven name ``backend: faiss``, so what was an inert misspelling is now a
``ValueError`` before the first write:

    FaissVectorStore requires a declared vector width: set 'dimensions' in
    the store config.

**The singular is not a typo of the plural, which is why this needs a guard
rather than a careful reader.** Both spellings are real keys one level apart:
``memory: {type: vector, dimension: 768}`` configures the store and
``dimensions`` beside it configures the *embedder*, while under
``vector_store:`` the store's key is ``dimensions`` and there is no singular at
all. A writer moving a sample between the two subsystems carries the wrong one
across, and nothing downstream complains -- the same shape as the sixteen
call sites ``test_documented_backend_config_keys.py`` was written for.

**Scope, stated rather than implied.** This asks two questions of a
``vector_store`` mapping, both about the width:

COVERED
    The width is spelled ``dimensions``. A ``dimension`` key there reaches
    no field of any store config and never did.

    A block naming a backend that cannot defer the question declares a
    width. ``REQUIRES_DECLARED_DIMENSIONS`` is read off the registered
    class rather than listed here, so a fifth backend joins the rule by
    declaring it.

NOT COVERED
    Whether every *other* key in the block is one a store config takes.
    That is ``test_documented_backend_config_keys.py``'s question asked of
    the vector-store registry instead of the database ones, and its scope
    note excludes vector stores because ``memory`` names a plugin in three
    registries and the backend name alone cannot say which. An enclosing
    ``vector_store`` key settles that, so the extension is available; what
    stops it being one line is that it is a different guard's traversal.
    A census run when this was written found the answer is not empty --
    ``persist_directory`` where the field is ``persist_path``, a flat
    ``nlist`` where the class documents ``index_params``, and a documented
    ``backend: pinecone`` that the factory does not register.

    The mirror-image error. A ``memory:`` strategy spelling the store's
    width ``dimensions`` hands it to the embedder instead, and that is a
    legitimate thing to write, so intent is not recoverable from the text.

    Whether the width is one the named model produces. The
    embedding-model table in ``packages/bots/docs/configuration.md`` is the
    document for that, and its numbers are facts about other people's
    models rather than about this tree.
"""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from dataknobs_data.vector.stores import vector_backends
from dataknobs_data.vector.stores.config import VectorStoreConfig
from tests._workspace import code_fences, documentation_files, rel

#: The enclosing key that makes a mapping a vector-store configuration.
#:
#: Position rather than content, for the reason the database guard gives for
#: its own context rule: a store config and the section of one holding
#: ``index_params`` are the same shape, and ``backend: memory`` names a plugin
#: in the vector-store, database and rate-limiter registries alike.
STORE_KEY = "vector_store"

#: The width key the store reads, and the one it does not.
DECLARED = "dimensions"
SINGULAR = "dimension"

#: Fence languages this reads. A ``json`` fence is not read: none in this tree
#: carries a store block, and a reader that claims a language it never
#: exercises is the narrowing this guard exists to catch.
YAML_FENCE = frozenset({"yaml", "yml"})
PYTHON_FENCE = frozenset({"python", "py"})

#: Measured at 30 when this was written -- 11 in YAML fences and 19 in Python
#: -- across 371 documents, once a symlinked site copy is folded onto its
#: source. The floor is a guard on the reader, not a target: a fence-reading
#: change that starts returning nothing reports green on every rule above, and
#: this is the assertion that does not. Set below the measurement so that
#: deleting a sample is an ordinary edit rather than a guard failure.
MINIMUM_BLOCKS_SCANNED = 24


class Block:
    """One documented ``vector_store`` mapping.

    ``syntax`` is which fence it came out of, kept so the corpus test can
    assert both readers are still returning something rather than one of
    them silently going quiet.
    """

    def __init__(self, path: Path, line: int, keys: dict[str, Any], syntax: str):
        self.path, self.line, self.keys, self.syntax = path, line, keys, syntax

    @property
    def backend(self) -> str | None:
        value = self.keys.get("backend")
        return value.strip() if isinstance(value, str) else None

    def __str__(self) -> str:
        return f"{rel(self.path)}:{self.line}  {{{', '.join(sorted(self.keys))}}}"


def _is_reference(keys: dict[str, Any]) -> bool:
    """Whether the mapping is a config-system reference rather than a config.

    ``vector_store: {$resource: vector_store, type: faiss}`` names a resource
    to resolve; the keys are the resolver's vocabulary and none of them is a
    store field. Detected by the ``$`` sigil, which is the config system's own
    marker, rather than by listing the directives it currently has.
    """
    return any(str(key).startswith("$") for key in keys)


def _yaml_blocks(node: Any, line_of: int) -> Iterator[Block]:
    """Walk composed YAML nodes, yielding the mappings under ``vector_store``.

    The line reported is the ``vector_store:`` key's own, not the first key
    inside it: that is the line a reader searches for, and the two differ by
    one in every block-style sample here.
    """
    if isinstance(node, yaml.MappingNode):
        for key, value in node.value:
            named = isinstance(key, yaml.ScalarNode) and key.value == STORE_KEY
            if named and isinstance(value, yaml.MappingNode):
                yield Block(
                    Path(),
                    key.start_mark.line + line_of + 1,
                    {
                        inner.value: getattr(item, "value", None)
                        for inner, item in value.value
                        if isinstance(inner, yaml.ScalarNode)
                    },
                    "yaml",
                )
            yield from _yaml_blocks(value, line_of)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            yield from _yaml_blocks(item, line_of)


def _python_blocks(tree: ast.AST, line_of: int) -> Iterator[Block]:
    """Every ``"vector_store": {...}`` dict literal in a parsed fence."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if not (isinstance(key, ast.Constant) and key.value == STORE_KEY):
                continue
            if not isinstance(value, ast.Dict):
                continue
            keys: dict[str, Any] = {
                inner.value: getattr(item, "value", None)
                for inner, item in zip(value.keys, value.values, strict=True)
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str)
            }
            yield Block(Path(), key.lineno + line_of, keys, "python")


def _document_blocks(path: Path) -> tuple[list[Block], int]:
    """The store blocks one document publishes, and the fences that declined.

    A fence that will not parse is counted rather than ignored: a document
    whose samples all fail to compose reads as a document with no samples,
    which is the failure this file's floor exists to make visible.
    """
    found: list[Block] = []
    declined = 0
    for fence in code_fences(path):
        if fence.lang in YAML_FENCE:
            try:
                composed = list(yaml.compose_all(textwrap.dedent(fence.body)))
            except yaml.YAMLError:
                declined += 1
                continue
            for root in composed:
                if root is None:
                    continue
                found.extend(_yaml_blocks(root, fence.line - 1))
        elif fence.lang in PYTHON_FENCE:
            if f'"{STORE_KEY}"' not in fence.body and f"'{STORE_KEY}'" not in fence.body:
                continue
            source = textwrap.dedent(fence.body)
            # The documented bare-fragment shape -- `"vector_store": {...}` alone
            # in a fence -- is a dict body rather than a module, so it is retried
            # wrapped. Same retry the database guard makes, for the same samples.
            for candidate, offset in ((source, 0), ("_ = {\n" + source + "\n}", -1)):
                try:
                    tree = ast.parse(candidate)
                except SyntaxError:
                    continue
                found.extend(_python_blocks(tree, fence.line - 1 + offset))
                break
            else:
                declined += 1
    for block in found:
        block.path = path
    return [b for b in found if not _is_reference(b.keys)], declined


def _documents() -> list[Path]:
    """Every document, with a site-tree symlink counted once, under its source.

    ``docs/packages/bots/guides/configuration.md`` *is*
    ``packages/bots/docs/configuration.md``; reporting both would send a
    reader to fix one line twice and would inflate the floor below with
    copies rather than coverage. The real file wins the collision, because
    the symlink is not the one to edit.
    """
    seen: dict[Path, Path] = {}
    for path in documentation_files():
        target = path.resolve()
        if target not in seen or not path.is_symlink():
            seen[target] = path
    return sorted(seen.values())


def _scanned() -> tuple[list[Block], int]:
    blocks: list[Block] = []
    declined = 0
    for path in _documents():
        found, refused = _document_blocks(path)
        blocks.extend(found)
        declined += refused
    return blocks, declined


def _requires_declared_width(backend: str) -> bool | None:
    """Whether ``backend``'s store refuses an undeclared width, or ``None``.

    ``None`` for a name the registry does not know -- there is no class to
    ask, and a documented backend nothing registers is a different finding
    from a missing width. The count of those is asserted below rather than
    left implicit.
    """
    store = vector_backends.get_factory(backend) or vector_backends.load_declared_type(backend)
    if store is None:
        return None
    required = getattr(store, "REQUIRES_DECLARED_DIMENSIONS", None)
    return bool(required) if isinstance(required, bool) else None


def test_no_documented_store_spells_the_width_in_the_singular() -> None:
    """``dimension`` under ``vector_store`` configures nothing, and always did."""
    blocks, _ = _scanned()
    offenders = [b for b in blocks if SINGULAR in b.keys]

    assert not offenders, (
        "a documented vector_store block spells the width `dimension`; the "
        "store reads `dimensions` and drops the rest, so the sample declares "
        "no width at all:\n  " + "\n  ".join(str(b) for b in offenders)
    )


def test_a_documented_store_that_cannot_defer_the_width_declares_one() -> None:
    """A sample naming faiss or pgvector and no width no longer constructs."""
    blocks, _ = _scanned()
    offenders = [
        b
        for b in blocks
        if b.backend and _requires_declared_width(b.backend) and DECLARED not in b.keys
    ]

    assert not offenders, (
        "a documented vector_store block names a backend that builds a "
        "fixed-width structure before the first write and declares no "
        f"`{DECLARED}`; the sample raises at construction:\n  "
        + "\n  ".join(str(b) for b in offenders)
    )


def test_the_sweep_reads_a_meaningful_corpus() -> None:
    """The reader still finds documented store blocks, in both syntaxes.

    Both are named rather than only the total, for the reason the database
    guard gives for the same pair: a floor on the sum is satisfied by either
    half alone, and half a reader reports green.
    """
    blocks, _ = _scanned()
    yaml_read = [b for b in blocks if b.syntax == "yaml"]
    python_read = [b for b in blocks if b.syntax == "python"]

    assert len(blocks) >= MINIMUM_BLOCKS_SCANNED, (
        f"only {len(blocks)} documented vector_store blocks were read; the "
        f"floor is {MINIMUM_BLOCKS_SCANNED}. Both rules above pass vacuously "
        "on an empty corpus."
    )
    assert yaml_read and python_read, (
        f"{len(yaml_read)} yaml and {len(python_read)} python blocks: a "
        "syntax that stops being read takes its rules with it"
    )


def test_the_fences_this_cannot_parse_stay_counted() -> None:
    """A declining fence is reported, and the honest count is not zero.

    Several samples elide their middle with ``...`` or show a fragment that
    is not a whole document, and those genuinely do not compose. The floor
    says so out loud; the ceiling is what fails if a reader change starts
    declining fences it used to read, which would empty the sweep while
    every rule above stayed green.

    Measured at 5 when this was written --- two prose-annotated YAML
    fragments, two elided ``RAGKnowledgeBase.from_config`` samples, one
    registry guide.
    """
    _, declined = _scanned()

    assert 1 <= declined <= 12, (
        f"{declined} fences would not parse; the expected band is 1-12 and "
        "a number outside it means the reader changed, not the documents"
    )


def test_the_rules_fire_on_a_document_that_breaks_them(tmp_path: Path) -> None:
    """Positive controls: the same reader, over text written to break each rule.

    Without these the suite above cannot distinguish *nothing is wrong* from
    *nothing was read*, and the corpus floor only answers half of that -- it
    counts blocks, not verdicts.
    """
    document = tmp_path / "sample.md"
    document.write_text(
        "```yaml\n"
        "knowledge_base:\n"
        "  vector_store:\n"
        "    backend: faiss\n"
        "    dimension: 384\n"
        "```\n"
        "\n"
        "```python\n"
        'config = {"vector_store": {"backend": "pgvector"}}\n'
        "```\n"
        "\n"
        "```yaml\n"
        "vector_store:\n"
        "  $resource: shared_store\n"
        "  type: faiss\n"
        "```\n",
        encoding="utf-8",
    )

    blocks, declined = _document_blocks(document)

    assert declined == 0
    assert len(blocks) == 2, "the $resource reference is not a store config"
    assert [b.syntax for b in blocks] == ["yaml", "python"]
    assert [b.line for b in blocks] == [3, 9], "a block names the line it is on"
    assert [SINGULAR in b.keys for b in blocks] == [True, False]
    assert [
        b.backend and _requires_declared_width(b.backend) and DECLARED not in b.keys for b in blocks
    ] == [True, True]


def test_the_width_key_the_rule_names_is_the_one_the_config_declares() -> None:
    """The rule reads the config class rather than restating it.

    ``DECLARED`` and ``SINGULAR`` are literals, and a guard whose vocabulary
    is a literal is one rename away from checking nothing. This is the
    assertion that fails on that rename instead.
    """
    fields = set(VectorStoreConfig.__dataclass_fields__)

    assert DECLARED in fields, f"the store no longer declares `{DECLARED}`"
    assert SINGULAR not in fields, (
        f"`{SINGULAR}` is now a store config field; the rule above would reject a correct sample"
    )


def test_every_registered_backend_answers_the_width_question() -> None:
    """No registered backend leaves ``REQUIRES_DECLARED_DIMENSIONS`` unreadable.

    The rule above skips a backend it cannot ask, so a registry whose classes
    stopped declaring the attribute would silently empty it.
    """
    unanswered = [
        name
        for name in vector_backends.list_canonical_keys()
        if _requires_declared_width(name) is None
    ]

    assert not unanswered, f"backends with no readable width contract: {unanswered}"


#: Backends the documentation names that the factory does not register.
#:
#: An open finding recorded rather than fixed, and recorded here because the
#: width rules skip such a block -- there is no class to ask -- so without a
#: declaration the skip is silent. ``pinecone`` has a "Vector Store Backends"
#: section and a production sample in ``packages/bots/docs/configuration.md``
#: and is in no registry; whether to implement it or withdraw the section is
#: not a question about widths, which is why this branch leaves it open.
#:
#: Declared rather than tolerated: the pair of assertions below fails both on
#: a *new* unregistered backend and on this one once it stops appearing, so
#: the list cannot quietly outlive what it records.
KNOWN_UNREGISTERED = frozenset({"pinecone"})


def test_a_documented_backend_is_one_the_factory_registers() -> None:
    """Named separately because it is a different defect from a missing width."""
    blocks, _ = _scanned()
    unknown = {
        b.backend for b in blocks if b.backend and _requires_declared_width(b.backend) is None
    }

    assert unknown <= KNOWN_UNREGISTERED, (
        "documented vector-store backends the factory does not register: "
        f"{sorted(unknown - KNOWN_UNREGISTERED)}"
    )
    assert unknown == KNOWN_UNREGISTERED, (
        f"{sorted(KNOWN_UNREGISTERED - unknown)} no longer appears in any "
        "documented vector_store block; drop it from KNOWN_UNREGISTERED"
    )
