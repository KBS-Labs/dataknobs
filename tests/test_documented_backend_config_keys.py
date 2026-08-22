"""Reproduce-first guard: a config key shown in the docs must be one a backend takes.

Sibling of ``test_documented_imports.py``, and for the same reason. That guard
exists because an import is mechanically checkable and so has no excuse for
being wrong. A backend config key became mechanically checkable when
``StructuredConfig.accepts()`` landed, and it turned out to be wrong at sixteen
call sites across eight documents.

None of them could fail anything before. ``mkdocs build --strict`` validates
links and nav, the doc-mirror manifest validates that two copies agree, and
``test_documented_imports.py`` reads the names a document loads -- imports,
directive values, paths named in prose -- and never the keys beside them. A
sample whose names all resolve and whose keys are fiction passes all three -- and used to pass at runtime too,
because ``from_dict`` projected the dict onto the declared fields and dropped
whatever was left. ``pool_size=20`` on a pooling page configured nothing.

The population was not one mistake repeated. It was three, and the next ten will
be one of them again:

- **A field that belongs to the sibling backend.** ``hosts`` is a field on the
  *async* Elasticsearch config; the sync one takes ``host``/``port``. Five sync
  samples passed ``hosts``, and all five appeared to work, because the value
  they were silently not applying was the default anyway.
- **A name from another library's vocabulary.** ``pool_size`` / ``max_overflow``
  are SQLAlchemy's spelling; the fields here are ``min_pool_size`` /
  ``max_pool_size``. Likewise ``username``/``password`` where the field is
  ``basic_auth``, and ``connection`` where it is ``connection_string``.
- **A field that never existed.** ``pretty`` and ``backup`` on the file backend,
  documented in an options list, matching nothing in the dataclass.

Scope is stated rather than implied, because a guard that quietly covers less
than it appears to is worse than none -- it also reports green:

COVERED
    **Call form** -- ``<factory>.create(backend=..., ...)``,
    ``<X>.from_backend("b", {...})``, ``database_factory(...)``.

    **Constructor form** -- ``SyncSQLiteDatabase({...})`` with an inline
    dict. The class name settles sync-vs-async outright, so this is the one
    form where a sibling-backend field is provable rather than merely likely.

    **Configuration-block form** -- a mapping in a YAML fence, in a quoted
    dict literal handed to ``Config.load`` or ``from_config``, or in YAML
    written into a Python string literal. All three are *parsed* and walked
    by one traversal, because they are one question asked of two parsers:
    which mapping here is a database config, and what keys does it hold.
    ``DB_CONTEXT`` answers the first from the enclosing keys -- ``memory``
    names a plugin in the vector-store and rate-limiter registries too, so
    the backend name alone cannot -- and a ``${DB_BACKEND:postgres}`` value
    is read through to the default an unconfigured reader installs.

    **A block naming no backend**, which is most of them: ``databases:`` /
    ``primary:`` / ``host: ...``. There is no class to ask, so the union of
    every database config is asked instead, and a key *none* of them accepts
    is wrong whichever backend the block turns out to be. Position settles
    what a mapping is, and it has to, because a container of named configs
    and a config are the same shape: ``DB_CONTAINER`` -- the plural member
    of ``DB_CONTEXT`` -- holds entries, every other member holds a config,
    and a mapping inside a config is a section of it whose own key is the
    thing judged. A block that resembles no config is skipped and counted
    rather than reported, because the context rule's trailing-``_db``
    alternative matches a query filter as readily as a store.

NOT COVERED
    A config bound to a variable and passed by name. Which binding a name
    refers to is not decidable by reading a prose document: one sample hands
    ``AsyncElasticsearchDatabase`` a ``config`` the surrounding text never
    defines, and the nearest earlier binding of that name belongs to a
    different example. Reporting that as a defect would make the guard cry
    wolf, and a guard that cries wolf gets deleted. Those sites are checked
    by hand, and the price of that was on display when this section was last
    edited: a sample building ``configs[env]`` and splatting it into
    ``factory.create`` carried a ``pool_size`` that raises, and was found by
    reading rather than by failing.

    A **documented key list in prose**. An options list under a heading is
    the third of the three defect classes above, and it is the one no
    scanner here reads -- these sweeps read code. A bot configuration guide
    listed ``pool_size``, ``max_overflow`` and ``pool_timeout`` as the
    connection options for its Postgres storage; none of the three exists.

    Whether an accepted key does anything. ``accepts()`` answers what the
    config class takes, and one class backs both Postgres backends, so
    ``min_pool_size`` / ``max_pool_size`` / ``command_timeout`` are accepted
    for a *sync* sample and documented by their own class as async-only --
    psycopg2 has no pool to bound. The flavor machinery cannot help here:
    both registries resolve Postgres to the same class, so the sibling-field
    check that catches ``hosts`` on sync Elasticsearch has nothing to
    compare. Two samples repaired alongside this guard were sync and now say
    so in a comment, found by reading the config class rather than by
    failing anything.

    Which of several backends an unnamed block *is*. A block naming no
    ``backend:`` is read against the union of every config class, so a key
    none of them accepts is reported and a key one of them accepts is not.
    That asymmetry is deliberate and it is only half the question: whether
    ``max_pool_size`` on an unnamed block is the *right* class's key needs
    the backend, and the block does not say.

The accepted-key set is read from the config class through ``accepts()``, and
the tolerated routing keys from the runtime's own ``_ROUTING_KEYS``, so this
file cannot drift from the code it checks. Coverage is asserted rather than
assumed: a backend whose driver is absent is still read against its declared
config class, the corpus floor counts the sites actually checked, each front
end is counted separately so one going silent cannot hide behind the other,
and every way a block leaves unchecked -- skipped for its context, in a fence
that would not parse, naming a backend that resolves to nothing, resembling
no config at all -- is a number with a test watching it, bounded above *and*
below so that a judgement which stopped judging cannot read as a clean
corpus.
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml

from dataknobs_common.registry import PluginRegistry
from dataknobs_common.structured_config import _ROUTING_KEYS, StructuredConfig
from dataknobs_data.backends import _register_sync_backends, async_backends, sync_backends
from tests._workspace import code_fences, documentation_files, rel

#: Keys the object-graph layer owns. They travel with a config on purpose and
#: the runtime check tolerates them, so the guard must too.
#:
#: Taken from the runtime's own set rather than retyped, so the two cannot
#: disagree about what is tolerated. ``config`` is added because it is
#: ``from_backend``'s *parameter* name -- a sample writing
#: ``from_backend("sqlite", config={...})`` names the parameter, not a config
#: key -- which is a fact about this scanner's regex, not about the runtime.
ROUTING = _ROUTING_KEYS | {"config"}

CALLSITE = re.compile(
    r"(?P<recv>[\w.]+)\s*\.\s*(?P<meth>create|from_backend)\s*\("
    r"|(?P<bare>\basync_database_factory|\bdatabase_factory)\s*\("
)
#: An explicit signal that the sample drives the async factory, or the sync one.
ASYNC_SIGNAL = re.compile(r"Async\w*(Factory|Database)|await\s|async_database_factory")
SYNC_SIGNAL = re.compile(r"(?<!Async)(?<!async_)\b(DatabaseFactory|SyncDatabase)\b")


def _balanced(text: str, open_at: int) -> int:
    """Index of the delimiter closing the one at ``open_at``, or -1."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] in "({[":
            depth += 1
        elif text[i] in ")}]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _top_level_pairs(body: str) -> dict[str, str]:
    """Map each depth-0 ``k=v`` / ``"k": v`` to its literal value snippet.

    Depth matters twice over: a key nested inside a value belongs to that
    value's own schema, and skipping past a value must not lose the brackets
    it contained -- ``hosts=[f("a", "b")],`` otherwise leaves the counter
    negative and every later key reads as nested.
    """
    pairs: dict[str, str] = {}
    depth, i, line_start = 0, 0, True
    while i < len(body):
        ch = body[i]
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif depth == 0:
            m = re.match(
                r"""\s*(?:['"](?P<q>[a-z_][a-z0-9_]*)['"]\s*:"""
                r"""|(?P<k>[a-z_][a-z0-9_]*)\s*=(?!=))\s*(?P<v>[^,\n]*)""",
                body[i:],
            )
            if m and (line_start or body[i - 1] in ",\n"):
                pairs[m.group("q") or m.group("k")] = m.group("v").strip()
                skipped = body[i : i + m.end()]
                depth += sum(c in "({[" for c in skipped) - sum(c in ")}]" for c in skipped)
                i += m.end() - 1
        line_start = ch in ",\n"
        i += 1
    return pairs


def _registries(flavor: str) -> list[PluginRegistry[Any]]:
    """The registries a sample of this flavor may be read against.

    ``"unknown"`` yields both: where the sample gives no signal which factory
    it drives, a key accepted by either is not evidence of a defect. Those
    sites are counted rather than silently waved through.
    """
    by_flavor: dict[str, list[PluginRegistry[Any]]] = {
        "sync": [sync_backends],
        "async": [async_backends],
        "unknown": [sync_backends, async_backends],
    }
    return by_flavor[flavor]


def _config_class(registry: PluginRegistry[Any], backend: str) -> type[StructuredConfig] | None:
    """``backend``'s config class in ``registry``, driver installed or not.

    ``get_factory`` returns ``None`` -- it does not raise -- for a backend
    declared unavailable, and that is deliberate on its part: a caller
    reaching for a factory means to create. Reading ``CONFIG_CLS`` off that
    ``None`` yields nothing, and a site with nothing to check against
    reports no rejected keys, so an absent ``psycopg2`` used to remove every
    Postgres sample from the sweep while the guard still reported green.

    ``load_declared_type`` exists for exactly this: reading a schema off a
    plugin that cannot be built here. The config classes are plain
    dataclasses in a module that does not import its driver at top level, so
    coverage survives the driver's absence instead of depending on it.

    The ``issubclass`` narrowing does two jobs at once: it is what makes the
    return type true rather than asserted, and it is what excludes a backend
    registered as a bare callable -- no ``CONFIG_CLS``, so no keys to accept
    and nothing for this guard to check.
    """
    cls = getattr(registry.get_factory(backend), "CONFIG_CLS", None)
    if cls is None and registry.is_known(backend):
        cls = getattr(registry.load_declared_type(backend), "CONFIG_CLS", None)
    return cls if isinstance(cls, type) and issubclass(cls, StructuredConfig) else None


def _config_classes(backend: str, flavor: str) -> list[type[StructuredConfig]]:
    """The config classes a sample's keys may legitimately belong to."""
    found = [_config_class(registry, backend) for registry in _registries(flavor)]
    return [cls for cls in found if cls is not None]


def _flavor(text: str, at: int, recv: str) -> str:
    """Which factory the sample drives, from an explicit signal or not at all."""
    binding = ""
    base = recv.partition(".")[0]
    if base:
        for m in re.finditer(rf"\b{re.escape(base)}\s*=\s*([^\n]+)", text[:at]):
            binding = m.group(1)
    probe = f"{recv} {binding} {text[max(0, at - 80) : at]}"
    if ASYNC_SIGNAL.search(probe):
        return "async"
    if SYNC_SIGNAL.search(probe):
        return "sync"
    return "unknown"


class Site:
    """One documented database-factory call."""

    def __init__(self, path: Path, line: int, backend: str, flavor: str, keys: set[str]):
        self.path, self.line, self.backend, self.flavor, self.keys = (
            path,
            line,
            backend,
            flavor,
            keys,
        )

    @property
    def in_scope(self) -> bool:
        """Whether this is a name the database registries know at all.

        ``pgvector`` and ``redis`` reach a *vector store* factory whose keys
        answer to a different config, and a doc registering an illustrative
        ``custom`` backend names one that exists nowhere. None of the three
        is a database backend, so none is this guard's to check -- which is
        a different thing from one it should check and cannot, and the two
        used to be indistinguishable in the output.
        """
        return any(registry.is_known(self.backend) for registry in _registries(self.flavor))

    @property
    def checked(self) -> bool:
        """Whether a config class was actually found to read the keys against."""
        return bool(_config_classes(self.backend, self.flavor))

    @property
    def rejected(self) -> list[str]:
        """Keys no candidate config class claims."""
        classes = _config_classes(self.backend, self.flavor)
        if not classes:
            return []
        return sorted(
            key for key in self.keys - ROUTING if not any(cls.accepts(key) for cls in classes)
        )

    def __str__(self) -> str:
        names = ", ".join(c.__name__ for c in _config_classes(self.backend, self.flavor))
        return f"{rel(self.path)}:{self.line} ({self.backend}) {names} rejects {self.rejected}"


def _sites() -> tuple[list[Site], int]:
    """Every documented call-form site, and how many the guard read leniently.

    A sample that documents the error it raises is not claiming to work -- the
    wrongness is the content, exactly as in the import guard's illustrative
    fences -- so a call followed by a ``ValueError`` is skipped.
    """
    sites: list[Site] = []
    unknown = 0
    for path in documentation_files():
        text = path.read_text(encoding="utf-8")
        for m in CALLSITE.finditer(text):
            open_at = text.index("(", m.start())
            close = _balanced(text, open_at)
            if close < 0:
                continue
            if "ValueError" in text[close : close + 220]:
                continue
            body = text[open_at + 1 : close]
            pairs = _top_level_pairs(body)
            positional = re.match(r"""\s*['"](?P<b>\w+)['"]\s*,?""", body)
            if m.group("meth") == "from_backend" and positional:
                backend = positional.group("b")
                rest = body[positional.end() :]
                if rest.lstrip().startswith("{"):
                    inner = body.index("{", positional.end())
                    pairs = _top_level_pairs(body[inner + 1 : _balanced(body, inner)])
                elif "=" not in rest and rest.strip():
                    continue  # config passed by variable -- no keys to read here
            else:
                raw = pairs.get("backend")
                value = re.match(r"""^['"]?(?P<b>\w+)['"]?""", raw) if raw else None
                if value is None:
                    continue
                backend = value.group("b")
            flavor = _flavor(text, m.start(), m.group("recv") or m.group("bare") or "")
            if flavor == "unknown":
                unknown += 1
            sites.append(Site(path, text[: m.start()].count("\n") + 1, backend, flavor, set(pairs)))
    return sites, unknown


@cache
def _scanned() -> tuple[list[Site], int]:
    """Scan once; every test below reads the same result."""
    return _sites()


# --- The configuration-block forms ---------------------------------------
#
# A ``backend`` named inside a configuration block reaches the same factory as
# a call does, and the keys beside it are the same keys. Two syntaxes carry
# such a block: a YAML fence, and a quoted dict literal handed to
# ``Config.load`` or ``from_config`` rather than to a factory or a
# constructor. They are read through one traversal because they are one
# question -- which mapping in this document is a database config, and what
# keys does it hold -- asked of two parsers.
#
# Both are parsed rather than pattern-matched, and that is the design. The
# difficulty was never finding a ``backend``; it is knowing whose block it is.
# ``memory`` names a plugin in the database registry, the vector-store
# registry and the rate-limiter registry alike, so only the *enclosing* keys
# can say. A parser answers that by construction. The line-walking reader this
# replaced had to rebuild nesting from indentation, was wrong about it twice
# in its first month -- a list item's leading key measured shallower than the
# block it opened, and ``- backend:`` on a dash line matched nothing at all --
# and could not express the question for Python at all. 108 dict-literal
# blocks went unread on that account, six of them defective, while this file
# reported green.

#: An ancestry segment that says the block below it holds a *database*
#: config. ``databases:`` is the environment-resources spelling and the
#: plural mapping in a config file, ``database:`` its singular, and a
#: ``<name>_db:`` block is the shape the reference docs use for a standalone
#: sample. ``conversation_storage:`` is a bot's, and it is passed to the
#: factory verbatim.
#:
#: The rule is stated as an allowlist rather than a list of subsystems to
#: skip because the failure directions are not symmetric: an unrecognised
#: context leaves a block unchecked, which
#: ``test_the_share_out_of_scope_stays_visible`` counts, whereas a denylist
#: that missed a new subsystem would report its keys as defects -- and a
#: guard that cries wolf is uninstalled by the second person to hit it.
DB_CONTEXT = re.compile(r"^(databases?|conversation_storage|\w*_db)$")

#: The plural member of ``DB_CONTEXT``, which holds *entries* rather than a
#: config. Its keys are names a document invented -- ``primary``, ``cache``,
#: ``archive`` -- so reading it as a config reports each of them as a
#: rejected key. Measured: a version that did produced 49 findings on this
#: corpus, 34 of them entry names. Every other member is singular and its
#: value is the config itself, which is the whole distinction, and it is
#: already spelled in ``DB_CONTEXT``'s own ``databases?``.
DB_CONTAINER = re.compile(r"^databases$")

#: The fence languages each parser reads.
YAML_FENCE = frozenset({"yaml", "yml"})
PYTHON_FENCE = frozenset({"python", "py"})

#: A whole value that is an environment substitution, with the default it
#: falls back to. Unconfigured, a reader installs that default, so it is the
#: name to read the block's keys against. A substitution with no default
#: names nothing checkable and is counted instead of skipped -- being
#: skipped is how the whole substitution form stayed invisible: the line
#: pattern this replaced required a literal name, so such a block was
#: neither scanned nor counted, and one carrying ``pool_size`` sat in a
#: document that spells the key correctly forty lines above.
SUBSTITUTION = re.compile(r"^\$\{[\w.]+(?::(?P<default>[^{}]*))?\}$")

#: A quoted ``backend`` key. Only Python fences holding one are parsed, which
#: keeps ``declined`` meaning "a fence claiming to hold a database block that
#: could not be read" rather than "a fence of prose". Most Python fences in a
#: prose document are fragments and do not parse; counting those would bury
#: the two that matter.
QUOTED_BACKEND = re.compile(r"""['"]backend['"]\s*:""")


class Shape:
    """How to read one parser's nodes as mappings, sequences and scalars.

    Three questions, because they are the only three the traversal asks.
    ``items`` answers ``None`` for a node that is not a mapping, which is
    what separates "a config block with no keys" from "not a config block".
    """

    def __init__(
        self,
        items: Callable[[Any], list[tuple[str, Any, int]] | None],
        children: Callable[[Any], list[Any]],
        scalar: Callable[[Any], str | None],
    ):
        self.items, self.children, self.scalar = items, children, scalar


def _yaml_items(node: Any) -> list[tuple[str, Any, int]] | None:
    if not isinstance(node, yaml.MappingNode):
        return None
    return [
        (str(key.value), value, key.start_mark.line + 1)
        for key, value in node.value
        if isinstance(key, yaml.ScalarNode)
    ]


def _yaml_children(node: Any) -> list[Any]:
    return list(node.value) if isinstance(node, yaml.SequenceNode) else []


def _yaml_scalar(node: Any) -> str | None:
    return str(node.value) if isinstance(node, yaml.ScalarNode) else None


def _python_items(node: Any) -> list[tuple[str, Any, int]] | None:
    if not isinstance(node, ast.Dict):
        return None
    return [
        (key.value, value, key.lineno)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]


def _python_children(node: Any) -> list[Any]:
    return list(ast.iter_child_nodes(node))


def _python_scalar(node: Any) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


YAML_SHAPE = Shape(_yaml_items, _yaml_children, _yaml_scalar)
PYTHON_SHAPE = Shape(_python_items, _python_children, _python_scalar)


#: Where a mapping sits relative to the nearest ``DB_CONTEXT`` key.
#: ``CONTAINER`` holds entries, ``PAYLOAD`` is a config, ``None`` is neither.
CONTAINER, PAYLOAD = "container", "payload"


class Block:
    """One mapping in a document, and what its ancestry says it is."""

    def __init__(
        self,
        line: int,
        backend: str | None,
        named: bool,
        keys: set[str],
        db: bool,
        payload: bool = False,
        section: bool = False,
    ):
        self.line, self.backend, self.named, self.keys, self.db = line, backend, named, keys, db
        #: Sits where a config lives, so its keys are a config's keys.
        self.payload = payload
        #: Holds a nested mapping -- evidence it is a config even when not
        #: one of its own keys is a spelling any config class accepts.
        self.section = section
        #: Keys whose value is a mapping *and* which open a database region
        #: of their own. They name a nested config rather than a field of
        #: this one, so they are not this config's to accept -- while
        #: ``database: myapp``, a scalar, stays the ordinary field it is.
        self.regions: set[str] = set()


def _blocks(
    node: Any,
    shape: Shape,
    chain: tuple[str, ...] = (),
    line: int = 0,
    level: str | None = None,
) -> list[Block]:
    """Every mapping reachable from ``node``, with the ancestry it sits under.

    Descent is by *key*, so the chain carried down is the real ancestry
    rather than whichever key happens to appear above the block in the text.
    That distinction is why this is a parser: a regex walking backwards
    through characters finds the nearest preceding ``"key":`` and cannot tell
    a preceding sibling from an enclosing parent. A prototype that did read
    two bot vector-memory blocks as database configs -- and an unsound sweep
    here is worse than none, for the crying-wolf reason above.

    ``level`` carries the second question, which only a block naming no
    backend has to ask: is this mapping a config, the container above one, or
    a section inside one? The chain cannot answer it -- every one of the
    three sits under the same ``databases:`` -- so the answer is threaded
    down from the key that opened the region. A mapping inside a config is
    left at ``None``: it belongs to that config's schema, so the question is
    whether the *section* is accepted -- asked once, on the parent -- rather
    than one question per interior key. The exception is a subsection whose
    own key opens a region (``conversation_storage:`` inside a config), which
    is a config in its own right and is read as one.
    """
    found: list[Block] = []
    pairs = shape.items(node)
    if pairs is not None:
        backend = next((shape.scalar(value) for key, value, _ in pairs if key == "backend"), None)
        named = any(key == "backend" for key, _, _ in pairs)
        found.append(
            Block(
                line,
                backend,
                named,
                {key for key, _, _ in pairs},
                any(DB_CONTEXT.match(key) for key in chain),
                payload=level == PAYLOAD,
                section=any(shape.items(value) is not None for _, value, _ in pairs),
            )
        )
        found[-1].regions = {
            key
            for key, value, _ in pairs
            if shape.items(value) is not None and DB_CONTEXT.match(key)
        }
        for key, value, at in pairs:
            below: str | None
            if DB_CONTAINER.match(key):
                below = CONTAINER
            elif DB_CONTEXT.match(key):
                below = PAYLOAD
            else:
                below = PAYLOAD if level == CONTAINER else None
            found.extend(_blocks(value, shape, (key, *chain), at, below))
        return found
    # A sequence under a container holds the entries themselves:
    # ``databases:`` / ``- name: primary`` / ``host: ...`` is one config.
    below = PAYLOAD if level == CONTAINER else level
    for child in shape.children(node):
        found.extend(_blocks(child, shape, chain, line, below))
    return found


def _backend_name(raw: str | None) -> str | None:
    """The backend a block names, read through an environment substitution.

    ``${DB_BACKEND:postgres}`` is what an unconfigured reader installs, so
    the default is the name to check the keys against.
    ``${features.search_backend}`` names nothing checkable and answers
    ``None``, which the caller counts rather than drops.
    """
    if raw is None:
        return None
    value = raw.strip().strip("'\"")
    substitution = SUBSTITUTION.match(value)
    if substitution is not None:
        default = (substitution.group("default") or "").strip()
        return default if re.fullmatch(r"[\w-]+", default) else None
    return value or None


class UnnamedSite:
    """A documented config block that names no backend.

    Most of this file resolves a block's ``backend:`` to a config class and
    asks that class. A block naming none has no class to ask, which left the
    whole population unchecked -- and a bare ``database:`` block teaching
    ``pool_size`` was corrected by hand for exactly that reason, which is the
    mechanism this file exists to make unnecessary.

    The backend does not have to be known to answer *half* the question. A
    key **no** config class accepts is wrong whichever backend the block
    turns out to be, so that half is decidable outright; a key some class
    accepts is left alone, because deciding whether it is the right class is
    the half that needs the backend. That asymmetry is the whole design, and
    it is why this reports a subset rather than guessing.
    """

    def __init__(self, path: Path, line: int, keys: set[str], regions: set[str] | None = None):
        self.path, self.line, self.keys = path, line, keys - (regions or set())
        self.backend: str | None = None

    @property
    def in_scope(self) -> bool:
        """Recognisable as a database config, or it would not be here."""
        return True

    @property
    def checked(self) -> bool:
        """Every config class was consulted, so the answer is not partial."""
        return True

    @property
    def rejected(self) -> list[str]:
        """Keys not one config class in either registry claims."""
        classes = _backend_classes().values()
        return sorted(
            key
            for key in self.keys - ROUTING
            # ``$resource`` / ``$required`` are the config layer's reference
            # markers, resolved and removed before a backend config is built.
            if not key.startswith("$") and not any(cls.accepts(key) for cls in classes)
        )

    def __str__(self) -> str:
        return f"{rel(self.path)}:{self.line} (no backend named) no config class accepts {
            self.rejected
        }"


#: What a config-block sweep returns. Two classes rather than one because
#: they answer ``rejected`` from different evidence -- a named block from its
#: own config class, an unnamed one from the union of all of them -- and
#: collapsing that into a nullable ``backend`` would hide which question was
#: asked of any given finding.
AnySite = Site | UnnamedSite


def _recognisable(block: Block) -> bool:
    r"""Whether a block naming no backend resembles a database config at all.

    ``DB_CONTEXT``'s ``\w*_db`` matches a query filter as readily as a store:
    ``default_filters:`` / ``case_db:`` / ``status: published`` configures
    nothing. Neither does a custom-``storage_class`` block, whose keys are
    the consumer's class's business rather than any backend's.

    Two kinds of evidence answer it, and either suffices. A key some config
    class accepts says the block speaks the vocabulary. A nested section says
    so too, and is needed on its own account: a config whose every key is a
    subsection -- ``postgres:`` / ``pool:`` / ``min_size: 5`` -- has no
    accepted spelling of its own, and it is the one carrying the finding.
    """
    classes = _backend_classes().values()
    return block.section or any(cls.accepts(key) for key in block.keys for cls in classes)


class Tally:
    """Every way a block left the sweep unchecked, kept as a number.

    A form the sweep cannot read is the defect this file exists to catch, and
    for a year it was this file's own: the dict-literal form was neither
    checked nor counted, so no assertion could see it missing. Each field
    here is one way a block leaves without being checked, and each has a test
    below watching its share.
    """

    def __init__(self) -> None:
        self.out_of_scope = 0
        self.declined = 0
        self.undecided = 0
        self.unrecognised = 0


def _embedded_yaml(tree: ast.Module, offset: int) -> list[tuple[Any, Shape, int]]:
    """YAML written into a Python string literal, read as the YAML it is.

    A document that shows a config file by assigning it to a name -- a
    triple-quoted ``yaml_config``, then loading it -- is writing YAML, and
    the keys in it are as real as any in a ``yaml`` fence. The line-scanning
    reader this replaced saw those blocks for the wrong reason (it read the
    document's text and never asked what a fence was), so a parser that
    dropped them would be trading one silent narrowing for another. Four
    documented database blocks live in exactly one such string.

    Only a literal composing to a *mapping* is taken. Every string is valid
    YAML as a scalar, so the mapping test is what keeps prose out: of the 79
    literals here that compose to a mapping, one carries a database block,
    and ``DB_CONTEXT`` is what decides even that.
    """
    found: list[tuple[Any, Shape, int]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if "\n" not in node.value or ":" not in node.value:
            continue
        try:
            composed = yaml.compose(node.value)
        except yaml.YAMLError:
            continue
        if isinstance(composed, yaml.MappingNode):
            found.append((composed, YAML_SHAPE, node.lineno + offset - 1))
    return found


def _roots(path: Path) -> tuple[list[tuple[Any, Shape, int]], int]:
    """Every parseable configuration root in ``path``, and how many declined.

    A YAML fence is parsed whole. A Python fence is parsed only when it names
    a quoted ``backend``, and is retried wrapped in a dict when it will not
    parse alone: the documented bare-fragment shape --
    ``"conversation_storage": {...}`` on its own in a fence -- is not a
    Python module but is a Python dict body. 14 fences in this tree need
    that retry and 125 do not.
    """
    roots: list[tuple[Any, Shape, int]] = []
    declined = 0
    for fence in code_fences(path):
        if fence.lang in YAML_FENCE:
            try:
                composed = [node for node in yaml.compose_all(fence.body) if node is not None]
            except yaml.YAMLError:
                declined += 1
                continue
            roots.extend((node, YAML_SHAPE, fence.line - 1) for node in composed)
        elif fence.lang in PYTHON_FENCE:
            source = textwrap.dedent(fence.body)
            for candidate, offset in ((source, 0), ("_ = {\n" + source + "\n}", -1)):
                try:
                    tree = ast.parse(candidate)
                except SyntaxError:
                    continue
                at = fence.line - 1 + offset
                if QUOTED_BACKEND.search(fence.body):
                    roots.append((tree, PYTHON_SHAPE, at))
                roots.extend(_embedded_yaml(tree, at))
                break
            else:
                declined += QUOTED_BACKEND.search(fence.body) is not None
    return roots, declined


def _config_blocks(path: Path) -> tuple[list[AnySite], Tally]:
    """The database-config blocks in one document, and what it left unread."""
    sites: list[AnySite] = []
    tally = Tally()
    roots, tally.declined = _roots(path)
    for root, shape, offset in roots:
        for block in _blocks(root, shape):
            backend = _backend_name(block.backend)
            if not block.db:
                # A name no database registry knows needs no watching --
                # ``backend: faiss`` is a vector store, and out of scope for
                # a reason that will not change. A *real* database backend
                # skipped for its context is the judgement worth counting.
                known = backend is not None and any(
                    reg.is_known(backend) for reg in (sync_backends, async_backends)
                )
                tally.out_of_scope += known
                continue
            if backend is None:
                if block.named:
                    # Named a backend that resolves to nothing checkable --
                    # ``${features.search_backend}`` supplies no default.
                    tally.undecided += 1
                elif not block.payload:
                    pass  # a container, or a section inside a config
                elif _recognisable(block):
                    sites.append(UnnamedSite(path, block.line + offset, block.keys, block.regions))
                else:
                    # Sits where a config would and resembles none. Counted,
                    # because a population that is only skipped is one no
                    # assertion can report as missing -- which is how this
                    # file's own dict-literal blind spot survived a year.
                    tally.unrecognised += 1
                continue
            sites.append(Site(path, block.line + offset, backend, "unknown", block.keys))
    return sites, tally


def _config_block_sites() -> tuple[list[AnySite], Tally]:
    """Every documented database-config block, and what the sweep left unread.

    ``"unknown"`` is the flavor, stated rather than inferred. A block handed
    to ``Config.load`` or written into a config file says nothing about which
    factory will build it, and measuring that says so: 107 of 108 dict blocks
    give no signal either way, so a lenient-share counter here would count
    the whole population and mean nothing by it.
    """
    sites: list[AnySite] = []
    total = Tally()
    for path in documentation_files():
        found, tally = _config_blocks(path)
        sites.extend(found)
        total.out_of_scope += tally.out_of_scope
        total.declined += tally.declined
        total.undecided += tally.undecided
        total.unrecognised += tally.unrecognised
    return sites, total


@cache
def _scanned_blocks() -> tuple[list[AnySite], Tally]:
    """Scan once; every block test below reads the same result."""
    return _config_block_sites()


# --- The constructor form ------------------------------------------------
#
# ``SyncSQLiteDatabase({...})`` names its backend in the class rather than in
# a ``backend=`` key, so the call-form scanner above does not see it -- and
# the class name settles the sync/async question outright, which is the one
# thing the other two forms have to guess at.


#: Documented backend class name -> its config class, read from the
#: registries so a renamed or added backend needs no edit here. The declared
#: type is used when a driver is absent, for the reason ``_config_class``
#: gives.
@cache
def _backend_classes() -> dict[str, type[StructuredConfig]]:
    found: dict[str, type[StructuredConfig]] = {}
    for registry in (sync_backends, async_backends):
        for name in registry.list_known_keys():
            backend = registry.get_factory(name) or registry.load_declared_type(name)
            config_cls = getattr(backend, "CONFIG_CLS", None)
            if backend is not None and isinstance(config_cls, type):
                found[backend.__name__] = config_cls
    return found


CONSTRUCTOR = re.compile(r"\b(?P<cls>(?:Sync|Async)\w*Database)\s*\(\s*\{")


def _constructor_sites() -> list[tuple[Path, int, type[StructuredConfig], set[str]]]:
    """Every ``<Backend>Database({...})`` with an inline dict literal.

    Only an inline literal. A config reached through a variable is
    deliberately not followed: which binding a name refers to is not
    decidable by reading a prose document, and guessing produces false
    reports. One sample here hands ``AsyncElasticsearchDatabase`` a
    ``config`` that the surrounding text never defines -- the nearest
    earlier binding of that name belongs to a different example entirely.
    A guard that called that a defect would be uninstalled by the second
    person to hit it.
    """
    sites: list[tuple[Path, int, type[StructuredConfig], set[str]]] = []
    for path in documentation_files():
        text = path.read_text(encoding="utf-8")
        for m in CONSTRUCTOR.finditer(text):
            config_cls = _backend_classes().get(m.group("cls"))
            if config_cls is None:
                continue
            open_at = text.index("{", m.start())
            close = _balanced(text, open_at)
            if close < 0:
                continue
            keys = set(_top_level_pairs(text[open_at + 1 : close]))
            sites.append((path, text[: m.start()].count("\n") + 1, config_cls, keys))
    return sites


@cache
def _scanned_constructors() -> list[tuple[Path, int, type[StructuredConfig], set[str]]]:
    """Scan once; every constructor test below reads the same result."""
    return _constructor_sites()


def test_every_documented_constructor_key_is_one_the_backend_accepts() -> None:
    """The form where sync and async are named outright, so nothing is lenient.

    This is where the sibling-backend defect is provable rather than merely
    likely: ``SyncSQLiteDatabase({... "pool_size": ...})`` is checked against
    the sync config alone, and ``pool_size`` belongs to the async one.
    """
    offenders = []
    for path, line, config_cls, keys in _scanned_constructors():
        rejected = sorted(k for k in keys - ROUTING if not config_cls.accepts(k))
        if rejected:
            offenders.append(f"{rel(path)}:{line} {config_cls.__name__} rejects {rejected}")
    assert not offenders, "documented constructor keys no backend field claims:\n" + "\n".join(
        offenders
    )


def test_the_constructor_sweep_reads_a_meaningful_corpus() -> None:
    """As above: a scanner that stops matching must fail rather than pass."""
    sites = _scanned_constructors()
    assert len(sites) >= 10, f"only {len(sites)} documented backend constructors found"


def test_every_documented_block_key_is_one_the_backend_accepts() -> None:
    """The same rule as the call form, on the shape most consumers copy.

    A bot's ``conversation_storage`` block reaches
    ``AsyncDatabaseFactory.create(**config)`` with every key it carries, so
    a ``pool_size`` there is not decoration -- it is a key the config class
    now rejects, in a document telling the reader to write it. Six of the
    seven this found the first time it ran were dict literals, invisible to
    every earlier form of this sweep; one sat under a heading reading
    "Configure Connection Pooling" and named two keys that do not exist.
    """
    sites, _ = _scanned_blocks()
    offenders = [str(site) for site in sites if site.rejected]
    assert not offenders, "documented config-block keys no backend field claims:\n" + "\n".join(
        offenders
    )


def test_the_block_sweep_reads_a_meaningful_corpus() -> None:
    """As for the call form: a scanner that stops matching must fail, not pass.

    Both syntaxes are named. A floor on the total alone would be satisfied by
    either one of them on its own, which is exactly the state this sweep
    replaced: the YAML half read 90 blocks and reported green while the
    Python half read none.
    """
    sites, _ = _scanned_blocks()
    checked = [site for site in sites if site.checked]
    assert len(checked) >= 150, f"only {len(checked)} of {len(sites)} config blocks were checked"
    assert {"memory", "postgres"} <= {site.backend for site in checked}

    suffix = {".md"}
    read = {site.path.suffix for site in checked}
    assert read <= suffix, f"blocks read from unexpected files: {sorted(read - suffix)}"


def test_the_unnamed_block_sweep_reads_a_meaningful_corpus() -> None:
    """The population that used to leave without being counted.

    Counted separately from the named blocks, for the reason
    ``test_both_syntaxes_are_actually_read`` gives: a floor on the total is
    satisfied by either half alone, and this half is the one that has
    already been silent once. ``DB_CONTAINER``, the payload level, or
    ``_recognisable`` each going wrong takes out this number and leaves the
    named floor healthy.
    """
    sites, _ = _scanned_blocks()
    unnamed = [site for site in sites if isinstance(site, UnnamedSite)]
    assert len(unnamed) >= 30, f"only {len(unnamed)} blocks naming no backend were read"


def test_the_share_resembling_no_config_stays_visible() -> None:
    """A block skipped for not resembling a config is counted, not dropped.

    ``_recognisable`` is the one judgement here that says "not mine", and a
    judgement that only skips is one no assertion can watch. Both bounds are
    asserted: a floor, because the honest answer is not zero -- a query
    filter under ``case_db:`` really does sit where a config would -- and a
    ceiling, because a ``_recognisable`` that started answering ``False``
    everywhere would empty the sweep while every other test stayed green.
    """
    _, tally = _scanned_blocks()
    assert tally.unrecognised >= 1, (
        "no block was skipped for resembling no config; either the tree "
        "changed or ``_recognisable`` stopped judging, which would mean the "
        "sweep is reporting on blocks it cannot read"
    )
    assert tally.unrecognised <= 40, (
        f"{tally.unrecognised} blocks sit where a config would and resemble "
        "none, which is more than this corpus should contain"
    )


def test_both_syntaxes_are_actually_read() -> None:
    """A floor on the total cannot see one parser going silent.

    The two front ends fail independently -- a change to the fence languages,
    to ``QUOTED_BACKEND``, or to either shape adapter takes out one and
    leaves the other reporting a healthy corpus. So each is counted.
    """
    from_yaml = from_python = 0
    for path in documentation_files():
        roots, _ = _roots(path)
        for _, shape, _ in roots:
            if shape is YAML_SHAPE:
                from_yaml += 1
            else:
                from_python += 1
    assert from_yaml >= 400, f"only {from_yaml} YAML documents parsed"
    assert from_python >= 100, f"only {from_python} Python fences parsed"


def test_the_share_out_of_scope_stays_visible() -> None:
    """Blocks skipped for their context are counted, not silently dropped.

    ``DB_CONTEXT`` decides whose config a block is. A subsystem it does not
    name has its blocks skipped, which is right -- ``rate_limiters/api`` and
    ``memory/strategies`` both carry ``backend: memory`` and neither holds
    database keys -- but it is also how coverage would quietly erode if the
    expression stopped matching.

    Both directions are asserted, because the expression fails both ways.
    One that matched nothing would skip every block, which the ceiling
    catches. One that matched everything would skip none -- and a ceiling
    reading "the share is small" is satisfied most perfectly by a counter at
    zero, which is the shape this file has already been caught by once.
    """
    sites, tally = _scanned_blocks()
    assert tally.out_of_scope * 2 <= len(sites), (
        f"{tally.out_of_scope} blocks naming a real database backend were "
        f"skipped as belonging to another subsystem, against {len(sites)} "
        "read as database configs"
    )
    assert tally.out_of_scope >= 20, (
        f"only {tally.out_of_scope} blocks were skipped for their context; "
        "the subsystems that carry a backend key without holding database "
        "config have not gone away, so the likelier reading is that "
        "``DB_CONTEXT`` has started matching them"
    )


def test_the_share_that_would_not_parse_stays_visible() -> None:
    """A fence the parser refuses is the one place this sweep can go blind.

    It is a small number and it should stay one. Four YAML fences here are
    schema descriptions rather than configs -- ``temperature: float
    (optional, default: 0.7)`` is prose in YAML clothing -- and two Python
    fences elide their dict with ``...``. All six are non-configs, so
    declining them is right; a sweep that started declining fifty would be
    reporting green over the difference, and one that stopped declining
    anything would be reporting green over a counter that had stopped
    being reached.
    """
    sites, tally = _scanned_blocks()
    assert tally.declined <= 10, (
        f"{tally.declined} fences claiming to hold a database block could "
        f"not be parsed, against {len(sites)} blocks read"
    )
    assert tally.declined >= 1, (
        "no fence failed to parse, which is either the six known non-configs "
        "having been repaired -- in which case lower this floor -- or the "
        "counter no longer being reached, which is what it is here to catch"
    )


def test_a_backend_that_cannot_be_resolved_is_counted_not_dropped() -> None:
    """``backend: ${features.search_backend}`` names nothing checkable.

    That is a fact about the sample, not a defect in it, so it is not
    reported -- but it is the *shape* whose whole population used to vanish.
    The line pattern this replaced required a literal name, so every
    ``backend: ${VAR:default}`` block was neither scanned nor counted, and
    ``out_of_scope`` could not see them either because it counted only lines
    that matched and then failed the context test. Nine such blocks were in
    a database context, one of them carrying a key no backend accepts.
    """
    _, tally = _scanned_blocks()
    assert tally.undecided >= 1, (
        "no block names a backend the sweep cannot resolve; either the tree "
        "changed or ``_backend_name`` stopped answering None, which is how "
        "this population disappeared last time"
    )
    assert tally.undecided <= 5, f"{tally.undecided} blocks name an unresolvable backend"


def _yaml_blocks(text: str, shape: Shape = YAML_SHAPE) -> list[Block]:
    """The blocks a dedented YAML fixture yields, as the sweep reads them."""
    root = yaml.compose(textwrap.dedent(text).strip("\n"))
    return _blocks(root, shape)


def _database_block(text: str, key: str = "backend") -> Block:
    """The one database-context block in a fixture that has exactly one."""
    found = [block for block in _yaml_blocks(text) if block.db and key in block.keys]
    assert len(found) == 1, f"fixture yielded {len(found)} database blocks, wanted 1"
    return found[0]


def test_a_list_items_leading_key_is_read() -> None:
    """The dash puts it two columns left of its siblings, not outside the block.

    The reader this replaced measured depth two ways: ``_ancestry`` counted
    the dash and ``_sibling_keys`` did not, so a leading key measured
    shallower than the block it opens and ended the walk instead of joining
    it. ``name`` leads every documented list today and is routing, which is
    why nothing was being missed -- and why nothing would have noticed when
    something was. A parser has no depth arithmetic to get wrong; the
    fixture stays because that is a claim worth re-checking, not because the
    mechanism is still there.
    """
    block = _database_block(
        """
        databases:
          - pool_size: 20
            backend: postgres
            host: db.internal
        """
    )

    assert block.keys == {"pool_size", "backend", "host"}


def test_a_backend_on_the_dash_line_is_seen() -> None:
    """``- backend: postgres`` is a block, not a non-match.

    A pattern anchored on indentation-then-``backend:`` skipped it silently:
    the site was not recorded as unchecked, it was never recorded at all, so
    no coverage assertion could notice its absence.
    """
    block = _database_block(
        """
        databases:
          - backend: postgres
            pool_size: 20
        """
    )

    assert block.backend == "postgres"
    assert block.keys == {"backend", "pool_size"}


def test_sibling_list_items_are_separate_blocks() -> None:
    """The boundary a raw-whitespace measurement got right by accident.

    Counting the dash without also stopping at one merged a resource list
    into a single block, and every backend in it was then checked against
    every other's keys -- nine documented sites reported keys belonging to
    their neighbours the moment the depth fix landed without this one.
    """
    blocks = [
        block
        for block in _yaml_blocks(
            """
        databases:
          - name: cache
            backend: memory
          - name: search
            backend: elasticsearch
            index: things
        """
        )
        if block.db
    ]

    assert [block.keys for block in blocks] == [
        {"name", "backend"},
        {"name", "backend", "index"},
    ]


def test_a_preceding_sibling_is_not_an_enclosing_parent() -> None:
    """The error that made a regex sweep of the Python form unshippable.

    A prototype searched backwards through the text for a ``"key":`` and
    took what it found as the block's parent. Here that finds
    ``conversation_storage`` and reads a bot's *vector memory* as a database
    config, whose keys answer to a different class entirely. Two documented
    blocks were misread that way, and an unsound sweep is worse here than
    none: a guard that cries wolf is uninstalled by the second person to hit
    it. Nesting settles it, which is the whole reason both forms are parsed.
    """
    source = textwrap.dedent(
        """
        config = {
            "conversation_storage": {"backend": "postgres"},
            "memory": {"type": "vector", "backend": "memory", "dimension": 768},
        }
        """
    )
    blocks = [block for block in _blocks(ast.parse(source), PYTHON_SHAPE) if block.keys]

    assert [(sorted(block.keys), block.db) for block in blocks] == [
        (["conversation_storage", "memory"], False),
        (["backend"], True),
        (["backend", "dimension", "type"], False),
    ], "both blocks must be found; only the storage one is a database config"


def test_a_bare_fragment_is_read_by_wrapping_it() -> None:
    """``"conversation_storage": {...}`` alone in a fence is a dict body.

    It is not a Python module and ``ast.parse`` says so -- ``illegal target
    for annotation`` -- but it is the form the bot documentation uses to show
    one section of a config, and 14 fences here are written that way. Both
    defective pooling samples are among them, so declining the shape would
    decline the finding.
    """
    fragment = '"conversation_storage": {"backend": "postgres", "pool_size": 20}'
    with pytest.raises(SyntaxError):
        ast.parse(fragment)

    blocks = [b for b in _blocks(ast.parse("_ = {\n" + fragment + "\n}"), PYTHON_SHAPE) if b.db]

    assert [block.keys for block in blocks] == [{"backend", "pool_size"}]


# --- The block that names no backend -------------------------------------


def test_a_block_that_names_no_backend_is_still_read(
    document: Callable[[str], tuple[list[AnySite], Tally]],
) -> None:
    """The population this file could not reach, and its cost.

    A config block is checked by resolving its ``backend:`` to a config class
    and asking that class. A block naming none had no class, so it left the
    sweep unchecked -- and a document teaching ``pool_size`` under a bare
    ``database:`` key was corrected by hand for exactly that reason, which is
    the mechanism this file exists to make unnecessary.

    A key **no** config class accepts is wrong whichever backend the block
    turns out to be, so the backend does not have to be known to answer.
    """
    sites, _ = document(
        """
        ```yaml
        databases:
          primary:
            host: localhost
            username: app
        ```
        """
    )
    assert [site.rejected for site in sites] == [["username"]]


def test_an_entry_name_is_not_read_as_a_config_key(
    document: Callable[[str], tuple[list[AnySite], Tally]],
) -> None:
    """``databases:`` holds entries; its keys are names a reader chose.

    Reading the container as a config reports every name a document invents
    -- ``primary``, ``cache``, ``archive`` -- as a rejected key. Measured on
    this corpus, a version that did produced 49 findings of which 34 were
    entry names.
    """
    sites, _ = document(
        """
        ```yaml
        databases:
          primary:
            host: localhost
          cache:
            host: elsewhere
        ```
        """
    )
    assert [site.rejected for site in sites] == [[], []]


def test_a_subsection_is_judged_by_its_own_key_not_its_contents(
    document: Callable[[str], tuple[list[AnySite], Tally]],
) -> None:
    """``pool:`` is the finding; ``min_size`` inside it is not a second one.

    A nested mapping under a config belongs to that config's schema, so the
    question is whether the *section* is accepted -- once. Descending into it
    reports each interior key separately and buries the one fact that
    matters, and the interior of a section that does not exist has no
    accepted spelling to be measured against.
    """
    sites, _ = document(
        """
        ```yaml
        database:
          host: localhost
          pool:
            min_size: 5
            max_size: 20
        ```
        """
    )
    assert [site.rejected for site in sites] == [["pool"]]


def test_a_config_whose_every_key_is_a_subsection_is_still_read(
    document: Callable[[str], tuple[list[AnySite], Tally]],
) -> None:
    """The block carrying the finding can have no accepted key of its own.

    ``postgres:`` / ``pool:`` / ``min_size: 5`` names nothing any config
    class accepts -- ``pool`` is the finding, and it is the block's only key.
    Requiring an accepted key as the price of being read would drop exactly
    the block being reported on, so a nested section counts as evidence that
    this is a config in its own right.
    """
    sites, _ = document(
        """
        ```yaml
        databases:
          postgres:
            pool:
              min_size: 5
        ```
        """
    )
    assert [site.rejected for site in sites] == [["pool"]]


def test_a_subsection_that_opens_a_region_is_read_as_its_own_config(
    document: Callable[[str], tuple[list[AnySite], Tally]],
) -> None:
    """``conversation_storage:`` inside a config is a config, not a section.

    The rule that a subsection belongs to its parent's schema is about keys
    the parent owns. A key that opens a database region of its own is the
    exception, and it has to be, or nesting one config inside another hides
    the inner one exactly the way this file's earlier blind spots did.
    """
    sites, _ = document(
        """
        ```yaml
        database:
          host: localhost
          conversation_storage:
            host: elsewhere
            username: app
        ```
        """
    )
    assert [site.rejected for site in sites] == [[], ["username"]]


def test_a_region_name_used_as_a_field_is_still_a_finding(
    document: Callable[[str], tuple[list[AnySite], Tally]],
) -> None:
    """A region opens on a *mapping*; the same word on a scalar is a key.

    ``conversation_storage:`` holding a block names a nested config and is
    not this config's to accept. ``conversation_storage: memory`` is a key
    with a value, no config class accepts it, and excluding it by name alone
    would lose the finding -- so the mapping is what earns the exemption,
    not the spelling.
    """
    sites, _ = document(
        """
        ```yaml
        database:
          host: localhost
          conversation_storage: memory
        ```
        """
    )
    assert [site.rejected for site in sites] == [["conversation_storage"]]


def test_a_block_resembling_no_config_is_left_alone(
    document: Callable[[str], tuple[list[AnySite], Tally]],
) -> None:
    r"""``\w*_db`` matches a query filter as readily as a database.

    ``default_filters:`` / ``case_db:`` / ``status: published`` is a filter
    on a store, not a configuration of one, and the context rule cannot tell
    them apart by name. A block carrying neither a key some config class
    accepts nor a subsection is not recognisable as a config, and a guard
    that cries wolf gets deleted -- so it is left alone rather than reported.
    """
    sites, _ = document(
        """
        ```yaml
        case_db:
          status: published
        ```
        """
    )
    assert sites == []


@pytest.fixture
def document(tmp_path: Path) -> Callable[[str], tuple[list[AnySite], Tally]]:
    """Write a markdown fixture and read it back through the real sweep.

    Driven through ``_config_blocks`` rather than through the helper under
    test, because a helper can be correct and unreachable: the first version
    of the string-literal test below called ``_embedded_yaml`` directly and
    stayed green when its one call site was deleted.
    """

    def write(text: str) -> tuple[list[AnySite], Tally]:
        path = tmp_path / "doc.md"
        path.write_text(textwrap.dedent(text).strip("\n") + "\n", encoding="utf-8")
        return _config_blocks(path)

    return write


def test_a_config_written_into_a_python_string_is_read(
    document: Callable[[str], tuple[list[AnySite], Tally]],
) -> None:
    """A YAML file shown by assigning it to a name is still a YAML file.

    The line-scanning reader this replaced saw these blocks for the wrong
    reason -- it read the document's text and never asked what a fence was --
    so a parser that dropped them would trade one silent narrowing for
    another. Four documented database blocks live in exactly one such
    string, and one of them names its backend through a substitution.
    """
    sites, _ = document(
        '''
        ```python
        yaml_config = """
        databases:
          - name: primary
            backend: postgres
            max_pool_size: 20
        """
        config.load(yaml.safe_load(yaml_config))
        ```
        '''
    )

    assert [(site.backend, sorted(site.keys)) for site in sites] == [
        ("postgres", ["backend", "max_pool_size", "name"])
    ]


def test_a_fence_that_will_not_parse_is_counted(
    document: Callable[[str], tuple[list[AnySite], Tally]],
) -> None:
    """The one place this sweep can go blind, so it is a number and not a gap.

    An upper bound alone does not hold it: a counter stuck at zero satisfies
    "the share is small" perfectly, which is how the dict-literal form stayed
    invisible for a year. Both directions are asserted -- the count moves
    when a fence declines, and stays put when one parses -- because a
    counter that only ever increments is the same defect wearing the
    opposite sign.
    """
    elided, declining = document(
        """
        ```python
        {"conversation_storage": {"backend": "postgres", ...}}
        ```
        """
    )
    read, parsing = document(
        """
        ```python
        {"conversation_storage": {"backend": "postgres"}}
        ```
        """
    )

    assert (declining.declined, elided) == (1, [])
    assert parsing.declined == 0
    assert [site.backend for site in read] == ["postgres"]


def test_a_block_skipped_for_its_context_is_counted(
    document: Callable[[str], tuple[list[AnySite], Tally]],
) -> None:
    """``DB_CONTEXT`` declining a block is a judgement, not a non-event.

    ``memory`` names a plugin in the vector-store and rate-limiter
    registries as well, so a block carrying it is skipped on the strength of
    its enclosing key alone. That is the right answer and the most fragile
    one here, so the share is watched -- and watching it means the counter
    has to move, which an upper bound cannot check.
    """
    sites, tally = document(
        """
        ```yaml
        conversation_storage:
          backend: memory
        rate_limiters:
          api:
            backend: memory
        ```
        """
    )

    assert [site.backend for site in sites] == ["memory"]
    assert tally.out_of_scope == 1


def test_a_backend_naming_nothing_checkable_is_counted(
    document: Callable[[str], tuple[list[AnySite], Tally]],
) -> None:
    """``backend: ${features.search_backend}`` resolves to no name at all.

    Not a defect in the sample -- it is a fact about it -- but it is the
    shape whose whole population used to vanish. The line pattern this
    replaced required a literal name, so a ``backend: ${VAR:default}`` block
    was neither scanned nor counted, and ``out_of_scope`` could not see them
    either because it counted only lines that matched and then failed the
    context test.
    """
    sites, tally = document(
        """
        ```yaml
        databases:
          - name: search
            backend: ${features.search_backend}
          - name: primary
            backend: ${PRIMARY_DB:postgres}
            max_pool_size: 20
        ```
        """
    )

    assert [site.backend for site in sites] == ["postgres"]
    assert (tally.undecided, tally.out_of_scope) == (1, 0)


def test_a_string_that_is_not_a_mapping_is_left_alone() -> None:
    """Every string is valid YAML as a scalar, so the mapping test is the filter.

    Without it a prose docstring becomes a config block, which is the
    crying-wolf failure this file refuses. 79 literals here compose to a
    mapping and one carries a database block; the rest of the tree's strings
    do not get that far.
    """
    prose = ast.parse('x = "a sentence: with a colon in it"')

    assert _embedded_yaml(prose, 0) == []


def test_a_substitution_resolves_to_the_default_a_reader_installs() -> None:
    """And to nothing at all when there is no default to install."""
    assert _backend_name("${DB_BACKEND:postgres}") == "postgres"
    assert _backend_name("postgres") == "postgres"
    assert _backend_name("${features.search_backend}") is None
    assert _backend_name(None) is None


@pytest.mark.parametrize(
    ("ancestry", "in_scope"),
    [
        (["conversation_storage"], True),
        (["conversations", "databases", "resources"], True),
        (["postgres_db"], True),
        (["production", "profiles", "database"], True),
        (["api", "rate_limiters"], False),
        (["strategies", "memory"], False),
        (["sources", "reasoning"], False),
        (["knowledge", "vector_stores", "resources"], False),
        (["ingredients", "banks", "settings"], False),
    ],
)
def test_the_context_rule_sorts_the_subsystems_it_was_written_for(
    ancestry: list[str], in_scope: bool
) -> None:
    """Pin the classification, so widening or narrowing ``DB_CONTEXT`` is visible.

    Every entry here is an ancestry that occurs in the documentation. The
    false ones all carry a ``backend:`` naming a real database backend --
    that is exactly why the backend name cannot be the discriminator.
    """
    assert any(DB_CONTEXT.match(key) for key in ancestry) is in_scope


def test_every_documented_key_is_one_the_backend_accepts() -> None:
    """No documented config key may be one its backend would reject."""
    sites, _ = _scanned()
    offenders = [str(site) for site in sites if site.rejected]
    assert not offenders, "documented config keys no backend field claims:\n" + "\n".join(offenders)


def test_the_guard_reads_a_meaningful_corpus() -> None:
    """A scanner that silently stops matching would otherwise report green.

    The floor is well under the count at the time of writing; it exists to fail
    when a change to the call pattern or the fence layout makes the sweep stop
    finding anything, not to track the corpus size.
    """
    sites, _ = _scanned()
    checked = [site for site in sites if site.checked]
    assert len(checked) >= 100, (
        f"only {len(checked)} of {len(sites)} documented factory calls were "
        "checked; a site found and skipped satisfies a found-count just as "
        "well as one found and cleared"
    )
    backends = {site.backend for site in checked}
    assert {"memory", "postgres", "elasticsearch"} <= backends, sorted(backends)


def test_the_lenient_share_stays_small() -> None:
    """Sites read against both flavors are the guard's blind spot -- keep them visible.

    A sample giving no signal which factory it drives has its keys checked
    against sync *and* async, so a sync sample using an async-only field passes.
    That is the ``hosts`` defect this guard was written for, so the share of
    such sites is worth watching rather than leaving unmeasured.
    """
    sites, unknown = _scanned()
    assert unknown * 2 <= len(sites), (
        f"{unknown} of {len(sites)} documented calls give no sync/async "
        "signal; the guard cannot catch a sibling-backend field in those"
    )


@pytest.mark.parametrize(
    ("backend", "flavor", "key", "accepted"),
    [
        ("elasticsearch", "sync", "hosts", False),
        ("elasticsearch", "sync", "host", True),
        ("elasticsearch", "async", "hosts", True),
        ("elasticsearch", "async", "username", False),
        ("postgres", "sync", "pool_size", False),
        ("postgres", "sync", "max_pool_size", True),
        ("postgres", "sync", "connection", False),
        ("postgres", "sync", "connection_string", True),
        ("file", "sync", "pretty", False),
        ("file", "sync", "format", True),
    ],
)
def test_the_accepted_sets_are_what_the_guard_assumes(
    backend: str, flavor: str, key: str, accepted: bool
) -> None:
    """Pin the specific accept/reject answers the docstring above claims.

    Without these the guard passes just as well against a config class that has
    quietly gained every key it used to reject.
    """
    classes = _config_classes(backend, flavor)
    assert classes, f"{backend} has no {flavor} config class"
    assert any(cls.accepts(key) for cls in classes) is accepted


# --- The guard's own coverage ---------------------------------------------
#
# Everything above answers "is this documented key wrong?". These answer
# "did the guard actually look?", which is the question a scanner cannot be
# trusted to answer about itself. A site the guard skips is indistinguishable
# in the output from a site it cleared.


def _probe_registry(installed: Any) -> PluginRegistry[Any]:
    """A sync backend registry describing an environment this one is not."""
    registry: PluginRegistry[Any] = PluginRegistry("probe_sync", canonicalize_keys=True)
    _register_sync_backends(registry, installed=installed)
    return registry


def test_a_backend_whose_driver_is_missing_stays_in_coverage() -> None:
    """An absent driver must not quietly subtract a backend from the sweep.

    ``get_factory`` returns ``None`` -- it does not raise -- for a backend
    declared unavailable, so reading ``CONFIG_CLS`` off it yields nothing
    and every key at every Postgres site becomes vacuously accepted. On a
    machine without ``psycopg2`` the guard would check no Postgres sample
    and still report green.

    The config class is a plain dataclass and imports without the driver,
    so the fix is to keep checking rather than to fail: ``load_declared_type``
    reaches it precisely so a caller can read a schema off a plugin it
    cannot build.
    """
    without_psycopg2 = _probe_registry(lambda module: module != "psycopg2")
    assert without_psycopg2.get_factory("postgres") is None, (
        "the probe did not actually simulate the driver's absence"
    )

    config_cls = _config_class(without_psycopg2, "postgres")

    assert config_cls is not None, "postgres dropped out of coverage with its driver"
    assert config_cls.accepts("max_pool_size") and not config_cls.accepts("pool_size")


def test_a_name_outside_the_database_registries_is_out_of_scope_not_clean() -> None:
    """``pgvector`` is a vector store; its keys are not database-config keys.

    The distinction the guard has to make is between a name it *should* be
    checking and cannot, and a name it should not be checking at all. Both
    currently produce an empty class list and so an empty ``rejected``.
    """
    site = Site(Path("doc.md"), 1, "pgvector", "unknown", {"dimensions"})
    assert not site.in_scope
    assert site.rejected == []


def test_a_known_backend_is_in_scope() -> None:
    """The other half of the distinction, so it cannot be satisfied vacuously."""
    assert Site(Path("doc.md"), 1, "postgres", "sync", set()).in_scope


def test_the_guard_checked_what_it_found() -> None:
    """Coverage is asserted, not assumed.

    ``test_the_guard_reads_a_meaningful_corpus`` counts sites *found*. A
    site found and skipped satisfies it just as well as one found and
    checked, which is the hole this closes.
    """
    call, _ = _scanned()
    blocks, _ = _scanned_blocks()
    unchecked = [str(site) for site in (*call, *blocks) if site.in_scope and not site.checked]
    assert not unchecked, "in-scope sites the guard could not check:\n" + "\n".join(unchecked)
