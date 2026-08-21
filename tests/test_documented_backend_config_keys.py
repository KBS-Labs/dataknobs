"""Reproduce-first guard: a config key shown in the docs must be one a backend takes.

Sibling of ``test_documented_imports.py``, and for the same reason. That guard
exists because an import is mechanically checkable and so has no excuse for
being wrong. A backend config key became mechanically checkable when
``StructuredConfig.accepts()`` landed, and it turned out to be wrong at sixteen
call sites across eight documents.

None of them could fail anything before. ``mkdocs build --strict`` validates
links and nav, the doc-mirror manifest validates that two copies agree, and the
import guard reads only the ``import`` lines. A sample whose imports resolve and
whose keys are fiction passes all three -- and used to pass at runtime too,
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

    **YAML form** -- a ``backend:`` line in a config block, where the
    enclosing keys say the block is a database config. ``DB_CONTEXT`` makes
    that judgement; ``memory`` names a plugin in the vector-store and
    rate-limiter registries too, so the backend name alone cannot.

NOT COVERED
    A config bound to a variable and passed by name. Which binding a name
    refers to is not decidable by reading a prose document: one sample hands
    ``AsyncElasticsearchDatabase`` a ``config`` the surrounding text never
    defines, and the nearest earlier binding of that name belongs to a
    different example. Reporting that as a defect would make the guard cry
    wolf, and a guard that cries wolf gets deleted. Those sites are checked
    by hand when the surrounding page is edited.

    A quoted dict literal handed to ``Config.load`` rather than to a factory
    or a constructor -- ``{"databases": [{"factory": "database", "backend":
    "memory", ...}]}`` -- which the object-graph layer builds later, by name.
    This is a real gap rather than an undecidable one, and it is the largest
    of the three: a sweep over the corpus finds 109 such blocks whose
    enclosing key is a database context, against 77 it correctly declines.
    Closing it needs the ancestry rule ``_ancestry`` applies to YAML rebuilt
    for Python syntax, and a prototype of that still mistook two vector-store
    blocks for database ones -- an unsound sweep is worse here than none,
    for the crying-wolf reason above. Tracked separately; the one site of
    the nine it flags that a runnable example actually builds is fixed.

The accepted-key set is read from the config class through ``accepts()``, and
the tolerated routing keys from the runtime's own ``_ROUTING_KEYS``, so this
file cannot drift from the code it checks. Coverage is asserted rather than
assumed: a backend whose driver is absent is still read against its declared
config class, and the corpus floor counts the sites actually checked.
"""

from __future__ import annotations

import re
import textwrap
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from dataknobs_common.registry import PluginRegistry
from dataknobs_common.structured_config import _ROUTING_KEYS, StructuredConfig
from dataknobs_data.backends import _register_sync_backends, async_backends, sync_backends
from tests._workspace import documentation_files, rel

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


# --- The YAML form -------------------------------------------------------
#
# A ``backend:`` line in a config block reaches the same factory as a call,
# and the keys beside it are the same keys. The difficulty is not parsing
# them, it is knowing whose they are: ``memory`` names a plugin in the
# database registry, the vector-store registry and the rate-limiter registry
# alike, so the backend name cannot say which config a block is. Only the
# enclosing keys can.

#: An ancestry segment that says the block below it holds a *database*
#: config. ``databases:`` is the environment-resources spelling and the
#: plural mapping in a config file, ``database:`` its singular, and a
#: ``<name>_db:`` block is the shape the reference docs use for a standalone
#: sample. ``conversation_storage:`` is a bot's, and it is passed to the
#: factory verbatim.
#:
#: The rule is stated as an allowlist rather than a list of subsystems to
#: skip because the failure directions are not symmetric: an unrecognised
#: context leaves a block unchecked, which ``test_the_yaml_share_out_of_scope
#: _stays_visible`` counts, whereas a denylist that missed a new subsystem
#: would report its keys as defects -- and a guard that cries wolf is
#: uninstalled by the second person to hit it.
DB_CONTEXT = re.compile(r"^(databases?|conversation_storage|\w*_db)$")

#: ``key:`` at some indentation, the unit both YAML helpers below work in.
YAML_KEY = re.compile(r"^(?P<indent>\s*)(?P<dash>-\s+)?(?P<key>[\w-]+):(?P<rest>.*)$")

#: The ``backend:`` line that opens a block. The optional dash matters: a
#: block written as a list element puts ``backend:`` on the dash line itself
#: (``- backend: postgres``), and a pattern anchored on ``\s*backend:`` does
#: not match it -- so the block was not read as unchecked, it was not seen.
BACKEND_LINE = re.compile(
    r"^(?P<indent>\s*)(?P<dash>-\s+)?backend:\s*['\"]?(?P<b>[\w-]+)['\"]?\s*(#.*)?$"
)


def _ancestry(lines: list[str], at: int, indent: int) -> list[str]:
    """The mapping keys enclosing line ``at``, innermost first."""
    chain: list[str] = []
    want = indent
    for j in range(at - 1, -1, -1):
        line = lines[j]
        if not line.strip() or line.strip().startswith("#"):
            continue
        m = YAML_KEY.match(line)
        if m is None:
            continue
        depth = len(m.group("indent")) + (len(m.group("dash")) if m.group("dash") else 0)
        if depth < want:
            chain.append(m.group("key"))
            want = depth
            if want == 0:
                break
    return chain


def _sibling_keys(lines: list[str], at: int, width: int) -> set[str]:
    """Every key in the mapping block that the ``backend:`` on line ``at`` is in.

    Deeper lines are stepped over rather than treated as the end of the
    block: a key whose value is a nested list or mapping is still a key of
    this block, and stopping at the first one truncates the read. That is
    not hypothetical -- an Elasticsearch sample whose ``hosts:`` is a list
    was read as having exactly two keys, so the ``username`` / ``password``
    / ``use_ssl`` below it were never checked against anything.
    """
    keys: set[str] = set()
    opening = YAML_KEY.match(lines[at])
    if opening is not None:
        keys.add(opening.group("key"))
    for forward in (True, False):
        span = range(at + 1, len(lines)) if forward else range(at - 1, -1, -1)
        for j in span:
            line = lines[j]
            if not line.strip() or line.strip().startswith("#"):
                continue
            m = YAML_KEY.match(line)
            if m is None:
                # Not a ``key:`` line -- a list element or a bare scalar.
                # Deeper content belongs to one of this block's keys;
                # anything shallower ends the block.
                if len(line) - len(line.lstrip()) > width:
                    continue
                break
            # Dash-aware, matching ``_ancestry``: ``  - name: cache`` puts a
            # list item's first key two columns left of its siblings, so
            # measuring raw whitespace both mismeasured that key and ended
            # the walk on the very line that opens the block.
            depth = len(m.group("indent")) + (len(m.group("dash")) if m.group("dash") else 0)
            if depth > width:
                continue  # nested under one of this block's keys
            if depth < width:
                break
            if m.group("dash"):
                # A dash at this depth opens a list item, which is the
                # block boundary a raw-whitespace measurement got right by
                # accident. Backwards it opens *this* item, so its key is
                # ours and the walk ends having taken it; forwards it opens
                # the next one, whose keys are a different block's. Without
                # this, a documented resource list reads as one block and
                # every backend in it is checked against every other's keys.
                if not forward:
                    keys.add(m.group("key"))
                break
            keys.add(m.group("key"))
    return keys


def _yaml_sites() -> tuple[list[Site], int]:
    """Every documented YAML database-config block, and how many were out of scope."""
    sites: list[Site] = []
    out_of_scope = 0
    for path in documentation_files():
        lines = path.read_text(encoding="utf-8").split("\n")
        for i, line in enumerate(lines):
            m = BACKEND_LINE.match(line)
            if m is None:
                continue
            width = len(m.group("indent")) + (len(m.group("dash")) if m.group("dash") else 0)
            if not any(DB_CONTEXT.match(key) for key in _ancestry(lines, i, width)):
                # Counted only when the name is one a database registry
                # knows. A ``backend: faiss`` block is out of scope for a
                # reason that needs no watching; a ``backend: memory`` one
                # is out of scope only because its *context* said so, and
                # that is the judgement worth keeping visible.
                if any(reg.is_known(m.group("b")) for reg in (sync_backends, async_backends)):
                    out_of_scope += 1
                continue
            sites.append(Site(path, i + 1, m.group("b"), "unknown", _sibling_keys(lines, i, width)))
    return sites, out_of_scope


@cache
def _scanned_yaml() -> tuple[list[Site], int]:
    """Scan once; every YAML test below reads the same result."""
    return _yaml_sites()


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


def test_every_documented_yaml_key_is_one_the_backend_accepts() -> None:
    """The same rule as the call form, on the shape most consumers copy.

    A bot's ``conversation_storage:`` block reaches
    ``AsyncDatabaseFactory.create(**config)`` with every key it carries, so
    a ``pool_size:`` there is not decoration -- it is a key the config class
    now rejects, in a document telling the reader to write it.
    """
    sites, _ = _scanned_yaml()
    offenders = [str(site) for site in sites if site.rejected]
    assert not offenders, "documented YAML config keys no backend field claims:\n" + "\n".join(
        offenders
    )


def test_the_yaml_sweep_reads_a_meaningful_corpus() -> None:
    """As for the call form: a scanner that stops matching must fail, not pass."""
    sites, _ = _scanned_yaml()
    checked = [site for site in sites if site.checked]
    assert len(checked) >= 60, f"only {len(checked)} of {len(sites)} YAML blocks were checked"
    assert {"memory", "postgres"} <= {site.backend for site in checked}


def test_the_yaml_share_out_of_scope_stays_visible() -> None:
    """Blocks skipped for their context are counted, not silently dropped.

    ``DB_CONTEXT`` decides whose config a block is. A subsystem it does not
    name has its blocks skipped, which is right -- ``rate_limiters/api`` and
    ``memory/strategies`` both carry ``backend: memory`` and neither holds
    database keys -- but it is also how coverage would quietly erode if the
    expression stopped matching. The share is small; assert that it is.
    """
    sites, out_of_scope = _scanned_yaml()
    assert out_of_scope * 4 <= len(sites), (
        f"{out_of_scope} YAML blocks naming a real database backend were "
        f"skipped as belonging to another subsystem, against {len(sites)} "
        "read as database configs"
    )


def _yaml(text: str) -> list[str]:
    """A dedented YAML fixture as the scanners see it."""
    return textwrap.dedent(text).strip("\n").split("\n")


def test_a_list_items_leading_key_is_read() -> None:
    """The dash puts it two columns left of its siblings, not outside the block.

    ``_ancestry`` counted the dash from the start and ``_sibling_keys`` did
    not, so a leading key measured shallower than the block it opens and
    ended the walk instead of joining it. ``name`` leads every documented
    list today and is routing, which is why nothing was being missed -- and
    why nothing would have noticed when something was.
    """
    lines = _yaml(
        """
        databases:
          - pool_size: 20
            backend: postgres
            host: db.internal
        """
    )
    assert _sibling_keys(lines, 2, 4) == {"pool_size", "backend", "host"}


def test_a_backend_on_the_dash_line_is_seen() -> None:
    """``- backend: postgres`` is a block, not a non-match.

    A pattern anchored on indentation-then-``backend:`` skipped it
    silently: the site was not
    recorded as unchecked, it was never recorded at all, so no coverage
    assertion could notice its absence.
    """
    lines = _yaml(
        """
        databases:
          - backend: postgres
            pool_size: 20
        """
    )
    m = BACKEND_LINE.match(lines[1])

    assert m is not None and m.group("b") == "postgres"
    assert _sibling_keys(lines, 1, 4) == {"backend", "pool_size"}


def test_sibling_list_items_are_separate_blocks() -> None:
    """The boundary a raw-whitespace measurement got right by accident.

    Counting the dash without also stopping at one merges a resource list
    into a single block, and every backend in it is then checked against
    every other's keys -- nine documented sites reported keys belonging to
    their neighbours the moment the depth fix landed without this one.
    """
    lines = _yaml(
        """
        databases:
          - name: cache
            backend: memory
          - name: search
            backend: elasticsearch
            index: things
        """
    )

    assert _sibling_keys(lines, 2, 4) == {"name", "backend"}
    assert _sibling_keys(lines, 4, 4) == {"name", "backend", "index"}


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
    sites, _ = _scanned()
    unchecked = [str(site) for site in sites if site.in_scope and not site.checked]
    assert not unchecked, "in-scope sites the guard could not check:\n" + "\n".join(unchecked)
