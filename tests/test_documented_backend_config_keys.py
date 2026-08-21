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
    Call-form sites: ``<factory>.create(backend=..., ...)``,
    ``<X>.from_backend("b", {...})``, ``database_factory(...)``. These are
    exactly where the unknown-key check fires.

NOT COVERED
    YAML config blocks; a config bound to a variable and passed by name; and
    ``backend:`` nested inside a bot / knowledge / vector-store config, whose
    enclosing keys belong to that subsystem rather than to a database config.

The accepted-key set is read from the config class through ``accepts()``, never
restated here, so this file cannot drift from the code it checks.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pytest

from dataknobs_data.backends import async_backends, sync_backends
from tests._workspace import documentation_files, rel

#: Keys the object-graph layer owns. They travel with a config on purpose and
#: the runtime check tolerates them, so the guard must too.
ROUTING = frozenset({"backend", "factory", "name", "type", "config"})

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


def _config_classes(backend: str, flavor: str) -> list[type]:
    """The config classes a sample's keys may legitimately belong to.

    ``flavor`` of ``"unknown"`` yields both: where the sample gives no signal
    which factory it drives, a key accepted by either is not evidence of a
    defect. Those sites are counted rather than silently waved through.
    """
    registries = {
        "sync": [sync_backends],
        "async": [async_backends],
        "unknown": [sync_backends, async_backends],
    }[flavor]
    found = []
    for registry in registries:
        try:
            cls = getattr(registry.get_factory(backend), "CONFIG_CLS", None)
        except Exception:  # a name that is not a registered backend
            cls = None
        if cls is not None:
            found.append(cls)
    return found


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
    assert len(sites) >= 100, f"only {len(sites)} documented factory calls found"
    backends = {site.backend for site in sites}
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
