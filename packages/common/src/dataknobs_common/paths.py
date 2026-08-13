"""Compose a filesystem path from untrusted parts without leaving a base.

One check, two spellings of the answer. :func:`safe_join` joins parts
onto a base directory and returns the result only when it stays inside
that directory, so a caller that turns a *name* into a *location* — a
config name, a domain id, a resource path — cannot be talked into
addressing a file the deployment did not put there.
:func:`safe_join_or_raise` is the same check for the common case where
``None`` has nothing useful to mean, and raises :class:`PathEscapeError`
instead.

**Which to call.** Take :func:`safe_join_or_raise` unless the caller has
a real second interpretation of ``None``. Returning the sentinel is
right when the caller probes several candidates and a refusal is one
outcome among them (:func:`~dataknobs_common.config_loading.find_config_file`
weighs every extension before deciding); everywhere else the sentinel
just becomes an ``if x is None: raise`` at the call site, and those
adapters drift — the four in this repo carried three different exception
types and worded the same refusal four ways before they were collapsed
here.

Two spellings escape a base, and both are easy to miss:

* a ``..`` segment walks up out of it — including one that only escapes
  in combination (``sub/../../x``), which no single-segment check sees;
* an **absolute** part discards the base entirely, because that is what
  ``Path("/base") / "/etc/passwd"`` evaluates to. A guard that rejects
  ``..`` and stops there leaves this one open.

Containment is judged **lexically** — ``os.path.normpath`` collapses the
segments and the result is compared component-wise against the base. No
filesystem call is made, which has three consequences worth stating:

* the guard is safe to run on an event loop (a ``Path.resolve()`` stats
  the filesystem and would block it), and it works on paths that do not
  exist yet, which is the case whenever the composed path is about to be
  *written*;
* a symlink is not followed. A file inside ``base`` that symlinks
  outside it is contained by this guard, deliberately: the symlink is
  part of the deployment's own tree — a Kubernetes ConfigMap mount is
  built out of them — and adjudicating it is a different question from
  the one asked here, which is whether the *name* addressed outside;
* correspondingly, a symlink inside ``base`` pointing outside it can
  widen where a contained name lands. Where that matters — a write that
  must not be redirected — check the final destination with
  ``Path.resolve()`` as well, off the event loop.

Example:
    >>> from pathlib import Path
    >>> from dataknobs_common.paths import safe_join
    >>> safe_join(Path("/srv/configs"), "domains/child.yaml")
    PosixPath('/srv/configs/domains/child.yaml')
    >>> safe_join(Path("/srv/configs"), "../../etc/passwd") is None
    True
"""

from __future__ import annotations

import os
from pathlib import Path, PurePath

__all__ = ["PathEscapeError", "safe_join", "safe_join_or_raise"]

#: Characters a path component may not contain whatever it addresses. A
#: NUL terminates the string inside the C library, so ``"a\x00b.yaml"``
#: opens ``a`` — the suffix the guard measured is not the one the kernel
#: sees. It is refused here rather than left to ``open()``, so that every
#: rejection of a bad name arrives as one type from one place.
_FORBIDDEN_IN_PART = ("\x00",)


class PathEscapeError(ValueError):
    """Raised when composing a name onto a base would leave that base.

    Subclasses :class:`ValueError`, which is what the containment sites
    raised before this type existed, so catching the old type still
    works and catching this one is strictly more precise. That precision
    is the point: ``except ValueError`` around a path composition also
    swallows every unrelated ``ValueError`` on the same call —
    ``pydantic.ValidationError`` among them, which subclasses it — so a
    consumer translating "your name addressed outside" into a 400 could
    not tell it apart from a malformed payload.
    """


def _significant(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Erase ``"."`` components, so a curdir base is a prefix of everything.

    ``normpath(".")`` is ``"."`` while ``normpath("./x")`` is ``"x"``, so a
    base of ``"."`` compared as a *string* is a prefix of nothing joined
    onto it -- the bug this guard replaced. Comparing ``PurePath.parts``
    already avoids it, because pathlib drops ``"."`` when it splits
    (``PurePath(".").parts`` is ``()`` on both flavours), so in practice
    this filter removes nothing.

    It is kept because the invariant it states -- a curdir base
    contributes no components -- is what makes the prefix comparison
    total over relative bases, and stating it here means the guard does
    not silently depend on that pathlib detail holding.
    """
    return tuple(p for p in parts if p != os.curdir)


def safe_join(base: str | Path, *parts: str) -> Path | None:
    """Join ``parts`` onto ``base``, or ``None`` if the result escapes it.

    Args:
        base: Directory the composed path must stay inside. May be
            relative; containment is then judged relative to it without
            consulting the current working directory.
        parts: Path components to append. Any of them may contain
            separators — a name addressing a subdirectory is contained
            and is returned — and an interior ``..`` that does not leave
            ``base`` (``a/../b``) is contained too.

    Returns:
        The joined path with ``.`` and ``..`` segments collapsed, or
        ``None`` when it addresses anything outside ``base``, or when a
        part contains NUL. ``base`` itself is contained, so joining no
        parts returns it.
    """
    if any(c in part for part in parts for c in _FORBIDDEN_IN_PART):
        return None

    base_path = Path(base)
    candidate = base_path.joinpath(*parts)

    base_norm = PurePath(os.path.normpath(base_path))
    candidate_norm = PurePath(os.path.normpath(candidate))

    # An absolute part replaces the base, and a relative base cannot be
    # compared against an absolute result without resolving the working
    # directory -- which is filesystem state this guard does not read.
    if base_norm.is_absolute() != candidate_norm.is_absolute():
        return None

    base_parts = _significant(base_norm.parts)
    candidate_parts = _significant(candidate_norm.parts)

    if candidate_parts[: len(base_parts)] != base_parts:
        return None
    # After normalization ``..`` survives only as a leading segment, which
    # is reachable here only when ``base`` is relative (an absolute base
    # absorbs them). ``sub/../../x`` under a relative base lands here.
    if os.pardir in candidate_parts[len(base_parts) :]:
        return None

    return Path(candidate_norm)


def safe_join_or_raise(
    base: str | Path,
    *parts: str,
    what: str,
    outside: str,
    supplied: str | None = None,
) -> Path:
    """Join ``parts`` onto ``base``, or raise :class:`PathEscapeError`.

    :func:`safe_join` with the refusal spelled as an exception. Use it
    wherever a refusal is simply an error — which is every composing
    site that is about to read, write or delete the result.

    Args:
        base: Directory the composed path must stay inside.
        parts: Path components to append, as :func:`safe_join` takes
            them.
        what: What the caller supplied, named as the caller would
            recognise it (``"domain_id"``, ``"subflow name"``). It opens
            the message, so it reads as a sentence about the input
            rather than about this function.
        outside: The boundary, named in the deployment's own terms
            (``"the backend's base path"``). Phrase it as a noun that
            follows "outside".
        supplied: The value to quote back. Defaults to the last part —
            pass it explicitly when the last part is a *derived* string
            (a name with ``.yaml`` appended, a prefixed draft filename)
            so the message names what the caller actually passed.

    Returns:
        The joined path, with its segments collapsed. Open **this**
        rather than recomposing from the raw name: a symlinked
        subdirectory inside ``base`` plus a ``..`` resolves through the
        link's target, so the two are not the same path.

    Raises:
        PathEscapeError: The composed path addresses something outside
            ``base``, or a part contains NUL.
    """
    joined = safe_join(base, *parts)
    if joined is None:
        shown = supplied if supplied is not None else (parts[-1] if parts else "")
        raise PathEscapeError(f"{what} addresses a location outside {outside}: {shown!r}")
    return joined
