"""Compose a filesystem path from untrusted parts without leaving a base.

One function. :func:`safe_join` joins parts onto a base directory and
returns the result only when it stays inside that directory, so a caller
that turns a *name* into a *location* — a config name, a domain id, a
resource path — cannot be talked into addressing a file the deployment
did not put there.

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

__all__ = ["safe_join"]


def _significant(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Drop a lone ``"."``, which ``normpath`` leaves behind for a curdir base.

    ``normpath(".")`` is ``"."`` while ``normpath("./x")`` is ``"x"``, so a
    base of ``"."`` is not a component prefix of anything joined onto it.
    Erasing it makes the prefix comparison total over relative bases.
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
        ``None`` when it addresses anything outside ``base``. ``base``
        itself is contained, so joining no parts returns it.
    """
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
