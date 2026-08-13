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
from dataclasses import dataclass
from pathlib import Path, PurePath

__all__ = ["PathAnchor", "PathEscapeError", "safe_join", "safe_join_or_raise"]

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


@dataclass(frozen=True)
class PathAnchor:
    """A boundary that stays fixed, and a position inside it that moves.

    For a loader that follows references from one file to another — an
    ``$include`` chain, a wizard subflow naming another wizard — the base a
    name *resolves* against and the boundary it may not *leave* are two
    different things, and conflating them breaks one of two ways.

    Bounding each hop to its own file's directory rejects
    ``sub/frag.yaml`` naming ``../shared.yaml``, which is inside the tree and
    is the ordinary shape of a shared-fragment directory. Bounding to the
    outermost base but *resolving* from it changes what every nested name
    means. Keeping the two separate — resolve per hop, contain against the
    root — is neither.

    Why a type rather than two arguments: the pair has an invariant (the
    position is inside the root, expressed relative to it) that nothing
    checks if the two travel separately, and a recursive loader is exactly
    where a swapped or stale one is easy to write and hard to see. Both
    fields together also make one call site — :meth:`resolve` — instead of a
    join whose arguments each caller assembles.

    The invariant is **enforced here**, not left to call order.
    :meth:`resolve` bounds what it returns, but :attr:`base` is a plain join
    and is the accessor a caller reaches for first, so a position that
    escaped would hand out a path nothing on the way had checked. Every
    anchor therefore validates its own position on construction, which makes
    :meth:`descend` refuse at the hop that left the tree rather than at some
    later use of the result.

    Attributes:
        root: The tree. Every name resolved through this anchor, at any
            depth, addresses inside it. Coerced to :class:`~pathlib.Path` and
            normalised lexically.
        rel_base: Directory of the file currently being read, relative to
            ``root``. ``""`` when that file is at the root. An absolute
            position inside ``root`` is re-expressed relative to it — a
            reference may be spelled absolutely and still land in the tree,
            and two spellings of one directory should be one anchor.
    """

    root: Path
    rel_base: str = ""

    def __post_init__(self) -> None:
        """Normalise the pair, and refuse a position outside the root.

        Raises:
            PathEscapeError: ``rel_base`` addresses outside ``root``.
        """
        root = Path(os.path.normpath(Path(self.root)))
        object.__setattr__(self, "root", root)

        position = str(self.rel_base)
        contained = safe_join(root, position) if position else root
        if contained is None:
            raise PathEscapeError(
                f"anchor position addresses a location outside its root: "
                f"{position!r} is not under {str(root)!r}"
            )

        relative = str(contained.relative_to(root))
        object.__setattr__(self, "rel_base", "" if relative == os.curdir else relative)

    @property
    def base(self) -> Path:
        """The directory references currently resolve from.

        Inside :attr:`root` by construction — see the class docstring.
        """
        return self.root / self.rel_base

    @classmethod
    def anchored_at(cls, entry: str | Path, root: str | Path | None = None) -> PathAnchor:
        """Anchor a tree around the file a load starts from.

        Args:
            entry: The file being loaded. Its directory becomes the starting
                position.
            root: The tree it may address within. Defaults to ``entry``'s own
                directory, which is the tree for the usual single-directory
                layout. Pass a wider one for a deployment that deliberately
                spans sibling directories — widening the boundary keeps it a
                boundary, where switching the check off does not.

        Returns:
            An anchor positioned at ``entry``'s directory.

        Raises:
            PathEscapeError: ``root`` was given and does not contain
                ``entry``. The two arguments disagree about which tree is
                being loaded, and no reading of that is not a mistake.
        """
        return cls.rooted_at(Path(entry).parent, root, _named=str(entry))

    @classmethod
    def rooted_at(
        cls,
        directory: str | Path,
        root: str | Path | None = None,
        _named: str | None = None,
    ) -> PathAnchor:
        """Anchor a tree around the directory a load resolves from.

        :meth:`anchored_at` for a caller that holds the directory rather than
        a file — a loader given a base path with the configuration already in
        hand, for instance.

        Args:
            directory: Where references resolve from.
            root: The tree they may address within. Defaults to ``directory``.
            _named: What to call ``directory`` in a refusal. Internal.

        Returns:
            An anchor positioned at ``directory``.

        Raises:
            PathEscapeError: ``root`` was given and does not contain
                ``directory``, or the two are not spelled the same way.
        """
        if root is None:
            return cls(Path(directory))

        root_path = Path(os.path.normpath(Path(root)))
        dir_path = Path(os.path.normpath(Path(directory)))
        shown = _named if _named is not None else str(directory)

        # Containment is judged lexically, so a relative path cannot be
        # compared against an absolute one without reading the working
        # directory. Named separately because the two may well denote the
        # same place, and "lies outside the tree" reads as a mistake about
        # *where* the file is rather than about how it was spelled.
        if root_path.is_absolute() != dir_path.is_absolute():
            raise PathEscapeError(
                f"entry file and the tree it declares must both be absolute or "
                f"both relative: {shown!r} against {str(root_path)!r}"
            )

        try:
            rel = str(dir_path.relative_to(root_path))
        except ValueError:
            raise PathEscapeError(
                f"entry file lies outside the tree it declares: "
                f"{shown!r} is not under {str(root_path)!r}"
            ) from None
        return cls(root_path, rel)

    def resolve(self, *parts: str, what: str, outside: str, supplied: str | None = None) -> Path:
        """Resolve a referenced name from here, or refuse to leave the root.

        Args:
            parts: The reference as the referencing file spelled it, resolved
                relative to :attr:`rel_base`. Several parts join as one name,
                for a caller assembling a layout convention around it
                (``"subflows", f"{name}.yaml"``).
            what: What the reference is, named as its author would recognise
                it (``"$include"``, ``"subflow name"``).
            outside: The boundary in the deployment's own terms
                (``"the configuration tree"``).
            supplied: The value to quote back, when ``parts`` is a derived
                spelling rather than what the author wrote.

        Returns:
            The resolved path, with its segments collapsed. Open this rather
            than recomposing from the raw name.

        Raises:
            PathEscapeError: The reference addresses outside :attr:`root`.
        """
        return safe_join_or_raise(
            self.root,
            self.rel_base,
            *parts,
            what=what,
            outside=outside,
            supplied=supplied,
        )

    def descend(self, *parts: str) -> PathAnchor:
        """Move to the directory of a file this one references.

        The new position is computed from the reference rather than from a
        resolved path, so it stays a pure string operation with nothing to
        re-derive against a base the path might not be under.

        Args:
            parts: The same reference passed to :meth:`resolve`.

        Returns:
            An anchor with the same root, positioned at the referenced file's
            directory.

        Raises:
            PathEscapeError: The reference leaves ``root``. Unreachable
                through a loader that calls :meth:`resolve` on each hop
                before descending it, which is why the check belongs on the
                type rather than at those call sites.
        """
        joined = str(PurePath(self.rel_base, *parts)) if parts else self.rel_base
        moved = PurePath(os.path.normpath(joined)).parent
        # Curdir and absolute spellings are collapsed by ``__post_init__``,
        # so two anchors meaning the same directory compare equal however
        # each of them got here.
        return PathAnchor(self.root, str(moved))
