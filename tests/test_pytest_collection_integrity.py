"""A run that names more than one package must still collect.

Every check in this file exists because of one defect: **which module-naming
rules apply is decided by how the command was invoked.** A single-package run
resolves its package's ``[tool.pytest.ini_options]``; a run naming two packages
— or naming none, which is how ``testpaths`` reaches all of them — resolves the
root ``pytest.ini`` instead. The two declared different import modes, so the
same file was imported by different rules depending on the argument list, and
neither configuration said so.

The visible cost was not in the gate, which runs one pytest process per target
and so never meets a multi-package invocation. It was on the two invocations a
person actually types: a bare ``pytest`` at the repo root, and this suite run
beside the packages it guards.

``test_a_whole_workspace_collection_reports_no_errors`` is the behavioural
backstop and cannot drift from pytest — it asks pytest. The structural checks
below it exist for the error message: they name the colliding directories,
which a collection traceback does not. They must also keep working when
collection is already broken, which is why they read the tree rather than a
successful run.

Two of them guard the guard. A structural check that stops seeing a whole
category of claimant, or quietly passes over a declaration it cannot parse,
does not go quiet — it goes *green*, which is worse than absent, and is the
same shape of failure as the one being checked for.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from tests._workspace import (
    ROOT,
    load_toml,
    pyprojects,
    rel,
    tracked_dirs,
    tracked_files,
)

# The import mode every invocation must resolve. Read from the root config
# rather than written down, so the two cannot disagree silently — a literal
# here would keep passing after someone changed the root and would then be
# asserting the old answer.
_IMPORT_MODE_RE = re.compile(r"--import-mode=(\S+)")


def _root_import_mode() -> str:
    """The ``--import-mode`` the root ``pytest.ini`` declares."""
    text = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    match = _IMPORT_MODE_RE.search(text)
    assert match is not None, (
        "pytest.ini declares no --import-mode. Every package's "
        "[tool.pytest.ini_options] mirrors this value, so removing it here "
        "leaves them asserting nothing — delete the mirrors too, or restore it."
    )
    return match.group(1)


def _package_pytest_blocks() -> dict[str, dict]:
    """Each package ``pyproject.toml`` that declares a pytest block."""
    blocks = {}
    for path in pyprojects():
        if path.parent == ROOT:
            continue
        ini = load_toml(path).get("tool", {}).get("pytest", {}).get("ini_options")
        if ini is not None:
            blocks[rel(path)] = ini
    return blocks


def _root_namespace_claims() -> dict[str, list[str]]:
    """Top-level names the repo root supplies, derived from live imports.

    ``tests/`` carries no ``__init__.py``, so it supplies ``tests`` only
    because something imports through it — which the guards in this directory
    do, for ``tests._workspace``. Derived from those imports rather than
    written down, so deleting the last such import retires the claim instead
    of leaving a stale assertion behind.
    """
    claims: dict[str, list[str]] = {}
    pattern = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)\.", re.M)
    for source in sorted((ROOT / "tests").glob("*.py")):
        for name in pattern.findall(source.read_text(encoding="utf-8")):
            if (ROOT / name).is_dir():
                claims.setdefault(name, [])
                if name not in claims[name]:
                    claims[name].append(name)
    return claims


def _resolve_anchor(node: ast.expr, conftest: Path, env: dict[str, Path]) -> Path:
    """Evaluate the small path grammar the declarations are written in.

    Deliberately tiny, and it raises rather than guessing on anything outside
    it: this guard reads the declarations to learn which directories supply
    top-level names, so a declaration it cannot read is a blind spot, and a
    blind spot that reports success is the thing being guarded against.
    """
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return conftest
        if node.id in env:
            return env[node.id]
        raise ValueError(f"name {node.id!r} is not bound to a path at module scope")
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        return _resolve_anchor(node.value, conftest, env).parent
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        segment = node.right
        if not (isinstance(segment, ast.Constant) and isinstance(segment.value, str)):
            raise ValueError("the right side of `/` is not a string literal")
        return _resolve_anchor(node.left, conftest, env) / segment.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and len(node.args) == 1
    ):
        return _resolve_anchor(node.args[0], conftest, env)
    raise ValueError(f"unsupported expression: {ast.unparse(node)}")


def _declared_import_roots() -> dict[Path, str]:
    """Every directory a ``conftest.py`` puts on ``sys.path``, and who declared it.

    Read from the declarations rather than listed here, because a list would
    keep passing after someone added a root — and an undeclared root is
    precisely a set of top-level names nothing is checking.

    Only ``conftest.py`` is searched. That is not a convenience: pytest loads a
    conftest before collecting anything beside it, so a declaration there holds
    for every module in the tree, while the same call in a collected test
    module runs after its siblings have already been imported.
    """
    roots: dict[Path, str] = {}
    for name in tracked_files():
        if PurePosixPath(name).name != "conftest.py":
            continue
        conftest = ROOT / name
        source = conftest.read_text(encoding="utf-8")
        if "declare_import_root" not in source:
            continue
        tree = ast.parse(source)

        env: dict[str, Path] = {}
        for stmt in tree.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                try:
                    env[stmt.targets[0].id] = _resolve_anchor(stmt.value, conftest, env)
                except ValueError:
                    continue  # not a path expression; nothing to bind

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "declare_import_root"
                and node.args
            ):
                continue
            try:
                anchor = _resolve_anchor(node.args[0], conftest, env)
            except ValueError as exc:
                raise AssertionError(
                    f"{name}:{node.lineno}: cannot resolve the argument to "
                    f"declare_import_root() — {exc}.\n\nThis guard reads the "
                    "declarations to learn which directories supply top-level "
                    "names, so one it cannot read is a hole it would report "
                    "green through. Write the anchor as `__file__`, a name "
                    "bound to a path at module scope, `.parent`, or "
                    '`/ "literal"` — or widen the grammar in _resolve_anchor.'
                ) from exc
            # Mirrors declare_import_root: a file anchors its own directory.
            directory = anchor if anchor.is_dir() else anchor.parent
            roots[directory.resolve()] = f"{name}:{node.lineno}"

    assert roots, (
        "no declare_import_root() call found in any conftest.py. Either the "
        "idiom changed name — in which case this guard is now checking "
        "nothing — or every test tree stopped declaring a root."
    )
    return roots


def _top_level_package_claims() -> dict[str, list[str]]:
    """Map each top-level importable name to every directory supplying it.

    Two mechanisms put a name at top level, and the pair is the point: each is
    invisible from the other's side, and the collision that broke collection
    needed both to be seen at once.

    *pytest's package-root walk.* A directory holding ``__init__.py`` whose
    parent does not is where pytest stops walking up, so the name it hands the
    import system is the directory's own, unqualified.

    *A declared ``sys.path`` root.* Every immediate child of a root is
    importable by its bare name — a directory with ``__init__.py`` as a regular
    package, one without as a PEP 420 namespace portion. This half was missing,
    and it is the larger surface by far: a root exposes *all* its children,
    including the ones nobody thought of as importable.

    The two tiers of collision differ, and the distinction is why this returns
    every claimant rather than only the regular packages. Two namespace
    portions of one name *merge* into a single package searched in
    ``sys.path`` order, so a same-named submodule in each resolves by
    accident of ordering. A regular package among the claimants does not
    merge: it wins outright, no matter which came first, and the others become
    unreachable. The second is what collapsed three trees onto ``tests``; the
    first is the quieter version of the same mistake, and neither is worth
    tolerating on the grounds that the other is worse.
    """
    claims = _root_namespace_claims()

    def claim(directory: Path) -> None:
        where = rel(directory)
        bucket = claims.setdefault(directory.name, [])
        if where not in bucket:
            bucket.append(where)

    for init in sorted(ROOT.glob("packages/*/tests/**/__init__.py")):
        directory = init.parent
        if (directory.parent / "__init__.py").exists():
            continue
        claim(directory)

    tracked = tracked_dirs()
    for root in sorted(_declared_import_roots()):
        for child in sorted(root.iterdir()):
            if child.is_dir() and rel(child) in tracked:
                claim(child)

    return claims


#: How long the child gets before the run is called a failure rather than
#: waited on. Collection imports every test module in the workspace, so any
#: module-level connect, service probe or lock acquisition happens inside this
#: call — and one that never returns would otherwise hang the suite with no
#: output at all, which reads as "still working" for as long as anyone is
#: willing to wait. Generous against the real figure (a few seconds) so that a
#: loaded CI runner does not trip it; it is a deadlock bound, not a budget.
_COLLECT_TIMEOUT = 300.0


def _collect(*targets: str) -> subprocess.CompletedProcess[str]:
    """Run pytest's collection phase in a clean child process."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:randomly",
        "-p",
        "no:cacheprovider",
        *targets,
    ]
    try:
        return subprocess.run(
            argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=_COLLECT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):  # pragma: no cover - text=True is set
            partial = partial.decode(errors="replace")
        raise AssertionError(
            f"collection did not finish within {_COLLECT_TIMEOUT:.0f}s: "
            f"{' '.join(argv[1:])}\n\nA module imported at collection time is "
            "blocking — a connection, a service probe, or a lock taken at "
            "module scope. Collection must not do I/O that can wait.\n\n"
            f"Last output:\n{partial[-2000:]}"
        ) from exc


def test_a_whole_workspace_collection_reports_no_errors() -> None:
    """A bare ``pytest`` at the repo root must collect every test.

    This is the invocation with no arguments to get wrong, and ``testpaths``
    points it at every package plus this directory — so it is also the widest
    multi-package run there is. Asking pytest rather than re-deriving its
    naming rules is the point: a check that reimplements them can drift into
    agreeing with itself while the real run still aborts.
    """
    result = _collect()
    errors = sorted(
        line for line in result.stdout.splitlines() if line.startswith("ERROR ")
    )
    assert not errors, (
        "a bare `pytest` at the repo root does not collect:\n  "
        + "\n  ".join(errors)
        + "\n\nEach of these resolved a module name that another file in the "
        "run had already claimed. See the collision report from "
        "test_no_two_directories_claim_the_same_top_level_name."
    )
    assert result.returncode == 0, (
        f"collection exited {result.returncode} without naming an ERROR line:\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )


def test_no_two_directories_claim_the_same_top_level_name() -> None:
    """No top-level importable name may be supplied by two directories.

    The failure this catches is silent in a single-package run and fatal in
    any run spanning both claimants — which is why it went unnoticed: every
    package passes alone.
    """
    claims = _top_level_package_claims()
    collisions = {name: dirs for name, dirs in claims.items() if len(dirs) > 1}

    def describe(where: str) -> str:
        # Which tier: a regular package wins outright, namespace portions merge
        # and then resolve by sys.path order. The reader needs to know which.
        kind = "package" if (ROOT / where / "__init__.py").exists() else "namespace"
        return f"{where} [{kind}]"

    assert not collisions, (
        "top-level import names supplied by more than one directory:\n"
        + "\n".join(
            f"  {name!r} supplied by: {', '.join(describe(d) for d in dirs)}"
            for name, dirs in sorted(collisions.items())
        )
        + "\n\nA regular package among the claimants wins outright and the "
        "others become unreachable; all-namespace claimants merge into one "
        "package searched in sys.path order. Either way a bare `import "
        "<name>` means different things depending on which run it is in."
    )


def test_the_claim_map_sees_namespace_portions_not_only_packages() -> None:
    """The half that was missing must be contributing something.

    Before, claims were read only from ``__init__.py``-bearing directories, so
    every namespace portion a declared root exposes was invisible — which is
    most of what a root exposes. This asserts at least one claim comes from a
    root's child that carries no ``__init__.py``: narrow the enumeration back
    to packages and the collision check goes quiet again, still passing.
    """
    roots = set(_declared_import_roots())
    portions = sorted(
        where
        for supplied in _top_level_package_claims().values()
        for where in supplied
        if (ROOT / where).resolve().parent in roots
        and not (ROOT / where / "__init__.py").exists()
    )
    assert portions, (
        "no claimed directory is a namespace portion under a declared import "
        "root. Every root exposes all of its children, so this should not be "
        "empty while any root has a subdirectory — the enumeration in "
        "_top_level_package_claims has probably narrowed back to packages."
    )


def test_an_unreadable_root_declaration_is_an_error_not_a_skip() -> None:
    """A declaration this guard cannot parse must fail, never be passed over.

    Skipping one would leave its directory's children unchecked while the
    guard still reported success — the failure mode the whole file exists to
    prevent, reproduced inside the guard itself.
    """
    unreadable = ast.parse("some_helper()", mode="eval").body
    with pytest.raises(ValueError, match="unsupported expression"):
        _resolve_anchor(unreadable, ROOT / "conftest.py", {})


@pytest.mark.parametrize("pyproject", sorted(_package_pytest_blocks()))
def test_every_package_pytest_block_mirrors_the_root_import_mode(
    pyproject: str,
) -> None:
    """A package's pytest block must resolve the same import mode as the root.

    Only one of the two configurations applies to any given run, and which one
    is decided by the argument list. When they disagree, a file that imports
    cleanly under ``pytest packages/llm/tests`` fails under ``pytest
    packages/llm/tests packages/common/tests`` — with nothing in either
    configuration to suggest the argument list was load-bearing.

    ``asyncio_mode`` was already mirrored for exactly this reason, and the
    worked example in the comment beside it is a run that failed on import
    mode for years afterwards.
    """
    ini = _package_pytest_blocks()[pyproject]
    addopts = ini.get("addopts", [])
    if isinstance(addopts, str):
        addopts = addopts.split()
    declared = [_IMPORT_MODE_RE.match(opt) for opt in addopts]
    found = [m.group(1) for m in declared if m is not None]
    expected = _root_import_mode()
    assert found == [expected], (
        f"{pyproject} declares import mode {found or 'nothing'}; the root "
        f"pytest.ini declares {expected!r}. A run that resolves this block "
        f"would import the same files by different rules than a run that "
        f"resolves the root."
    )
