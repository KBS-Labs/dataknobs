"""Reproduce-first tests for the doc-mirror guard (``bin/docs-mirror-check.py``).

Each test exercises a specific failure mode the guard must catch (drift,
symlink/transclude replacement, unclassified docs, manifest references to
missing files, ambiguous line exceptions) plus the clean-tree pass and the
``--fix`` idempotence the checker promises. Everything runs against a sandbox
tree under ``tmp_path`` with the module's ``ROOT``/``MANIFEST`` globals patched
to that sandbox, so no test touches the repo's real manifest or docs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "bin" / "docs-mirror-check.py"


@pytest.fixture(scope="module")
def mirror_mod():
    spec = importlib.util.spec_from_file_location("docs_mirror_check", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tree(tmp_path, monkeypatch, mirror_mod):
    """A patched ``(module, pkg_dir, site_dir)`` sandbox rooted at ``tmp_path``."""
    pkg_dir = tmp_path / "packages" / "demo" / "docs"
    site_dir = tmp_path / "docs" / "packages" / "demo"
    pkg_dir.mkdir(parents=True)
    site_dir.mkdir(parents=True)
    monkeypatch.setattr(mirror_mod, "ROOT", tmp_path)
    monkeypatch.setattr(
        mirror_mod, "MANIFEST", tmp_path / ".dataknobs" / "docs-mirror-manifest.json"
    )
    return mirror_mod, pkg_dir, site_dir


def _w(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _manifest(**entry) -> dict:
    base = {"package_dir": "packages/demo/docs", "site_dir": "docs/packages/demo"}
    base.update(entry)
    return {"packages": {"demo": base}}


# --------------------------------------------------------------------------
# Link canonicalization (the core comparison primitive)
# --------------------------------------------------------------------------


def test_canonicalize_line_rewrites_bare_md_link(mirror_mod):
    assert mirror_mod.canonicalize_line("see [X](FOO_BAR.md)") == "see [X](foo-bar.md)"


def test_canonicalize_line_preserves_anchor(mirror_mod):
    got = mirror_mod.canonicalize_line("[X](FOO_BAR.md#the-section)")
    assert got == "[X](foo-bar.md#the-section)"


def test_canonicalize_line_leaves_paths_urls_and_anchors(mirror_mod):
    for target in ("sub/FOO.md", "https://x.test/FOO.md", "#same-page", "FOO.txt"):
        line = f"[X]({target})"
        assert mirror_mod.canonicalize_line(line) == line


def test_fenced_code_block_link_is_not_rewritten(mirror_mod):
    """Finding 3: a link-like token inside a ``` fence is literal example text."""
    text = "\n".join(
        [
            "prose [A](FIRST_ONE.md)",
            "```markdown",
            "example [B](SECOND_ONE.md)",
            "```",
            "more [C](THIRD_ONE.md)",
        ]
    )
    out = mirror_mod.canonicalize_text(text)
    assert out[0] == "prose [A](first-one.md)"  # prose rewritten
    assert out[2] == "example [B](SECOND_ONE.md)"  # fenced content untouched
    assert out[4] == "more [C](third-one.md)"  # fence closed, prose rewritten again


def test_tilde_fence_and_length_tracked(mirror_mod):
    text = "\n".join(["~~~", "[B](INSIDE_TILDE.md)", "~~~", "[C](OUTSIDE.md)"])
    out = mirror_mod.canonicalize_text(text)
    assert out[1] == "[B](INSIDE_TILDE.md)"
    assert out[3] == "[C](outside.md)"


def test_inline_code_span_link_is_not_rewritten(mirror_mod):
    """Finding 3: a link-like token inside an inline `code span` is literal."""
    line = "real [A](REAL_ONE.md) but code `[B](CODE_ONE.md)` stays"
    got = mirror_mod.canonicalize_line(line)
    assert "[A](real-one.md)" in got
    assert "`[B](CODE_ONE.md)`" in got


# --------------------------------------------------------------------------
# mirror: drift detection + clean pass
# --------------------------------------------------------------------------


def test_clean_mirror_passes(tree):
    mod, pkg, site = tree
    _w(pkg / "API_REFERENCE.md", "# API\n\nsee [G](OTHER_DOC.md)\n")
    _w(site / "api-reference.md", "# API\n\nsee [G](other-doc.md)\n")
    res = mod.Result()
    mod.check_mirror(
        {"package": "API_REFERENCE.md", "site": "api-reference.md"}, pkg, site, res
    )
    assert res.ok


def test_mirror_drift_is_detected(tree):
    mod, pkg, site = tree
    _w(pkg / "API_REFERENCE.md", "# API\n\nthe source truth\n")
    _w(site / "api-reference.md", "# API\n\na hand-edited divergence\n")
    res = mod.Result()
    mod.check_mirror(
        {"package": "API_REFERENCE.md", "site": "api-reference.md"}, pkg, site, res
    )
    assert not res.ok
    assert any("mirror drift" in e for e in res.errors)


def test_mirror_with_fenced_link_example_stays_in_sync(tree):
    """A fenced literal link is uncanonicalized on both sides, so they match."""
    mod, pkg, site = tree
    body = "# API\n\n```md\n[x](FOO_BAR.md)\n```\nprose [y](REAL_DOC.md)\n"
    _w(pkg / "API_REFERENCE.md", body)
    _w(site / "api-reference.md", body.replace("REAL_DOC.md", "real-doc.md"))
    res = mod.Result()
    mod.check_mirror(
        {"package": "API_REFERENCE.md", "site": "api-reference.md"}, pkg, site, res
    )
    assert res.ok, res.errors


def test_mirror_flagged_when_site_is_symlink(tree):
    mod, pkg, site = tree
    _w(pkg / "API_REFERENCE.md", "# API\n")
    (site / "api-reference.md").symlink_to(pkg / "API_REFERENCE.md")
    res = mod.Result()
    mod.check_mirror(
        {"package": "API_REFERENCE.md", "site": "api-reference.md"}, pkg, site, res
    )
    assert not res.ok
    assert any("is a symlink" in e for e in res.errors)


# --------------------------------------------------------------------------
# line_exceptions (Finding 4)
# --------------------------------------------------------------------------


def test_unique_line_exception_is_applied(tree):
    mod, pkg, site = tree
    _w(pkg / "API_REFERENCE.md", "# API\n\nsee [G](BATCH_GUIDE.md) here\n")
    _w(site / "api-reference.md", "# API\n\nsee [G](migration.md) here\n")
    pair = {
        "package": "API_REFERENCE.md",
        "site": "api-reference.md",
        "line_exceptions": [
            {"package": "see [G](BATCH_GUIDE.md) here", "site": "see [G](migration.md) here"}
        ],
    }
    res = mod.Result()
    mod.check_mirror(pair, pkg, site, res)
    assert res.ok, res.errors


def test_ambiguous_line_exception_is_detected(tree):
    """Finding 4: a recurring package line makes the content-match ambiguous."""
    mod, pkg, site = tree
    line = "see [G](BATCH_GUIDE.md) here"
    _w(pkg / "API_REFERENCE.md", f"# API\n\n{line}\n\n{line}\n")
    _w(site / "api-reference.md", "# API\n\nsee [G](migration.md) here\n\nx\n")
    pair = {
        "package": "API_REFERENCE.md",
        "site": "api-reference.md",
        "line_exceptions": [{"package": line, "site": "see [G](migration.md) here"}],
    }
    res = mod.Result()
    mod.check_mirror(pair, pkg, site, res)
    assert not res.ok
    assert any("ambiguous line_exception" in e for e in res.errors)


# --------------------------------------------------------------------------
# symlink / transclude replacement
# --------------------------------------------------------------------------


def test_symlink_replaced_by_real_file_is_detected(tree):
    mod, pkg, site = tree
    _w(pkg / "DEDUP.md", "# Dedup\n")
    _w(site / "dedup.md", "# Dedup (hand copy)\n")  # real file, not a symlink
    res = mod.Result()
    mod.check_symlink({"package": "DEDUP.md", "site": "dedup.md"}, pkg, site, res)
    assert not res.ok
    assert any("not a symlink" in e for e in res.errors)


def test_symlink_wrong_target_is_detected(tree):
    mod, pkg, site = tree
    _w(pkg / "DEDUP.md", "# Dedup\n")
    _w(pkg / "OTHER.md", "# Other\n")
    (site / "dedup.md").symlink_to(pkg / "OTHER.md")  # points at the wrong source
    res = mod.Result()
    mod.check_symlink({"package": "DEDUP.md", "site": "dedup.md"}, pkg, site, res)
    assert not res.ok
    assert any("should point at" in e for e in res.errors)


def test_transclude_replaced_by_handcopy_is_detected(tree):
    mod, pkg, site = tree
    _w(pkg / "GROUNDED_SOURCES.md", "# Grounded\n")
    _w(site / "grounded-sources.md", "# Grounded\n\nhand-authored, no include\n")
    res = mod.Result()
    mod.check_transclude(
        {"package": "GROUNDED_SOURCES.md", "site": "grounded-sources.md"}, pkg, site, res
    )
    assert not res.ok
    assert any("no `--8<--" in e for e in res.errors)


def test_transclude_correct_include_passes(tree):
    mod, pkg, site = tree
    _w(pkg / "GROUNDED_SOURCES.md", "# Grounded\n")
    _w(
        site / "grounded-sources.md",
        '--8<-- "packages/demo/docs/GROUNDED_SOURCES.md"\n',
    )
    res = mod.Result()
    mod.check_transclude(
        {"package": "GROUNDED_SOURCES.md", "site": "grounded-sources.md"}, pkg, site, res
    )
    assert res.ok, res.errors


# --------------------------------------------------------------------------
# completeness gate
# --------------------------------------------------------------------------


def test_unclassified_package_doc_is_detected(tree):
    mod, pkg, site = tree
    _w(pkg / "SURPRISE.md", "# new doc nobody classified\n")
    res = mod.Result()
    mod.check_completeness({}, pkg, site, res)
    assert not res.ok
    assert any("unclassified package doc" in e for e in res.errors)


def test_unclassified_site_doc_is_detected(tree):
    mod, pkg, site = tree
    _w(site / "surprise.md", "# new site page nobody classified\n")
    res = mod.Result()
    mod.check_completeness({}, pkg, site, res)
    assert not res.ok
    assert any("unclassified site doc" in e for e in res.errors)


def test_manifest_reference_to_missing_file_is_detected(tree):
    mod, pkg, site = tree
    # Classified in the manifest entry but neither file exists on disk.
    entry = {"diverge": [{"package": "GHOST.md", "site": "ghost.md"}]}
    res = mod.Result()
    mod.check_completeness(entry, pkg, site, res)
    assert not res.ok
    assert any("references missing package doc" in e for e in res.errors)
    assert any("references missing site doc" in e for e in res.errors)


def test_transclude_subdir_source_exempt_from_completeness(tree):
    """A transclusion may source from a package subdir (e.g. ``guides/events.md``).

    Such a source is not a top-level package doc, so the top-level completeness
    gate must not flag it as a manifest reference to a missing package doc (the
    top-level glob would never list it).
    """
    mod, pkg, site = tree
    (pkg / "guides").mkdir()
    _w(pkg / "guides" / "events.md", "# Events\n")
    _w(site / "events.md", '--8<-- "packages/demo/docs/guides/events.md"\n')
    entry = {"transclude": [{"package": "guides/events.md", "site": "events.md"}]}
    res = mod.Result()
    mod.check_completeness(entry, pkg, site, res)
    assert res.ok, res.errors


def test_double_classification_is_detected(tree):
    mod, pkg, site = tree
    _w(pkg / "X.md", "# x\n")
    _w(site / "x.md", "# x\n")
    entry = {
        "mirror": [{"package": "X.md", "site": "x.md"}],
        "package_only": ["X.md"],
    }
    res = mod.Result()
    mod.check_completeness(entry, pkg, site, res)
    assert not res.ok
    assert any("classified twice" in e for e in res.errors)


# --------------------------------------------------------------------------
# --fix regeneration + idempotence
# --------------------------------------------------------------------------


def test_fix_regenerates_drifted_mirror_and_is_idempotent(tree):
    mod, pkg, site = tree
    _w(pkg / "API_REFERENCE.md", "# API\n\nsee [G](OTHER_DOC.md)\n")
    _w(site / "api-reference.md", "# STALE\n")
    pair = {"package": "API_REFERENCE.md", "site": "api-reference.md"}

    assert mod.fix_mirror(pair, pkg, site) is True  # first run rewrites
    assert (site / "api-reference.md").read_text() == "# API\n\nsee [G](other-doc.md)\n"
    assert mod.fix_mirror(pair, pkg, site) is False  # second run is a no-op

    res = mod.Result()
    mod.check_mirror(pair, pkg, site, res)
    assert res.ok, res.errors


def test_fix_skips_ambiguous_exception_without_corrupting(tree):
    """Finding 4: --fix must not rewrite every occurrence of a recurring line."""
    mod, pkg, site = tree
    line = "see [G](BATCH_GUIDE.md) here"
    original = f"# API\n\n{line}\n\n{line}\n"
    _w(pkg / "API_REFERENCE.md", original)
    _w(site / "api-reference.md", "# STALE\n")
    pair = {
        "package": "API_REFERENCE.md",
        "site": "api-reference.md",
        "line_exceptions": [{"package": line, "site": "see [G](migration.md) here"}],
    }
    assert mod.fix_mirror(pair, pkg, site) is False  # refuses the ambiguous rewrite
    assert (site / "api-reference.md").read_text() == "# STALE\n"  # left untouched


# --------------------------------------------------------------------------
# run() end-to-end
# --------------------------------------------------------------------------


def test_run_returns_zero_on_clean_tree(tree):
    mod, pkg, site = tree
    _w(pkg / "API_REFERENCE.md", "# API\n\nsee [G](OTHER_DOC.md)\n")
    _w(site / "api-reference.md", "# API\n\nsee [G](other-doc.md)\n")
    manifest = _manifest(mirror=[{"package": "API_REFERENCE.md", "site": "api-reference.md"}])
    assert mod.run(manifest, only=None, fix=False) == 0


def test_run_returns_one_on_drift(tree):
    mod, pkg, site = tree
    _w(pkg / "API_REFERENCE.md", "# API\n\ntruth\n")
    _w(site / "api-reference.md", "# API\n\ndrifted\n")
    manifest = _manifest(mirror=[{"package": "API_REFERENCE.md", "site": "api-reference.md"}])
    assert mod.run(manifest, only=None, fix=False) == 1
