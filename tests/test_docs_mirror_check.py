"""Reproduce-first tests for the doc-mirror guard (``bin/docs-mirror-check.py``).

Each test exercises a specific failure mode the guard must catch (drift,
symlink/transclude replacement, unclassified docs, manifest references to
missing files, ambiguous line exceptions) plus the clean-tree pass and the
``--fix`` idempotence the checker promises. Everything runs against a sandbox
tree under ``tmp_path`` with the module's ``ROOT``/``MANIFEST`` globals patched
to that sandbox, so no test touches the repo's real manifest or docs.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests._workspace import load_bin_module

#: What the ``tree`` fixture hands back: the loaded guard module, its
#: sandbox package-docs dir, and its sandbox site-docs dir.
_Tree = tuple[ModuleType, Path, Path]


@pytest.fixture(scope="module")
def mirror_mod() -> ModuleType:
    """Loaded through the shared helper, which reads the source rather than a cache.

    This fixture used to carry its own ``spec_from_file_location`` copy, which
    is the version of the loader that can hand back a previous edit of the
    guard — so a reproduce-first cycle over ``bin/docs-mirror-check.py`` could
    be answered by the code it had just replaced.
    """
    return load_bin_module("docs-mirror-check")


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mirror_mod: ModuleType) -> _Tree:
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


def _manifest(**entry: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "package_dir": "packages/demo/docs",
        "site_dir": "docs/packages/demo",
    }
    base.update(entry)
    return {"packages": {"demo": base}}


# --------------------------------------------------------------------------
# Link canonicalization (the core comparison primitive)
# --------------------------------------------------------------------------


def test_canonicalize_line_rewrites_bare_md_link(mirror_mod: ModuleType) -> None:
    assert mirror_mod.canonicalize_line("see [X](FOO_BAR.md)") == "see [X](foo-bar.md)"


def test_canonicalize_line_preserves_anchor(mirror_mod: ModuleType) -> None:
    got = mirror_mod.canonicalize_line("[X](FOO_BAR.md#the-section)")
    assert got == "[X](foo-bar.md#the-section)"


def test_canonicalize_line_leaves_paths_urls_and_anchors(mirror_mod: ModuleType) -> None:
    for target in ("sub/FOO.md", "https://x.test/FOO.md", "#same-page", "FOO.txt"):
        line = f"[X]({target})"
        assert mirror_mod.canonicalize_line(line) == line


def test_fenced_code_block_link_is_not_rewritten(mirror_mod: ModuleType) -> None:
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


def test_tilde_fence_and_length_tracked(mirror_mod: ModuleType) -> None:
    text = "\n".join(["~~~", "[B](INSIDE_TILDE.md)", "~~~", "[C](OUTSIDE.md)"])
    out = mirror_mod.canonicalize_text(text)
    assert out[1] == "[B](INSIDE_TILDE.md)"
    assert out[3] == "[C](outside.md)"


def test_inline_code_span_link_is_not_rewritten(mirror_mod: ModuleType) -> None:
    """Finding 3: a link-like token inside an inline `code span` is literal."""
    line = "real [A](REAL_ONE.md) but code `[B](CODE_ONE.md)` stays"
    got = mirror_mod.canonicalize_line(line)
    assert "[A](real-one.md)" in got
    assert "`[B](CODE_ONE.md)`" in got


# --------------------------------------------------------------------------
# mirror: drift detection + clean pass
# --------------------------------------------------------------------------


def test_clean_mirror_passes(tree: _Tree) -> None:
    mod, pkg, site = tree
    _w(pkg / "API_REFERENCE.md", "# API\n\nsee [G](OTHER_DOC.md)\n")
    _w(site / "api-reference.md", "# API\n\nsee [G](other-doc.md)\n")
    res = mod.Result()
    mod.check_mirror({"package": "API_REFERENCE.md", "site": "api-reference.md"}, pkg, site, res)
    assert res.ok


def test_mirror_drift_is_detected(tree: _Tree) -> None:
    mod, pkg, site = tree
    _w(pkg / "API_REFERENCE.md", "# API\n\nthe source truth\n")
    _w(site / "api-reference.md", "# API\n\na hand-edited divergence\n")
    res = mod.Result()
    mod.check_mirror({"package": "API_REFERENCE.md", "site": "api-reference.md"}, pkg, site, res)
    assert not res.ok
    assert any("mirror drift" in e for e in res.errors)


def test_mirror_with_fenced_link_example_stays_in_sync(tree: _Tree) -> None:
    """A fenced literal link is uncanonicalized on both sides, so they match."""
    mod, pkg, site = tree
    body = "# API\n\n```md\n[x](FOO_BAR.md)\n```\nprose [y](REAL_DOC.md)\n"
    _w(pkg / "API_REFERENCE.md", body)
    _w(site / "api-reference.md", body.replace("REAL_DOC.md", "real-doc.md"))
    res = mod.Result()
    mod.check_mirror({"package": "API_REFERENCE.md", "site": "api-reference.md"}, pkg, site, res)
    assert res.ok, res.errors


def test_mirror_flagged_when_site_is_symlink(tree: _Tree) -> None:
    mod, pkg, site = tree
    _w(pkg / "API_REFERENCE.md", "# API\n")
    (site / "api-reference.md").symlink_to(pkg / "API_REFERENCE.md")
    res = mod.Result()
    mod.check_mirror({"package": "API_REFERENCE.md", "site": "api-reference.md"}, pkg, site, res)
    assert not res.ok
    assert any("is a symlink" in e for e in res.errors)


# --------------------------------------------------------------------------
# line_exceptions (Finding 4)
# --------------------------------------------------------------------------


def test_unique_line_exception_is_applied(tree: _Tree) -> None:
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


def test_ambiguous_line_exception_is_detected(tree: _Tree) -> None:
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


def test_symlink_replaced_by_real_file_is_detected(tree: _Tree) -> None:
    mod, pkg, site = tree
    _w(pkg / "DEDUP.md", "# Dedup\n")
    _w(site / "dedup.md", "# Dedup (hand copy)\n")  # real file, not a symlink
    res = mod.Result()
    mod.check_symlink({"package": "DEDUP.md", "site": "dedup.md"}, pkg, site, res)
    assert not res.ok
    assert any("not a symlink" in e for e in res.errors)


def test_symlink_wrong_target_is_detected(tree: _Tree) -> None:
    mod, pkg, site = tree
    _w(pkg / "DEDUP.md", "# Dedup\n")
    _w(pkg / "OTHER.md", "# Other\n")
    (site / "dedup.md").symlink_to(pkg / "OTHER.md")  # points at the wrong source
    res = mod.Result()
    mod.check_symlink({"package": "DEDUP.md", "site": "dedup.md"}, pkg, site, res)
    assert not res.ok
    assert any("should point at" in e for e in res.errors)


def test_transclude_replaced_by_handcopy_is_detected(tree: _Tree) -> None:
    mod, pkg, site = tree
    _w(pkg / "GROUNDED_SOURCES.md", "# Grounded\n")
    _w(site / "grounded-sources.md", "# Grounded\n\nhand-authored, no include\n")
    res = mod.Result()
    mod.check_transclude(
        {"package": "GROUNDED_SOURCES.md", "site": "grounded-sources.md"}, pkg, site, res
    )
    assert not res.ok
    assert any("no `--8<--" in e for e in res.errors)


def test_transclude_correct_include_passes(tree: _Tree) -> None:
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


def test_unclassified_package_doc_is_detected(tree: _Tree) -> None:
    mod, pkg, site = tree
    _w(pkg / "SURPRISE.md", "# new doc nobody classified\n")
    res = mod.Result()
    mod.check_completeness({}, pkg, site, res)
    assert not res.ok
    assert any("unclassified package doc" in e for e in res.errors)


def test_unclassified_site_doc_is_detected(tree: _Tree) -> None:
    mod, pkg, site = tree
    _w(site / "surprise.md", "# new site page nobody classified\n")
    res = mod.Result()
    mod.check_completeness({}, pkg, site, res)
    assert not res.ok
    assert any("unclassified site doc" in e for e in res.errors)


def test_manifest_reference_to_missing_file_is_detected(tree: _Tree) -> None:
    mod, pkg, site = tree
    # Classified in the manifest entry but neither file exists on disk.
    entry = {"diverge": [{"package": "GHOST.md", "site": "ghost.md"}]}
    res = mod.Result()
    mod.check_completeness(entry, pkg, site, res)
    assert not res.ok
    assert any("references missing package doc" in e for e in res.errors)
    assert any("references missing site doc" in e for e in res.errors)


def test_transclude_subdir_source_exempt_from_completeness(tree: _Tree) -> None:
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


def test_unpaired_subdir_page_is_classifiable_without_going_recursive(tree: _Tree) -> None:
    """The subdir exemption must cover the unpaired classes, not just the pairs.

    It was written for the paired classes alone, when the unpaired ones
    had no per-class check to fall back on. ``check_unpaired`` is now
    that check, and it matches on the canonicalized basename at any
    depth. Without the exemption a genuinely unpaired subdirectory page
    could not be classified at all under a non-recursive scope -- the
    entry tripped "manifest references missing site doc", because the
    completeness pass globs only top-level basenames -- so the only way
    to record it was to opt the whole package into ``recursive``, and
    the practical answer was to leave it unclassified and unverified.
    """
    mod, pkg, site = tree
    (site / "guides").mkdir()
    _w(site / "guides" / "environment-aware.md", "# env\n")
    res = mod.Result()
    mod.check_completeness({"site_only": ["guides/environment-aware.md"]}, pkg, site, res)
    assert res.ok, res.errors


def test_an_unpaired_subdir_page_is_still_checked_for_a_counterpart(tree: _Tree) -> None:
    """The exemption skips the completeness set, not the class's own check.

    That distinction is the whole justification for it: exempting a file
    from a top-level glob is only safe while something else still asks
    the question the class exists to answer.
    """
    mod, pkg, site = tree
    (site / "guides").mkdir()
    _w(site / "guides" / "environment-aware.md", "# env\n")
    _w(pkg / "ENVIRONMENT_AWARE.md", "# env\n")
    res = mod.Result()
    mod.check_unpaired({"site_only": ["guides/environment-aware.md"]}, pkg, site, res)
    assert not res.ok
    assert any("ENVIRONMENT_AWARE.md" in e for e in res.errors)


def test_an_unpaired_entry_naming_a_file_that_is_not_there_is_reported(tree: _Tree) -> None:
    """The subdir exemption skipped an *existence* check, not just a glob.

    ``check_completeness`` derives "manifest references missing doc"
    from the classified set, so exempting a nested path from that set
    also exempts it from being checked to exist. The paired classes can
    afford it -- each per-class check opens both files -- but
    ``check_unpaired`` only globs the *other* tree for a counterpart,
    and a name with no counterpart is exactly what it is looking for.
    So a stale entry left by a rename, or a typo made while writing
    one, passed all three layers: not in the top-level on-disk set, not
    in the classified set, and no partner to find.
    """
    mod, pkg, site = tree
    (site / "guides").mkdir()
    _w(site / "guides" / "environment-aware.md", "# env\n")
    res = mod.Result()
    # The file is at guides/environment-aware.md; the manifest says otherwise.
    mod.check_unpaired({"site_only": ["guides/environments.md"]}, pkg, site, res)
    assert not res.ok
    assert any("environments.md" in e for e in res.errors), res.errors


def test_an_unpaired_entry_that_does_exist_stays_quiet(tree: _Tree) -> None:
    """The existence check must not fire on the ordinary case."""
    mod, pkg, site = tree
    (site / "guides").mkdir()
    _w(site / "guides" / "environment-aware.md", "# env\n")
    res = mod.Result()
    mod.check_unpaired({"site_only": ["guides/environment-aware.md"]}, pkg, site, res)
    assert res.ok, res.errors


def test_package_only_with_a_site_counterpart_is_detected(tree: _Tree) -> None:
    """``package_only`` must mean unpaired, not "I did not classify this".

    The completeness pass is satisfied by either answer, and there is no
    ``check_package_only``, so naming a doc here opts it out of every
    per-class invariant. That made the class the path of least
    resistance for a pair that was awkward to express -- and the guard
    then reported the package clean while the site copy could drift,
    move or vanish unnoticed.
    """
    mod, pkg, site = tree
    _w(pkg / "USER_GUIDE.md", "# guide\n")
    (site / "guides").mkdir()
    _w(site / "guides" / "user-guide.md", "# guide\n")
    res = mod.Result()
    mod.check_unpaired({"package_only": ["USER_GUIDE.md"]}, pkg, site, res)
    assert not res.ok
    assert any("USER_GUIDE.md" in e and "user-guide.md" in e for e in res.errors)


def test_site_only_with_a_package_counterpart_is_detected(tree: _Tree) -> None:
    """The same hole from the other side, including a nested source."""
    mod, pkg, site = tree
    (pkg / "html").mkdir()
    _w(pkg / "html" / "HTML_CONVERSION.md", "# convert\n")
    _w(site / "html-conversion.md", "# convert\n")
    res = mod.Result()
    mod.check_unpaired({"site_only": ["html-conversion.md"]}, pkg, site, res)
    assert not res.ok
    assert any("html-conversion.md" in e and "html/HTML_CONVERSION.md" in e for e in res.errors)


def test_genuinely_unpaired_docs_pass(tree: _Tree) -> None:
    """A doc with no counterpart anywhere is what the class is for."""
    mod, pkg, site = tree
    _w(pkg / "BENCHMARKING.md", "# bench\n")
    _w(site / "index.md", "# landing\n")
    res = mod.Result()
    mod.check_unpaired(
        {"package_only": ["BENCHMARKING.md"], "site_only": ["index.md"]}, pkg, site, res
    )
    assert res.ok, res.errors


def test_a_classified_pair_is_not_reported_by_the_unpaired_check(tree: _Tree) -> None:
    """Only the two unpaired classes are in scope; a real pair is fine."""
    mod, pkg, site = tree
    _w(pkg / "TOOLS.md", "# tools\n")
    (site / "guides").mkdir()
    _w(site / "guides" / "tools.md", "# tools\n")
    res = mod.Result()
    mod.check_unpaired(
        {"symlink": [{"package": "TOOLS.md", "site": "guides/tools.md"}]}, pkg, site, res
    )
    assert res.ok, res.errors


def test_double_classification_is_detected(tree: _Tree) -> None:
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


def test_fix_regenerates_drifted_mirror_and_is_idempotent(tree: _Tree) -> None:
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


def test_fix_skips_ambiguous_exception_without_corrupting(tree: _Tree) -> None:
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


def test_run_returns_zero_on_clean_tree(tree: _Tree) -> None:
    """Clean now includes every relative link resolving, so the fixture links.

    Two earlier versions of this fixture are worth recording, because each was
    clean under the rule of its day and is a failure under the next one.
    ``API_REFERENCE.md`` exercised the canonicaliser through the mirror
    comparison, and stopped being clean when ``run`` began requiring
    lower-hyphen package docs. It became ``OTHER_DOC.md`` against
    ``other-doc.md`` -- the same fold moved into the link text -- which was
    clean only while an unresolvable link was reported rather than failed. It
    is one spelling now, present in both trees, and the fold it was testing
    belongs to the unit tests above.
    """
    mod, pkg, site = tree
    _w(pkg / "api-reference.md", "# API\n\nsee [G](other-doc.md)\n")
    _w(site / "api-reference.md", "# API\n\nsee [G](other-doc.md)\n")
    _w(pkg / "other-doc.md", "# O\n")
    _w(site / "other-doc.md", "# O\n")
    manifest = _manifest(
        mirror=[
            {"package": "api-reference.md", "site": "api-reference.md"},
            {"package": "other-doc.md", "site": "other-doc.md"},
        ]
    )
    assert mod.run(manifest, only=None, fix=False) == 0


def test_run_returns_one_on_drift(tree: _Tree) -> None:
    mod, pkg, site = tree
    _w(pkg / "api-reference.md", "# API\n\ntruth\n")
    _w(site / "api-reference.md", "# API\n\ndrifted\n")
    manifest = _manifest(mirror=[{"package": "api-reference.md", "site": "api-reference.md"}])
    assert mod.run(manifest, only=None, fix=False) == 1


def test_run_fails_on_a_link_no_rename_can_reach(
    tree: _Tree, capsys: pytest.CaptureFixture[str]
) -> None:
    """The population that was reported is now failed, end to end through run().

    Both halves of the link work are enforced here. The naming half always was;
    this half waited on a policy for what a package doc may link to when the
    target is in another package, is site-native, or is gone. There is one now,
    so the informational branch is gone and the two populations differ only in
    the remedy their message names.
    """
    mod, pkg, _site = tree
    _w(pkg / "guide.md", "# G\n\nsee [X](../elsewhere/nothing.md)\n")
    manifest = _manifest(package_only=["guide.md"])
    assert mod.run(manifest, only=None, fix=False) != 0
    out = capsys.readouterr().out
    assert "link resolution" in out
    assert "../elsewhere/nothing.md" in out


def test_absolute_site_url_is_not_a_relative_link(tree: _Tree) -> None:
    """The policy's targets are exempt by construction, not by an allowlist.

    A characterisation test, not a reproduce-first one: it passes today and is
    written to keep passing. The whole policy rests on it -- an absolute site
    URL is simply not a relative ``.md`` link, so nothing has to declare that a
    cross-tree link was deliberate. If ``link_targets`` ever started resolving
    ``https://`` targets, every one of those decisions would turn into a
    finding at once, and this is what turns that into a red test instead.
    """
    mod, pkg, site = tree
    _w(
        pkg / "guide.md",
        "# G\n\nsee [X](https://kbs-labs.github.io/dataknobs/packages/common/packs/)\n"
        "and [Y](https://kbs-labs.github.io/dataknobs/api/reference/bots/#dataknobs_bots.DynaBot)\n",
    )
    assert mod.link_targets((pkg / "guide.md").read_text()) == []
    res = mod.Result()
    mod.check_link_resolution({"package_only": ["guide.md"]}, pkg, site, res)
    assert res.ok, res.errors


# --------------------------------------------------------------------------
# one document, one name  (name parity + lower-hyphen spelling)
# --------------------------------------------------------------------------


def test_paired_doc_spelled_two_ways_is_detected(tree: _Tree) -> None:
    """Reproduces the shape that made 89 package-tree links unresolvable.

    When the package tree spells a doc one way and the site tree another, a
    bare link to it is correct in at most one of them -- and whichever tree
    loses, nothing was checking it.
    """
    mod, pkg, site = tree
    _w(pkg / "user-guide.md", "# G\n")
    _w(site / "userguide.md", "# G\n")
    res = mod.Result()
    mod.check_name_parity({"mirror": [{"package": "user-guide.md", "site": "userguide.md"}]}, res)
    assert not res.ok
    assert any("name parity" in e for e in res.errors)


def test_paired_doc_nested_differently_but_named_alike_passes(tree: _Tree) -> None:
    """Only the basename is compared; the two trees nest some docs differently.

    ``guides/tools.md`` against ``tools.md`` is not a naming failure and no
    rename fixes it, so name parity must not claim it.
    """
    mod, _pkg, _site = tree
    res = mod.Result()
    mod.check_name_parity({"symlink": [{"package": "tools.md", "site": "guides/tools.md"}]}, res)
    assert res.ok, res.errors


def test_diverge_pair_may_be_spelled_two_ways(tree: _Tree) -> None:
    """``diverge`` records two genuinely different documents, so parity cannot bind it.

    Requiring a shared name would contradict the classification -- and two real
    pairs rely on this (``multi-tenant.md`` <-> ``guides/bot-manager.md`` and
    ``environment-variable-substitution.md`` <-> ``environment-variables.md``).
    """
    mod, _pkg, _site = tree
    res = mod.Result()
    mod.check_name_parity(
        {"diverge": [{"package": "multi-tenant.md", "site": "guides/bot-manager.md"}]}, res
    )
    assert res.ok, res.errors


def test_upper_snake_package_doc_is_detected(tree: _Tree) -> None:
    """The recurrence guard: a *new* doc in the old convention, paired or not.

    Name parity cannot see this one -- a new doc need not be paired with
    anything -- which is why the spelling rule is a separate check rather than
    a property of the pair.
    """
    mod, pkg, _site = tree
    _w(pkg / "TOOL_CONTEXT.md", "# T\n")
    res = mod.Result()
    mod.check_doc_spelling(pkg, res)
    assert not res.ok
    assert any("doc spelling" in e and "tool-context.md" in e for e in res.errors)


def test_nested_package_doc_is_checked_even_when_not_recursive(tree: _Tree) -> None:
    """``recursive`` scopes which docs must be *classified*, not how they are spelled.

    A doc can be spelled wrong without being classified at all, which is exactly
    the case worth catching, so this check reads the tree rather than the
    manifest.
    """
    mod, pkg, _site = tree
    (pkg / "guides").mkdir()
    _w(pkg / "guides" / "MULTI_TENANT.md", "# M\n")
    res = mod.Result()
    mod.check_doc_spelling(pkg, res)
    assert not res.ok
    assert any("multi-tenant.md" in e for e in res.errors)


def test_readme_is_exempt_from_the_spelling_rule(tree: _Tree) -> None:
    """GitHub renders ``README.md`` as a directory index; ``readme.md`` does not."""
    mod, pkg, _site = tree
    _w(pkg / "README.md", "# index\n")
    res = mod.Result()
    mod.check_doc_spelling(pkg, res)
    assert res.ok, res.errors


def test_conforming_package_tree_is_quiet(tree: _Tree) -> None:
    mod, pkg, _site = tree
    _w(pkg / "user-guide.md", "# G\n")
    _w(pkg / "05.updated-plan.md", "# P\n")
    res = mod.Result()
    mod.check_doc_spelling(pkg, res)
    assert res.ok, res.errors


# --------------------------------------------------------------------------
# link resolution
# --------------------------------------------------------------------------


def test_link_broken_only_by_spelling_is_detected(tree: _Tree) -> None:
    """The defect PR-scale renaming drained: right link text, wrong filename.

    Nothing checked link resolution before this, which is how 89 of them
    accumulated while every guard reported green.
    """
    mod, pkg, site = tree
    _w(pkg / "architecture.md", "# A\n\nsee [C](configuration.md)\n")
    _w(pkg / "CONFIGURATION.md", "# C\n")
    res = mod.Result()
    mod.check_link_resolution(
        {"package_only": ["architecture.md", "CONFIGURATION.md"]}, pkg, site, res
    )
    assert not res.ok
    assert any("link spelling" in e and "CONFIGURATION.md" in e for e in res.errors)


def test_spelling_break_is_caught_where_path_exists_would_lie(tree: _Tree) -> None:
    """The regression that hid the whole population: case-insensitive existence.

    ``Path.exists()`` answers *yes* on a case-insensitive checkout when the link
    says ``configuration.md`` and only ``CONFIGURATION.md`` is on disk -- so the
    obvious implementation of this check would have reported green on the very
    tree that motivated it, on the machine it was written on.

    Both assertions hold on either kind of filesystem. The naive verdict is
    captured rather than asserted, because *it* is what differs by platform;
    that is the point.
    """
    mod, pkg, site = tree
    _w(pkg / "architecture.md", "# A\n\nsee [C](configuration.md)\n")
    _w(pkg / "CONFIGURATION.md", "# C\n")

    naive = (pkg / "configuration.md").exists()  # True on macOS, False on CI
    assert not mod._exists_cs(pkg / "configuration.md"), (
        f"case-sensitive existence must not follow Path.exists() (which said {naive})"
    )

    res = mod.Result()
    mod.check_link_resolution(
        {"package_only": ["architecture.md", "CONFIGURATION.md"]}, pkg, site, res
    )
    assert not res.ok, f"verdict must not depend on the filesystem (Path.exists said {naive})"


def test_link_with_no_target_under_any_spelling_fails(tree: _Tree) -> None:
    """A target absent under every spelling fails, now that a policy exists.

    This population used to be counted and printed, because no rename reaches
    it and answering it needed a decision: the two trees nest the same document
    differently, or the target is site-native, or it is gone. The decision was
    made -- link the published page by its absolute site URL, publish the
    target into the package tree, or name it in prose -- so the check no longer
    distinguishes the two populations and simply requires every relative link
    to resolve in every tree its document is served from.
    """
    mod, pkg, site = tree
    _w(pkg / "guide.md", "# G\n\nsee [X](../api/reference.md)\n")
    res = mod.Result()
    mod.check_link_resolution({"package_only": ["guide.md"]}, pkg, site, res)
    assert not res.ok
    assert any("link resolution" in e and "../api/reference.md" in e for e in res.errors)


def test_symlink_doc_link_is_resolved_against_both_trees(tree: _Tree) -> None:
    """One file served at two paths must resolve from both of them.

    A symlinked page is the same bytes under a second path, so a sibling link
    in it is read from the site directory too -- and a sibling that exists only
    in the package tree leaves the site copy pointing at nothing.
    """
    mod, pkg, site = tree
    _w(pkg / "tools.md", "# T\n\nsee [C](context.md)\n")
    _w(pkg / "context.md", "# C\n")
    (site / "tools.md").symlink_to(pkg / "tools.md")
    res = mod.Result()
    mod.check_link_resolution(
        {"symlink": [{"package": "tools.md", "site": "tools.md"}]}, pkg, site, res
    )
    assert not res.ok
    assert any("docs/packages/demo" in e for e in res.errors), res.errors


def test_link_in_a_fence_or_code_span_is_not_resolved(tree: _Tree) -> None:
    """Example text is not a link, so it cannot be a broken one.

    Same two primitives the canonicaliser uses -- a doc showing markdown syntax
    would otherwise report a finding for every sample it contains.
    """
    mod, pkg, site = tree
    _w(
        pkg / "guide.md",
        "# G\n\n```md\n[a](NOT_A_REAL_DOC.md)\n```\n\nand `[b](ALSO_NOT.md)` inline\n",
    )
    res = mod.Result()
    mod.check_link_resolution({"package_only": ["guide.md"]}, pkg, site, res)
    assert res.ok, res.errors


def test_urls_anchors_and_non_md_targets_are_not_resolved(tree: _Tree) -> None:
    mod, pkg, site = tree
    _w(
        pkg / "guide.md",
        "# G\n\n[a](https://x.test/FOO.md) [b](#section) [c](script.py) [d](/abs/x.md)\n",
    )
    res = mod.Result()
    mod.check_link_resolution({"package_only": ["guide.md"]}, pkg, site, res)
    assert res.ok, res.errors


def test_resolving_links_pass_quietly(tree: _Tree) -> None:
    mod, pkg, site = tree
    _w(pkg / "guide.md", "# G\n\nsee [C](configuration.md) and [S](sub/deep.md)\n")
    _w(pkg / "configuration.md", "# C\n")
    (pkg / "sub").mkdir()
    _w(pkg / "sub" / "deep.md", "# D\n")
    res = mod.Result()
    mod.check_link_resolution({"package_only": ["guide.md", "configuration.md"]}, pkg, site, res)
    assert res.ok, res.errors
