"""Reproduce-first tests for the doc-mirror guard (``bin/docs-mirror-check.py``).

Each test exercises a specific failure mode the guard must catch (drift,
symlink/transclude replacement, unclassified docs, manifest references to
missing files, a retired or misspelled class key) plus the clean-tree pass.
Everything runs against a sandbox
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
        "transclude": [{"package": "X.md", "site": "x.md"}],
        "package_only": ["X.md"],
    }
    res = mod.Result()
    mod.check_completeness(entry, pkg, site, res)
    assert not res.ok
    assert any("classified twice" in e for e in res.errors)


# --------------------------------------------------------------------------
# End-to-end through run()
# --------------------------------------------------------------------------


def test_run_returns_zero_on_clean_tree(tree: _Tree) -> None:
    """Clean now includes every relative link resolving, so the fixture links.

    Three earlier versions of this fixture are worth recording, because each
    was clean under the rule of its day and is a failure under the next one.
    ``API_REFERENCE.md`` exercised the canonicaliser through the mirror
    comparison, and stopped being clean when ``run`` began requiring
    lower-hyphen package docs. It became ``OTHER_DOC.md`` against
    ``other-doc.md`` -- the same fold moved into the link text -- which was
    clean only while an unresolvable link was reported rather than failed. It
    was then one spelling present in both trees as two hand-authored copies,
    which was clean only while ``mirror`` existed to classify them.

    It is a ``transclude`` now, which is the point: the site page is not a copy
    to keep in agreement, it is the same text at a second path. The sandbox
    still needs the site file on disk, because ``check_completeness`` reads the
    directory rather than the manifest -- but what it contains is an include,
    and nothing compares it to anything.
    """
    mod, pkg, site = tree
    _w(pkg / "api-reference.md", "# API\n\nsee [G](other-doc.md)\n")
    _w(site / "api-reference.md", '--8<-- "packages/demo/docs/api-reference.md"\n')
    _w(pkg / "other-doc.md", "# O\n")
    _w(site / "other-doc.md", '--8<-- "packages/demo/docs/other-doc.md"\n')
    manifest = _manifest(
        transclude=[
            {"package": "api-reference.md", "site": "api-reference.md"},
            {"package": "other-doc.md", "site": "other-doc.md"},
        ]
    )
    assert mod.run(manifest, only=None) == 0


def test_a_retired_class_key_is_rejected_rather_than_ignored(tree: _Tree) -> None:
    """An entry key the guard does not know is refused, not skipped.

    Reproduce-first for the ``mirror`` retirement, and deliberately general
    beyond it: the manifest had no key validation at all. The class names were
    enumerated at five call sites and anything else simply never matched, so a
    ``mirror`` entry written after the class was retired read as no
    classification at all.

    The fixture is nested on purpose, because that is where the omission is
    *silent* rather than merely misreported. A top-level doc nobody classified
    fails the completeness pass -- the right verdict for the wrong reason. A
    nested one in a non-recursive package is not part of the completeness set,
    so a retired entry there leaves both files verified by nothing and the
    package reporting clean.
    """
    mod, pkg, site = tree
    (pkg / "nested").mkdir()
    (site / "nested").mkdir()
    _w(pkg / "nested" / "old.md", "# Old\n")
    _w(site / "nested" / "old.md", "# Old\n")
    manifest = _manifest(mirror=[{"package": "nested/old.md", "site": "nested/old.md"}])
    assert mod.run(manifest, only=None) == 1


def test_an_underscore_key_is_commentary_and_is_exempt(tree: _Tree) -> None:
    """The manifest's own comment convention, which the check found the hard way.

    JSON has no comments, so ``_note`` and ``_schema`` at the top level are how
    the manifest carries its reasoning -- and ``structures`` and ``utils`` each
    carry a per-package ``_note`` saying why the package is entirely
    ``site_only``. The first run of :func:`check_known_classes` against the real
    tree refused both of those. Exempting the prefix is not a hole in the rule;
    refusing it would have been an instruction to delete the explanation.
    """
    mod, pkg, site = tree
    _w(pkg / "guide.md", "# G\n")
    _w(site / "guide.md", '--8<-- "packages/demo/docs/guide.md"\n')
    manifest = _manifest(
        _note="why this package looks the way it does",
        transclude=[{"package": "guide.md", "site": "guide.md"}],
    )
    assert mod.run(manifest, only=None) == 0


def test_a_misspelled_class_key_is_rejected(tree: _Tree) -> None:
    """The same refusal, for the typo that was always possible.

    ``symlnk`` never matched any of the five enumerations either, so a pair
    recorded under it was opted out of every invariant while the manifest
    looked complete. Nested for the reason the test above gives.
    """
    mod, pkg, site = tree
    (pkg / "nested").mkdir()
    (site / "nested").mkdir()
    _w(pkg / "nested" / "guide.md", "# G\n")
    _w(site / "nested" / "guide.md", "# G\n")
    manifest = _manifest(symlnk=[{"package": "nested/guide.md", "site": "nested/guide.md"}])
    assert mod.run(manifest, only=None) == 1


def test_run_returns_one_when_a_transclude_became_a_hand_copy(tree: _Tree) -> None:
    """The end-to-end failure that replaced end-to-end drift.

    This was ``test_run_returns_one_on_drift``, and drift is not expressible
    any more: a ``transclude`` pair is one text at two paths, so there is no
    second copy to diverge. What *is* still possible is someone pasting the
    source's content over the include -- recreating the hand copy the retired
    class used to legitimise -- and that is the failure this asserts, through
    ``run`` rather than through ``check_transclude`` alone.
    """
    mod, pkg, site = tree
    _w(pkg / "api-reference.md", "# API\n\ntruth\n")
    _w(site / "api-reference.md", "# API\n\ntruth\n")
    manifest = _manifest(transclude=[{"package": "api-reference.md", "site": "api-reference.md"}])
    assert mod.run(manifest, only=None) == 1


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
    assert mod.run(manifest, only=None) != 0
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
    mod.check_name_parity(
        {"transclude": [{"package": "user-guide.md", "site": "userguide.md"}]}, res
    )
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
