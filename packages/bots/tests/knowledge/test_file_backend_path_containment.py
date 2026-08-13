"""A domain id or resource path must not address outside the backend's base.

:class:`FileKnowledgeBackend` turns two caller-supplied identifiers into
filesystem locations — ``domain_id`` via ``_kb_path`` and a resource
``path`` via ``_file_path`` — and both reach destructive sinks:
``mkdir(parents=True)``, ``shutil.rmtree``, ``unlink``, and an atomic
content write.

Each test asserts on the **sink**, not on the composed path: that the
call is refused *and* that nothing appeared, was read, or was removed
outside the base. A path assertion alone cannot tell "composed wrong"
from "composed wrong and acted on it", and it is the acting that
matters here — before the guard, ``delete_kb("../sacrificial")``
returned ``True`` having removed the tree.

Both escape spellings are covered per site, because a guard that catches
one is not a guard: a ``..`` segment, and an **absolute** part, which
discards the base outright (``Path("/base") / "/etc/passwd"`` is
``/etc/passwd``). A contained name that merely *looks* dangerous — a
subdirectory, or an interior ``a/../b`` — must still work **where the
name is a location**: that is a resource ``path`` inside ``content/``,
whose whole purpose is to nest.

It is not true of ``domain_id``, and several tests here used to assert
that it was. A domain occupies one *slot* of the layout rather than
naming a location within it, so a separator inside it reaches the
literal segments the layout puts around it — a different knowledge
base's metadata document, or its whole subtree under a prefix delete.
Containment cannot see that: the name never leaves the base. Those
tests are rewritten in place rather than deleted, each stating what it
used to assert and why the invariant was the wrong one.

Two further shapes are covered below, and neither is "an identifier
escaped the base":

* **the boundary is not always the base.** With a tenant context the
  composition has two hops, and containment has to be judged against the
  inner one — a ``domain_id`` that walks sideways into a sibling tenant
  never leaves ``base_path``, so every base-anchored assertion passes
  while tenant isolation is gone.
* **a guard does not cover what is appended after it.** ``_snapshot_file``
  and ``key_pattern`` each build on ``domain_id``'s guard and then add a
  second untrusted component, which is why "every helper routes through a
  guarded method" was true and still left two holes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dataknobs_common.paths import PathEscapeError, SegmentEscapeError
from dataknobs_common.tenancy import BoundTenantContext

from dataknobs_bots.knowledge.storage.file import FileKnowledgeBackend
from dataknobs_bots.knowledge.storage.key_layout import KnowledgeKeyKind


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """The backend's base directory, with a sibling it must never reach."""
    inside = tmp_path / "base"
    inside.mkdir()
    (tmp_path / "outside").mkdir()
    return inside


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A sibling of the base, standing in for anything else on the volume."""
    return tmp_path / "outside"


async def _backend(base: Path) -> FileKnowledgeBackend:
    backend = FileKnowledgeBackend(base_path=base)
    await backend.initialize()
    return backend


# --- domain_id -> _kb_path (S1) ------------------------------------------


async def test_create_kb_refuses_a_domain_id_that_walks_out(base: Path, outside: Path) -> None:
    """``mkdir(parents=True)`` must not run outside the base."""
    backend = await _backend(base)

    with pytest.raises(ValueError):
        await backend.create_kb("../outside/pwned")

    assert not (outside / "pwned").exists()


async def test_create_kb_refuses_an_absolute_domain_id(base: Path, outside: Path) -> None:
    """An absolute part discards the base; rejecting ``..`` alone misses it."""
    backend = await _backend(base)
    target = outside / "pwned-absolute"

    with pytest.raises(ValueError):
        await backend.create_kb(str(target))

    assert not target.exists()


async def test_delete_kb_refuses_a_domain_id_that_walks_out(base: Path, tmp_path: Path) -> None:
    """``shutil.rmtree`` is the sharpest sink in the class."""
    backend = await _backend(base)
    sacrificial = tmp_path / "outside" / "sacrificial"
    sacrificial.mkdir()
    (sacrificial / "precious.txt").write_text("keep me")

    with pytest.raises(ValueError):
        await backend.delete_kb("../outside/sacrificial")

    assert (sacrificial / "precious.txt").read_text() == "keep me"


async def test_get_info_refuses_a_domain_id_that_walks_out(base: Path) -> None:
    """A read-shaped method raises rather than answering about outside."""
    backend = await _backend(base)

    with pytest.raises(ValueError):
        await backend.get_info("../outside")


async def test_a_domain_id_naming_a_subdirectory_is_now_refused(base: Path) -> None:
    """BEHAVIOUR CHANGE, and the old assertion had the wrong invariant.

    This test read ``create_kb("team/alpha")`` succeeding as proof that
    "containment is not a ``/``-rejecting character class" — true of
    containment, and containment was the wrong question. A nested domain
    stays inside the base, so the guard passed it, while the path it
    composes lands in the layout's own slots: ``acme/content`` addresses
    exactly what an ordinary content file named ``_metadata.json`` under
    ``acme`` addresses, and ``delete_kb("acme")`` removes the whole of
    it. Nor was the nesting usable — ``list_kbs`` enumerates one level,
    so the KB this test created could never be found again.

    So the rule is one segment, and it is checked before containment.
    Containment still applies to what is genuinely a location: a
    resource ``path`` inside ``content/``, where nesting IS the point,
    is unaffected — see the ``_file_path`` tests below.
    """
    backend = await _backend(base)

    with pytest.raises(SegmentEscapeError, match="domain_id"):
        await backend.create_kb("team/alpha")

    assert not (base / "team").exists()


async def test_an_interior_parent_ref_can_no_longer_alias_one_directory(
    base: Path,
) -> None:
    """The aliasing this test characterised is gone, not merely filed.

    It used to record that ``team/../beta`` was *contained* — the path
    collapses to ``beta`` and never leaves the base — while the
    identifier was not canonicalized, so one directory answered to two
    names and reported only one of them. That was left as a separate
    finding on the grounds that containment had done its job.

    The segment rule removes the shape entirely: an identifier with no
    separator cannot collapse, so there is no second spelling for a
    directory to answer to.
    """
    backend = await _backend(base)

    with pytest.raises(SegmentEscapeError, match="domain_id"):
        await backend.create_kb("team/../beta")

    assert not (base / "beta").exists()
    assert await backend.list_kbs() == []


# --- resource path -> _file_path (S2) ------------------------------------


async def test_put_file_refuses_a_path_that_walks_out(base: Path, outside: Path) -> None:
    """The atomic content write must not land outside the base."""
    backend = await _backend(base)
    await backend.create_kb("dom")

    with pytest.raises(ValueError):
        await backend.put_file("dom", "../../../outside/pwned.md", b"owned")

    assert not (outside / "pwned.md").exists()


async def test_put_file_refuses_an_absolute_path(base: Path, outside: Path) -> None:
    backend = await _backend(base)
    await backend.create_kb("dom")
    target = outside / "pwned-absolute.md"

    with pytest.raises(ValueError):
        await backend.put_file("dom", str(target), b"owned")

    assert not target.exists()


async def test_get_file_refuses_to_read_outside_the_base(base: Path, outside: Path) -> None:
    """Before the guard this returned the file's bytes, not ``None``."""
    backend = await _backend(base)
    await backend.create_kb("dom")
    (outside / "secret.txt").write_text("SECRET")

    with pytest.raises(ValueError):
        await backend.get_file("dom", "../../../outside/secret.txt")


async def test_delete_file_refuses_to_unlink_outside_the_base(base: Path, outside: Path) -> None:
    backend = await _backend(base)
    await backend.create_kb("dom")
    victim = outside / "victim.txt"
    victim.write_text("keep me")

    with pytest.raises(ValueError):
        await backend.delete_file("dom", "../../../outside/victim.txt")

    assert victim.read_text() == "keep me"


async def test_file_exists_refuses_a_path_that_walks_out(base: Path, outside: Path) -> None:
    """An escaping name is never a legitimate "absent" answer."""
    backend = await _backend(base)
    await backend.create_kb("dom")
    (outside / "probe.txt").write_text("x")

    with pytest.raises(ValueError):
        await backend.file_exists("dom", "../../../outside/probe.txt")


async def test_stream_file_refuses_a_path_that_walks_out(base: Path, outside: Path) -> None:
    backend = await _backend(base)
    await backend.create_kb("dom")
    (outside / "streamed.txt").write_text("SECRET")

    with pytest.raises(ValueError):
        await backend.stream_file("dom", "../../../outside/streamed.txt")


async def test_a_resource_path_in_a_subdirectory_still_works(base: Path) -> None:
    """The content tree is explicitly nested; ``subdir/file`` is normal."""
    backend = await _backend(base)
    await backend.create_kb("dom")

    await backend.put_file("dom", "subdir/nested.md", b"fine")

    assert await backend.get_file("dom", "subdir/nested.md") == b"fine"


# --- the tenant subtree is the boundary, not the base --------------------


async def test_a_domain_id_must_not_cross_into_another_tenants_state(base: Path) -> None:
    """Bug: containment was judged against ``base_path``, but with a tenant
    context the boundary is ``base_path/{state_prefix}``.

    A ``domain_id`` that walks *sideways* satisfies the outer bound and
    lands in a sibling tenant's subtree — no ``..`` escapes the base, so
    every test that checks only "did it leave base_path" passes. Tenant
    ``acme`` could read tenant ``bob``'s state-version token through the
    public ``get_state_version``, which is the isolation the state prefix
    exists to provide.
    """
    backend = await _backend(base)
    acme = BoundTenantContext(tenant_id="acme", domain_id="proj")
    bob = BoundTenantContext(tenant_id="bob", domain_id="proj")

    await backend.create_kb("proj")
    await backend.set_ingestion_status("proj", "ready", ctx=bob)
    bobs_token = await backend.get_state_version("proj", ctx=bob)
    assert bobs_token is not None

    # base/_scoped/tenants/acme/_state/../../bob/_state/proj
    # == base/_scoped/tenants/bob/_state/proj
    #
    # Refused by the segment rule now, which runs first and rejects the
    # name for carrying any structure at all. The tenant-subtree bound
    # below it is unchanged and still the thing that would catch this if
    # the segment rule were ever relaxed — the two answer different
    # questions and neither subsumes the other.
    with pytest.raises(SegmentEscapeError, match="domain_id"):
        await backend.get_state_version("../../bob/_state/proj", ctx=acme)

    # And acme's own view is unchanged by the attempt.
    assert await backend.get_state_version("proj", ctx=acme) is None


async def test_each_tenant_still_reaches_its_own_state(base: Path) -> None:
    """The tighter bound must not break the layout it is protecting."""
    backend = await _backend(base)
    acme = BoundTenantContext(tenant_id="acme", domain_id="proj")
    bob = BoundTenantContext(tenant_id="bob", domain_id="proj")

    await backend.create_kb("proj")
    await backend.set_ingestion_status("proj", "ready", ctx=acme)
    await backend.set_ingestion_status("proj", "pending", ctx=bob)

    assert (base / "_scoped" / "tenants" / "acme" / "_state" / "proj").is_dir()
    assert (base / "_scoped" / "tenants" / "bob" / "_state" / "proj").is_dir()
    acme_token = await backend.get_state_version("proj", ctx=acme)
    bob_token = await backend.get_state_version("proj", ctx=bob)
    assert acme_token is not None and bob_token is not None
    assert acme_token != bob_token


async def test_a_tenant_reaches_its_own_state_under_a_plain_domain(base: Path) -> None:
    """The tenant subtree still works; the domain inside it is one segment.

    This replaces a test asserting that a tenant could nest its domain
    (``team/alpha`` under ``_scoped/tenants/acme/_state/``). Nesting is refused
    now for the reason the domain tests above give, and what actually
    needed pinning here — that the tenant's state lands under its own
    prefix and nowhere else — is pinned without it.
    """
    backend = await _backend(base)
    ctx = BoundTenantContext(tenant_id="acme", domain_id="alpha")

    await backend.create_kb("alpha")
    await backend.set_ingestion_status("alpha", "ready", ctx=ctx)

    assert (base / "_scoped" / "tenants" / "acme" / "_state" / "alpha").is_dir()

    with pytest.raises(SegmentEscapeError, match="domain_id"):
        await backend.set_ingestion_status("team/alpha", "ready", ctx=ctx)


# --- components appended AFTER a guard are not covered by it -------------


async def test_a_snapshot_version_must_not_address_outside_the_snapshot_dir(
    base: Path, outside: Path
) -> None:
    """Bug: ``_snapshot_file`` routed through the guarded ``_kb_path`` and
    then appended ``f"{version}.json"`` on top of the approved path.

    ``version`` arrives from the public ``list_changes_since`` as a token
    the caller persisted and handed back, so routing through a guard that
    a sibling component is appended after covers ``domain_id`` and not
    this. The read returns the file's top-level keys to the caller as
    ``ChangeSet.deleted``.
    """
    backend = await _backend(base)
    await backend.create_kb("dom")
    secret = outside / "secret.json"
    secret.write_text('{"private-key-name": "x"}')

    with pytest.raises(PathEscapeError, match="snapshot version"):
        await backend.list_changes_since("dom", "../../../outside/secret")


async def test_key_pattern_must_not_build_a_glob_outside_the_base(base: Path) -> None:
    """Bug: ``key_pattern`` composed ``domain_id`` into an f-string over
    ``str(self._base_path)``, reaching neither guard.

    Its output is handed to ``Path.glob`` or an inotify watch, so an
    escaping domain installs a watch over a tree the deployment did not
    choose — a leak with no filesystem call on this class at all.
    """
    backend = await _backend(base)

    with pytest.raises(PathEscapeError, match="domain_id"):
        backend.key_pattern(KnowledgeKeyKind.CONTENT, domain_id="../../elsewhere")


async def test_key_pattern_still_serves_its_legitimate_shapes(base: Path) -> None:
    """The wildcard form and a plain domain survive; a nested one does not.

    The nested case this used to assert (``team/alpha``) built a watch
    over another knowledge base's tree — the same defect as the key
    composition, one layer out, since a pattern is what a subscription
    is installed from.

    ``domain_id=None`` is the all-domains spelling and is not a name, so
    it is untouched. An *empty* domain is refused rather than read as
    the same thing: widening a watch to every domain because a name came
    back empty is precisely the failure being guarded.
    """
    backend = await _backend(base)

    assert backend.key_pattern(KnowledgeKeyKind.CONTENT) == f"{base}/*/content/**"
    assert (
        backend.key_pattern(KnowledgeKeyKind.CONTENT, domain_id="alpha")
        == f"{base}/alpha/content/**"
    )
    with pytest.raises(SegmentEscapeError, match="domain_id"):
        backend.key_pattern(KnowledgeKeyKind.CONTENT, domain_id="team/alpha")
    with pytest.raises(SegmentEscapeError, match="domain_id"):
        backend.key_pattern(KnowledgeKeyKind.CONTENT, domain_id="")


# --- the refusal is one catchable type ------------------------------------


async def test_a_refusal_is_distinguishable_from_any_other_value_error(base: Path) -> None:
    """The guards raised bare ``ValueError``, which a consumer cannot tell
    from an unrelated one on the same call — ``pydantic.ValidationError``
    subclasses it, and this repo's own test suite was caught by exactly
    that ambiguity. ``PathEscapeError`` narrows it while staying a
    ``ValueError``, so catching the old type still works.
    """
    backend = await _backend(base)
    await backend.create_kb("dom")

    with pytest.raises(PathEscapeError):
        await backend.get_file("dom", "../../etc/passwd")
    # Still a ValueError: narrowing, not a breaking change.
    with pytest.raises(ValueError):
        await backend.get_file("dom", "../../etc/passwd")
