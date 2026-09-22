# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A width stated once is the width both halves get.

``RAGKnowledgeBase`` and ``VectorMemory`` each build **both** a vector store
and an embedding provider, and each used to take the width twice: once for the
store (``vector_store.dimensions`` / ``dimension``) and once for the embedder
(``dimensions``, or ``dimensions`` inside the nested ``embedding`` dict). The
two were never compared, and ``_embed_and_store_chunks`` writes the embedder's
output straight into the store -- so a config that stated one and not the other
produced vectors of a width the store had declared it would not hold.

Nothing reported that. ``MemoryVectorStore.add_vectors`` accepted any width it
was handed, so a store declaring 384 and holding 768-wide rows searched fine as
long as every row and every query came from the same embedder, and the declared
number was simply a lie. Thirty-six test files in this repository stated 384
against an embedder producing 768 for as long as they had existed.

The rule these tests pin is the one ``OllamaProvider.embed`` already states for
its own layer -- *"a stated width is never ignored"*. Here that means the store's
declared width reaches the embedder when the config did not give the embedder
one of its own, so the single number a consumer wrote is the number both halves
use. An embedder width stated explicitly still wins: this supplies a missing
value, it does not overrule a given one.

The two subsystems share ``build_embedding_config``, which is where the rule
lives; the end-to-end cases below are what make it true of the surfaces a
consumer actually calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.knowledge.rag import RAGKnowledgeBase
from dataknobs_bots.memory.vector import VectorMemory
from dataknobs_bots.providers import build_embedding_config, create_embedding_provider

WIDTH = 384
"""A width the echo provider does not default to, so agreement cannot be luck.

``EchoProvider`` falls back to 768. Any assertion below that reads 384 is
therefore reading a number that travelled, not one that happened to match.
"""


# ---------------------------------------------------------------------------
# The helper both subsystems call
# ---------------------------------------------------------------------------


def test_the_store_width_is_supplied_when_the_flat_form_states_none() -> None:
    """Legacy flat keys: the embedder gets the store's width as its own."""
    built = build_embedding_config(
        embedding_provider="echo",
        embedding_model="test",
        store_dimensions=WIDTH,
    )

    assert built["dimensions"] == WIDTH


def test_the_store_width_is_supplied_inside_the_nested_form() -> None:
    """Nested ``embedding``: the width has to go *inside* it to be read.

    ``create_embedding_provider`` reads the top-level ``dimensions`` only on
    its legacy-flat branch; with a nested section present it reads endpoint,
    key and width from inside that section. A top-level value would be
    accepted here and silently dropped there, which is the failure mode this
    case exists to rule out.
    """
    built = build_embedding_config(
        embedding={"provider": "echo", "model": "test"},
        store_dimensions=WIDTH,
    )

    assert built["embedding"]["dimensions"] == WIDTH


def test_an_empty_nested_section_is_treated_as_the_flat_form() -> None:
    """An empty ``embedding`` is not the nested form, and must not become one.

    ``create_embedding_provider`` takes its nested branch only for a truthy
    section, so an empty one falls through to the flat keys and the width
    belongs at the top level. Writing it inside instead would do worse than
    miss: a section holding ``{"dimensions": N}`` is truthy, so the nested
    branch *would* be taken, with no ``provider`` or ``model`` in it -- and
    the caller's echo embedder would silently become the ollama default.
    """
    built = build_embedding_config(
        embedding={},
        embedding_provider="echo",
        embedding_model="test",
        store_dimensions=WIDTH,
    )

    assert built["embedding"] == {}
    assert built["dimensions"] == WIDTH


@pytest.mark.parametrize(
    "section",
    ["ollama", "dimensions please", ["ollama"], ["dimensions"], 768],
    ids=["str", "str-holding-the-word", "list", "list-holding-the-word", "int"],
)
def test_a_section_that_is_not_a_mapping_is_the_flat_form_too(section: Any) -> None:
    """The condition is mirrored whole, or it is not mirrored.

    ``create_embedding_provider`` takes its nested branch for a section that
    is truthy **and a dict**; this mirrored only the truthy half, and the
    dropped half is the only one that can disagree --- an empty dict is
    falsy on both readings, while a non-mapping section is nested to one and
    flat to the other. Three ways that showed, all measured:

    * ``"ollama"`` and ``768`` raise ``TypeError`` out of the width copy,
      where before the width existed they fell through to the flat branch
      exactly as the helper does;
    * ``"dimensions please"`` and ``["dimensions"]`` satisfy the ``in``
      check by accident, raise nothing, and **lose the width** --- this
      function records the nested form, the helper reads the flat one, and
      the store's number is written where nothing will look for it.

    A non-mapping section is not this function's to refuse: the helper
    accepts one and ignores it, so refusing here would reject a config that
    works. What it must do is agree about which form is in play.
    """
    built = build_embedding_config(
        embedding=section,
        embedding_provider="echo",
        embedding_model="test",
        store_dimensions=WIDTH,
    )

    assert built["embedding"] == section, "the section is passed through untouched"
    assert built["dimensions"] == WIDTH, (
        "the helper will read the flat branch for this section, so the width "
        "belongs at the top level"
    )


@pytest.mark.parametrize(
    "section",
    [None, {}, "ollama", ["dimensions"], 768, {"provider": "echo", "model": "test"}],
    ids=["none", "empty", "str", "list-holding-the-word", "int", "real"],
)
async def test_the_width_lands_where_the_provider_actually_reads_it(section: Any) -> None:
    """The agreement itself, asserted through both functions over every shape.

    The test above pins what ``build_embedding_config`` writes; this pins
    that the provider ``create_embedding_provider`` returns is carrying that
    number. It is the property the two share a predicate for, and it holds
    whether or not they keep sharing one --- so re-inlining the condition
    fails here even if the copy looks right, which is how the first copy got
    in.
    """
    built = build_embedding_config(
        embedding=section,
        embedding_provider="echo",
        embedding_model="test",
        store_dimensions=WIDTH,
    )

    provider = await create_embedding_provider(built)

    assert provider.config.dimensions == WIDTH, (
        f"the width was written where this provider does not read it: {built}"
    )


def test_every_key_the_flat_branch_forwards_is_one_this_can_project() -> None:
    """The sibling mirror, guarded rather than trusted.

    ``build_embedding_config`` takes a named parameter per top-level
    passthrough, so the set it can emit is written here and the set the
    helper forwards is written there --- the same two-copies shape that let
    the branch condition drift, one line along, and not yet drifted. A key
    added to the helper and not here would be a typed field a caller holds
    and nothing forwards.
    """
    import inspect

    from dataknobs_llm import FLAT_EMBEDDING_PASSTHROUGHS

    projectable = set(inspect.signature(build_embedding_config).parameters)
    missing = sorted(set(FLAT_EMBEDDING_PASSTHROUGHS) - projectable)

    assert not missing, (
        f"`create_embedding_provider` forwards {missing} from the top level and "
        "`build_embedding_config` has no parameter to put them there with, so a "
        "caller holding one as a typed field cannot get it to the provider"
    )

    built = build_embedding_config(
        embedding_provider="echo",
        embedding_model="test",
        **dict.fromkeys(FLAT_EMBEDDING_PASSTHROUGHS, 1),
    )
    assert set(FLAT_EMBEDDING_PASSTHROUGHS) <= set(built), (
        f"a passthrough was accepted as a parameter and then dropped: {built}"
    )


def test_the_nested_form_takes_the_width_only_where_it_is_read() -> None:
    """No top-level twin beside the section's own width.

    The top-level key is dead on the nested branch. Emitting one anyway
    would put two widths in one dict with only one of them consulted, which
    is the shape this whole fallback exists to remove.
    """
    built = build_embedding_config(
        embedding={"provider": "echo", "model": "test"},
        store_dimensions=WIDTH,
    )

    assert "dimensions" not in built


def test_a_stated_embedder_width_is_not_overruled() -> None:
    """The store's width supplies a missing value; it does not win a contest.

    A consumer who states both has said something deliberate -- most usefully
    that the two differ and they know it. Overruling that would trade one
    silent disagreement for another.
    """
    flat = build_embedding_config(
        embedding_provider="echo",
        embedding_model="test",
        dimensions=16,
        store_dimensions=WIDTH,
    )
    nested = build_embedding_config(
        embedding={"provider": "echo", "model": "test", "dimensions": 16},
        store_dimensions=WIDTH,
    )

    assert flat["dimensions"] == 16
    assert nested["embedding"]["dimensions"] == 16


def test_the_nested_section_the_caller_passed_is_not_mutated() -> None:
    """The config dict a caller holds is theirs, and comes back unchanged.

    ``RAGKnowledgeBaseConfig.embedding`` is a live field on a config object a
    consumer may read afterwards or reuse; writing the store's width into it
    would make the projection a side effect on its own input.
    """
    section: dict[str, Any] = {"provider": "echo", "model": "test"}

    build_embedding_config(embedding=section, store_dimensions=WIDTH)

    assert section == {"provider": "echo", "model": "test"}


def test_nothing_is_added_when_no_store_width_is_known() -> None:
    """Sparse-dict parity: an absent width stays absent, not ``None``.

    The helper's contract is that only set keys appear, so a store config
    naming no width leaves the embedder exactly as it was before this rule
    existed.
    """
    assert build_embedding_config(embedding_provider="echo") == {
        "embedding_provider": "echo",
    }


# ---------------------------------------------------------------------------
# The surfaces a consumer calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_rag_base_writes_the_width_its_store_declared() -> None:
    """The reproducer: 384 declared once, and the write used to be refused.

    Against the unfixed code the embedder produced 768, and
    ``load_markdown_text`` raised ``ValueError: Vector dimension mismatch:
    expected 384, got 768`` from inside ``add_vectors`` -- one config, two
    widths, and the store the one that noticed.
    """
    kb = await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": WIDTH},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )
    try:
        assert kb.embedding_provider.embedding_dim == WIDTH

        stored = await kb.load_markdown_text("# T\n\nBody.", source="t.md")

        assert stored > 0, "the ingest stored nothing, so nothing was written"
        assert kb.vector_store.dimensions == WIDTH
    finally:
        await kb.close()


@pytest.mark.asyncio
async def test_a_rag_base_using_the_nested_form_does_the_same() -> None:
    """The preferred config shape reaches the embedder by the other route."""
    kb = await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": WIDTH},
            "embedding": {"provider": "echo", "model": "test"},
        }
    )
    try:
        assert kb.embedding_provider.embedding_dim == WIDTH
        assert await kb.load_markdown_text("# T\n\nBody.", source="t.md") > 0
    finally:
        await kb.close()


@pytest.mark.asyncio
async def test_vector_memory_writes_the_width_its_store_declared() -> None:
    """The same defect on the other subsystem, whose two keys differ by an ``s``.

    ``VectorMemory`` spells the store's width ``dimension`` and the embedder's
    ``dimensions``. Its own docstring warns that the plural one is *"forwarded
    to the embedding provider, not the vector store"* -- a warning that exists
    because the pair is easy to get wrong, and that nothing enforced.
    """
    memory = await VectorMemory.from_config(
        {
            "backend": "memory",
            "dimension": WIDTH,
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )
    try:
        assert memory.embedding_provider.embedding_dim == WIDTH

        await memory.add_message("Test message", "user")

        assert memory.vector_store.dimensions == WIDTH
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_a_width_stated_for_the_embedder_alone_still_reaches_it() -> None:
    """The pre-existing passthrough is untouched by the new fallback.

    A config naming the embedder's width and *not* the store's was already
    correct, and stays so: the fallback has nothing to supply, and the store
    takes the factory's own default rather than this value.
    """
    memory = await VectorMemory.from_config(
        {
            "backend": "memory",
            "dimension": 16,
            "dimensions": 16,
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )
    try:
        assert memory.embedding_provider.embedding_dim == 16
        await memory.add_message("Test message", "user")
    finally:
        await memory.close()


# ---------------------------------------------------------------------------
# A width nobody wrote is not a width that was stated
# ---------------------------------------------------------------------------
#
# ``store_dimensions`` supplies the embedder the width **the config gave the
# store**. ``VectorMemoryConfig.dimension`` spelled a default of 1536, and
# ``VectorMemory._ainit`` wrote it into ``store_config`` unconditionally, so
# that argument was never ``None`` --- a class default reached the embedder
# as though a consumer had typed it.
#
# The precedent is one field up in the same dataclass. ``backend`` is
# ``str | None = None`` because spelling its default *"sent every config that
# omitted the key to the factory as an explicit choice"*, making an
# unpersisted store indistinguishable from one that was asked for. A width is
# the same shape of claim, and ``RAGKnowledgeBase`` --- the other caller of
# the same helper --- already reads ``vector_store.get("dimensions")``, which
# is absent when unstated. The two callers meant different things by
# "declared".


async def test_a_width_nobody_wrote_is_supplied_to_neither_half() -> None:
    """A config naming no width leaves both halves on their own defaults.

    Measured against ``origin/main``, this config produced a store declaring
    1536 and an embedder emitting 768 --- a disagreement the store had no way
    to report. Making them agree *on the 1536* is the other way to close it,
    and it is worse: the number is the dataclass's, not the consumer's, and
    with ``OllamaProvider`` it is fatal rather than merely wrong, because
    ``embed`` refuses a stated width its model cannot produce. A config
    naming no width anywhere would raise, citing a 1536 nobody typed.
    """
    memory = await VectorMemory.from_config(
        {"backend": "memory", "embedding_provider": "echo", "embedding_model": "test"}
    )
    assert memory.vector_store.dimensions == 0, (
        "no width was declared, so the store should hold the undeclared "
        "sentinel rather than the dataclass's 1536"
    )
    assert len(await memory.embedding_provider.embed("probe")) == 768, (
        "the embedder should be on its own default, not told to emit 1536"
    )


async def test_the_undeclared_pair_can_still_write() -> None:
    """And the pair still writes, which is what makes the above safe.

    An undeclared store width is only a tenable answer because
    ``VectorStoreBase._check_batch_width`` compares a batch to a *declared*
    width and this store declares none. The two changes are one story, and
    this is the assertion that joins them: leaving the width unstated must
    not turn ``add_message`` into the ``expected 0, got 768`` refusal.

    Asserted on the row reaching the store rather than on a later
    ``get_context``, which answers empty here for a reason that has nothing
    to do with widths --- the echo provider's vectors do not clear the
    default ``similarity_threshold``, and they do not at ``0.0`` either.
    """
    memory = await VectorMemory.from_config(
        {"backend": "memory", "embedding_provider": "echo", "embedding_model": "test"}
    )
    await memory.add_message("the sky is blue", "user")
    assert await memory.vector_store.count() == 1


async def test_a_width_the_consumer_does_write_still_reaches_both() -> None:
    """The positive control: supplying a *stated* width is the whole feature.

    ``dimension`` is the store's knob and the only one this config sets, so
    the embedder must still be handed it --- which is what would break if the
    fix above had removed the supply instead of the phantom.
    """
    memory = await VectorMemory.from_config(
        {
            "backend": "memory",
            "dimension": WIDTH,
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )
    assert memory.vector_store.dimensions == WIDTH
    assert len(await memory.embedding_provider.embed("probe")) == WIDTH
