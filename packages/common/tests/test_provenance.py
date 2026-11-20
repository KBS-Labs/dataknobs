"""Provenance records: frozen, and serialised so a stored record reads back unchanged.

The fixture is a purchasing summary a model wrote from two vendor quotes, with
a quote search run first, and a buyer's edit to its first line. It is written
for these tests: the quotes, vendor and ids are invented and carry no real
data.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from dataknobs_common.provenance import (
    LLMInvocation,
    ProvenanceRecord,
    RevisionRecord,
    SourceReference,
    ToolInvocation,
    create_provenance,
    now_iso,
)

AT = "2026-11-19T12:00:00+00:00"

QUOTE = SourceReference(
    source_id="QUOTE-0117",
    source_type="row",
    source_location="vendor-quotes",
    relevance="the lower unit price",
    excerpt="Linden Ceramics: 500 mugs at 2.40 each, delivered in 14 days.",
    confidence=0.9,
)
OTHER = SourceReference(source_id="QUOTE-0121", source_type="row")
SEARCH = ToolInvocation(
    tool_name="search_quotes",
    tool_version="1.2",
    parameters={"item": "stoneware mug", "max": 5},
    timestamp=AT,
)
CALL = LLMInvocation(
    purpose="write",
    model="ollama:qwen2.5:7b",
    prompt_hash="ab12",
    timestamp=AT,
    invocation_id="llm-1",
)
EDIT = RevisionRecord(
    revision_id="rev-1",
    previous_version="Linden Ceramics is cheaper.",
    reason="edited",
    changes_summary="line 1 edited",
    triggered_by="user:buyer",
    timestamp=AT,
    part="line-1",
    after="Linden Ceramics is cheaper per mug, but slower to deliver.",
    sources=("QUOTE-0117", "QUOTE-0121"),
)
RECORD = ProvenanceRecord(
    created_by="llm:ollama:qwen2.5:7b",
    created_at=AT,
    creation_method="llm_assisted",
    creation_context={"task": "Compare the mug quotes."},
    sources=(QUOTE, OTHER),
    tool_chain=(SEARCH,),
    llm_invocations=(CALL,),
    review_history=("eval-3",),
    revision_history=(EDIT,),
)

#: The keys dataknobs-bots' records serialise to.
BOTS_KEYS = {
    "created_by",
    "created_at",
    "creation_method",
    "creation_context",
    "sources",
    "tool_chain",
    "llm_invocations",
    "review_history",
    "revision_history",
}


def test_a_record_round_trips() -> None:
    assert ProvenanceRecord.from_dict(RECORD.to_dict()) == RECORD


def test_each_record_round_trips_on_its_own() -> None:
    for record in (QUOTE, OTHER, SEARCH, CALL, EDIT):
        assert type(record).from_dict(record.to_dict()) == record


def test_an_unset_optional_field_is_left_out() -> None:
    assert OTHER.to_dict() == {"source_id": "QUOTE-0121", "source_type": "row", "confidence": 1.0}
    assert QUOTE.to_dict() == {
        "source_id": "QUOTE-0117",
        "source_type": "row",
        "confidence": 0.9,
        "source_location": "vendor-quotes",
        "relevance": "the lower unit price",
        "excerpt": "Linden Ceramics: 500 mugs at 2.40 each, delivered in 14 days.",
    }
    assert "tool_version" not in ToolInvocation(tool_name="search_quotes", timestamp=AT).to_dict()
    assert SEARCH.to_dict()["tool_version"] == "1.2"


def test_without_the_additions_a_record_serialises_to_the_bots_keys_alone() -> None:
    plain = ProvenanceRecord(
        created_by="user:buyer",
        created_at=AT,
        llm_invocations=(LLMInvocation(purpose="write", timestamp=AT),),
        revision_history=(RevisionRecord(revision_id="rev-1", timestamp=AT),),
    )
    shown = plain.to_dict()
    assert set(shown) == BOTS_KEYS
    assert set(shown["llm_invocations"][0]) == {"purpose", "model", "prompt_hash", "timestamp"}
    assert set(shown["revision_history"][0]) == {
        "revision_id",
        "previous_version",
        "reason",
        "changes_summary",
        "triggered_by",
        "timestamp",
    }


def test_the_additions_are_keys_of_their_own() -> None:
    shown = RECORD.to_dict()
    assert shown["llm_invocations"][0]["invocation_id"] == "llm-1"
    revision = shown["revision_history"][0]
    assert (revision["part"], revision["after"], revision["sources"]) == (
        "line-1",
        "Linden Ceramics is cheaper per mug, but slower to deliver.",
        ["QUOTE-0117", "QUOTE-0121"],
    )


def test_a_line_removed_records_an_empty_after_and_an_unrecorded_one_none() -> None:
    removed = dataclasses.replace(EDIT, reason="rejected", after="")
    assert removed.to_dict()["after"] == ""
    assert RevisionRecord.from_dict(removed.to_dict()).after == ""
    assert "after" not in dataclasses.replace(EDIT, after=None).to_dict()


def test_the_records_are_frozen_and_a_revision_is_a_new_record() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        RECORD.created_by = "user:buyer"  # type: ignore[misc]
    revised = dataclasses.replace(
        RECORD, revision_history=(*RECORD.revision_history, RevisionRecord(timestamp=AT))
    )
    assert len(revised.revision_history) == 2
    assert RECORD.revision_history == (EDIT,), "the record handed on is unchanged"


def test_any_iterable_is_held_as_a_tuple() -> None:
    record = ProvenanceRecord(
        created_at=AT,
        sources=[QUOTE, OTHER],  # type: ignore[arg-type]
        tool_chain=iter([SEARCH]),  # type: ignore[arg-type]
        llm_invocations=(c for c in [CALL]),  # type: ignore[arg-type]
        review_history=["eval-3"],  # type: ignore[arg-type]
        revision_history=[EDIT],  # type: ignore[arg-type]
    )
    assert record.sources == (QUOTE, OTHER)
    assert record.tool_chain == (SEARCH,)
    assert record.llm_invocations == (CALL,)
    assert record.review_history == ("eval-3",)
    assert record.revision_history == (EDIT,)
    revision = RevisionRecord(revision_id="rev-2", sources=["QUOTE-0117"])  # type: ignore[arg-type]
    assert revision.sources == ("QUOTE-0117",)


def test_create_provenance_names_its_maker_and_method_and_takes_any_iterable() -> None:
    made = create_provenance(
        "llm:ollama:qwen2.5:7b", "llm_assisted", sources=[QUOTE], created_at=AT
    )
    assert made == ProvenanceRecord(
        created_by="llm:ollama:qwen2.5:7b",
        created_at=AT,
        creation_method="llm_assisted",
        sources=(QUOTE,),
    )


def test_a_revision_without_an_id_is_given_one_and_a_caller_may_pass_its_own() -> None:
    first, second = RevisionRecord(timestamp=AT), RevisionRecord(timestamp=AT)
    assert re.fullmatch(r"rev_[0-9a-f]{12}", first.revision_id)
    assert first.revision_id != second.revision_id
    assert RevisionRecord(revision_id="rev-1").revision_id == "rev-1"
    read = RevisionRecord.from_dict({"reason": "edited"})
    assert re.fullmatch(r"rev_[0-9a-f]{12}", read.revision_id)


def test_a_record_read_with_missing_keys_takes_the_defaults() -> None:
    read = ProvenanceRecord.from_dict({})
    assert read == ProvenanceRecord(created_at="")
    assert SourceReference.from_dict({"source_id": "QUOTE-0121", "source_type": "row"}) == OTHER
    assert LLMInvocation.from_dict({"purpose": "write"}) == LLMInvocation(
        purpose="write", timestamp=""
    )
    assert ToolInvocation.from_dict({"tool_name": "search_quotes"}) == ToolInvocation(
        tool_name="search_quotes", timestamp=""
    )


def test_a_time_is_stamped_in_utc_when_none_is_given() -> None:
    assert now_iso().endswith("+00:00")
    assert CALL.timestamp == AT
    assert LLMInvocation(purpose="write").timestamp.endswith("+00:00")
    assert ProvenanceRecord().created_at.endswith("+00:00")


#: A record as dataknobs-bots stored it before the records moved here: bots'
#: keys only, lists where the records now hold tuples.
STORED = {
    "created_by": "system:generator:quiz_gen_v1",
    "created_at": AT,
    "creation_method": "generator",
    "creation_context": {"topic": "algebra", "parameters": {"count": 5}},
    "sources": [
        {"source_id": "doc_1", "source_type": "document", "confidence": 1.0},
        {
            "source_id": "art_2",
            "source_type": "artifact",
            "confidence": 0.8,
            "source_location": "/docs/math.pdf",
            "relevance": "Derived from",
            "excerpt": "Chapter 3",
        },
    ],
    "tool_chain": [
        {"tool_name": "quiz_generator", "parameters": {"count": 5}, "timestamp": AT},
        {
            "tool_name": "formatter",
            "parameters": {},
            "timestamp": AT,
            "tool_version": "1.2.0",
        },
    ],
    "llm_invocations": [
        {"purpose": "decode_intent", "model": "llama3.2", "prompt_hash": "", "timestamp": AT}
    ],
    "review_history": ["eval_a", "eval_b"],
    "revision_history": [
        {
            "revision_id": "rev_0123456789ab",
            "previous_version": "1.0.0",
            "reason": "User edit",
            "changes_summary": "Fixed wording",
            "triggered_by": "user:jane",
            "timestamp": AT,
        }
    ],
}


def test_a_record_stored_before_the_additions_reads_back_unchanged() -> None:
    assert ProvenanceRecord.from_dict(STORED).to_dict() == STORED


def test_every_record_is_frozen() -> None:
    for record, name in (
        (QUOTE, "confidence"),
        (SEARCH, "parameters"),
        (CALL, "invocation_id"),
        (EDIT, "after"),
        (RECORD, "sources"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, name, None)


def test_editing_what_to_dict_returns_does_not_reach_the_record() -> None:
    shown = RECORD.to_dict()
    shown["creation_context"]["task"] = "changed"
    shown["tool_chain"][0]["parameters"]["max"] = 50
    shown["review_history"].append("eval-4")
    shown["revision_history"][0]["sources"].append("QUOTE-9999")

    assert RECORD.to_dict() != shown
    assert RECORD.creation_context == {"task": "Compare the mug quotes."}
    assert SEARCH.parameters == {"item": "stoneware mug", "max": 5}
    assert RECORD.review_history == ("eval-3",)
    assert EDIT.sources == ("QUOTE-0117", "QUOTE-0121")


def test_from_dict_does_not_share_the_stored_mappings() -> None:
    stored = RECORD.to_dict()
    read = ProvenanceRecord.from_dict(stored)
    stored["creation_context"]["task"] = "changed"
    stored["tool_chain"][0]["parameters"]["max"] = 50
    assert read == RECORD


def test_a_single_string_is_refused_where_several_values_are_held() -> None:
    with pytest.raises(TypeError, match=r"ProvenanceRecord: review_history"):
        ProvenanceRecord(review_history="eval-3")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"RevisionRecord: sources"):
        RevisionRecord(sources="QUOTE-0117")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"create_provenance: review_history"):
        create_provenance("user:buyer", "manual", review_history="eval-3")  # type: ignore[arg-type]


def test_create_provenance_fills_the_defaults_and_refuses_an_unknown_field() -> None:
    made = create_provenance("user:buyer", "manual")
    assert made.created_at.endswith("+00:00")
    assert made.creation_context == {}
    assert (made.sources, made.tool_chain, made.revision_history) == ((), (), ())
    context = {"task": "Compare the mug quotes."}
    from_factory = create_provenance("user:buyer", "manual", creation_context=context)
    context["task"] = "changed"
    assert from_factory.creation_context == {"task": "Compare the mug quotes."}
    with pytest.raises(TypeError):
        create_provenance("user:buyer", "manual", reviewer="eval-3")  # type: ignore[call-arg]


def test_the_records_compare_by_value() -> None:
    assert ProvenanceRecord.from_dict(RECORD.to_dict()) == RECORD
    assert dataclasses.replace(QUOTE, confidence=0.5) != QUOTE
    assert {QUOTE, OTHER, dataclasses.replace(QUOTE)} == {QUOTE, OTHER}


def test_the_names_are_reached_from_the_module_not_the_package_door() -> None:
    import dataknobs_common
    from dataknobs_common import provenance

    assert set(provenance.__all__) == {
        "LLMInvocation",
        "ProvenanceRecord",
        "RevisionRecord",
        "SourceReference",
        "ToolInvocation",
        "create_provenance",
        "now_iso",
    }
    assert not set(provenance.__all__) & set(dataknobs_common.__all__)


def test_the_callers_mapping_is_frozen_on_construction() -> None:
    context = {"task": "Compare the mug quotes.", "filters": {"item": ["mug"]}}
    parameters = {"item": "stoneware mug", "filters": {"max": 5}}
    record = ProvenanceRecord(created_at=AT, creation_context=context)
    search = ToolInvocation(tool_name="search_quotes", parameters=parameters, timestamp=AT)

    context["task"] = "changed"
    context["filters"]["item"].append("jug")
    parameters["filters"]["max"] = 50

    assert record.creation_context == {
        "task": "Compare the mug quotes.",
        "filters": {"item": ("mug",)},
    }
    assert search.parameters == {"item": "stoneware mug", "filters": {"max": 5}}


def test_a_reader_cannot_change_a_records_mapping() -> None:
    record = create_provenance(
        "user:buyer", "manual", creation_context={"filters": {"item": ["mug"]}}
    )
    with pytest.raises(TypeError):
        record.creation_context["task"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.creation_context["filters"]["item"] = "jug"
    with pytest.raises(AttributeError):
        record.creation_context["filters"]["item"].append("jug")
    with pytest.raises(TypeError):
        SEARCH.parameters["max"] = 50  # type: ignore[index]
    assert record.to_dict()["creation_context"] == {"filters": {"item": ["mug"]}}
    assert SEARCH.to_dict()["parameters"] == {"item": "stoneware mug", "max": 5}


def test_to_dict_thaws_the_mapping_back_to_plain_json_containers() -> None:
    import json

    record = create_provenance(
        "user:buyer", "manual", creation_context={"filters": {"item": ["mug"], "max": 5}}
    )
    shown = record.to_dict()
    assert type(shown["creation_context"]) is dict
    assert type(shown["creation_context"]["filters"]) is dict
    assert type(shown["creation_context"]["filters"]["item"]) is list
    assert json.loads(json.dumps(shown)) == shown


def test_a_mapping_field_takes_any_mapping_and_refuses_anything_else() -> None:
    from types import MappingProxyType

    from dataknobs_common.copying import FrozenMapping

    read_only = MappingProxyType({"task": "Compare the mug quotes."})
    record = ProvenanceRecord(created_at=AT, creation_context=read_only)
    assert isinstance(record.creation_context, FrozenMapping)
    assert record.creation_context == {"task": "Compare the mug quotes."}
    assert isinstance(ProvenanceRecord().creation_context, FrozenMapping)
    assert isinstance(ToolInvocation(tool_name="search_quotes").parameters, FrozenMapping)
    with pytest.raises(TypeError, match=r"ProvenanceRecord\.creation_context takes a mapping"):
        ProvenanceRecord(creation_context=[("task", "x")])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"ToolInvocation\.parameters takes a mapping"):
        ToolInvocation(tool_name="search_quotes", parameters=None)  # type: ignore[arg-type]


def test_a_mapping_that_holds_itself_is_refused() -> None:
    context: dict[str, object] = {"task": "Compare the mug quotes."}
    context["self"] = context
    with pytest.raises(ValueError):
        ProvenanceRecord(creation_context=context)


def test_a_replaced_record_keeps_its_frozen_mapping() -> None:
    revised = dataclasses.replace(RECORD, created_by="user:buyer")
    assert revised.creation_context == RECORD.creation_context
    with pytest.raises(TypeError):
        revised.creation_context["task"] = "changed"  # type: ignore[index]


def test_only_the_records_without_a_mapping_are_hashable() -> None:
    from collections.abc import Hashable

    for record in (QUOTE, CALL, EDIT):
        assert isinstance(record, Hashable)
        assert hash(record) == hash(dataclasses.replace(record))
    for record in (SEARCH, RECORD):
        assert not isinstance(record, Hashable)
        with pytest.raises(TypeError):
            hash(record)
