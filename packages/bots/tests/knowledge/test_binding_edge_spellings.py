"""The binding means the same thing at every surface that reads it.

Two spellings of "is there a binding?" coexisted. The store layer settled
this for itself — ``VectorStoreBase._is_scoped`` is ``domain_id is not
None`` precisely because a truthiness test made an empty-string domain
isolate on three backends and run unscoped on a fourth, "a tenant
boundary that disappeared on a config-selected backend swap". The
knowledge base inherited both spellings: identity stamping and filter
composition test ``is not None``, while the chunk-id fold tested
truthiness.

An empty-string domain therefore got scoped reads and a scoped write tag
while its chunk ids stayed unnamespaced — which is the collision the
binding exists to prevent, reintroduced at the one value where the two
spellings disagree.
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_bots.knowledge import RAGKnowledgeBase


class TestTheEmptyStringDomainIsADomain:
    """``""`` is a configured scope, not an absent one."""

    def test_it_folds_into_the_chunk_id(self) -> None:
        """The fold must agree with what the filter and the stamp did."""
        prefix, sep = RAGKnowledgeBase._derive_chunk_prefix("overview", {"domain_id": ""})

        assert sep == "\x1f", "an empty-string domain fell back to the legacy shape"
        assert prefix == "\x1foverview", (
            "the chunk id carries no domain, so two domains over one store collide "
            "on it — while their reads and writes are scoped apart"
        )

    def test_an_absent_domain_still_keeps_the_historical_shape(self) -> None:
        """Only ``None``/absent is unbound. This is the byte-identical path."""
        assert RAGKnowledgeBase._derive_chunk_prefix("overview", None) == ("overview", "_")
        assert RAGKnowledgeBase._derive_chunk_prefix("overview", {}) == ("overview", "_")
        assert RAGKnowledgeBase._derive_chunk_prefix("overview", {"domain_id": None}) == (
            "overview",
            "_",
        )

    def test_a_normal_domain_is_unchanged(self) -> None:
        """The ordinary case, pinned alongside the edge one."""
        assert RAGKnowledgeBase._derive_chunk_prefix("overview", {"domain_id": "bot-a"}) == (
            "bot-a\x1foverview",
            "\x1f",
        )


class TestEveryConflictIsReported:
    """A warning keyed only by the key hides every later conflict.

    ``_warn_once`` exists because these conditions recur per chunk and
    per read, so an unguarded warning buries the first under thousands.
    But one knowledge base can be handed many *different* conflicting
    values over its life, and collapsing them by key means the second
    onwards are silently re-tagged with nothing said.
    """

    @staticmethod
    def _kb() -> Any:
        kb = RAGKnowledgeBase.__new__(RAGKnowledgeBase)
        kb._identity_warnings = set()
        return kb

    def test_a_second_distinct_conflict_is_still_reported(self, caplog: Any) -> None:
        kb = self._kb()

        with caplog.at_level(logging.WARNING):
            kb._stamp_identity({"domain_id": "umbrella"}, "domain_id", "acme")
            kb._stamp_identity({"domain_id": "initech"}, "domain_id", "acme")

        reported = "\n".join(r.getMessage() for r in caplog.records)
        assert "umbrella" in reported
        assert "initech" in reported, "the second conflicting value was silently swallowed"

    def test_the_same_conflict_repeated_is_still_reported_once(self, caplog: Any) -> None:
        """The per-chunk flood this guard exists for stays suppressed."""
        kb = self._kb()

        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                kb._stamp_identity({"domain_id": "umbrella"}, "domain_id", "acme")

        assert len(caplog.records) == 1
