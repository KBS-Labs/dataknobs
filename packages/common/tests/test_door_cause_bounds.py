"""The doors in ``dataknobs_common`` that write a message around a registry failure.

``tokenizer_from_config``, ``rules_from_config`` and the ontology loader's rung
builder each catch a registry's bounded failure and write their own message
around it. They used to interpolate the factory's error from ``__cause__``,
which put whatever a consumer's factory said -- a connection URL included -- in
front of a document's author. They now repeat a factory's words only when
dataknobs wrote them, and name a consumer factory's error by its type.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from dataknobs_common.exceptions import ValidationError, _is_written_by_dataknobs
from dataknobs_common.ontology import load_ontology
from dataknobs_common.ontology.loader import build_resolver
from dataknobs_common.text import (
    TOKENIZERS,
    VARIANT_RULES,
    rules_from_config,
    tokenizer_from_config,
)

DSN = "postgres://admin:hunter2@db.internal:5432/prod"


@pytest.fixture
def leaky_tokenizer() -> Iterator[str]:
    def factory(config: Any) -> Any:
        raise ValidationError(f"cannot load the vocabulary from {DSN}", context={"url": DSN})

    TOKENIZERS.register("test_leaky", factory)
    try:
        yield "test_leaky"
    finally:
        TOKENIZERS.unregister("test_leaky")


@pytest.fixture
def leaky_rule() -> Iterator[str]:
    def factory(config: Any) -> Any:
        raise ValueError(f"cannot load the lexicon from {DSN}")

    VARIANT_RULES.register("test_leaky", factory)
    try:
        yield "test_leaky"
    finally:
        VARIANT_RULES.unregister("test_leaky")


class TestTokenizerFromConfig:
    def test_a_consumer_factorys_words_are_named_by_type(self, leaky_tokenizer: str) -> None:
        with pytest.raises(ValidationError) as caught:
            tokenizer_from_config(leaky_tokenizer)

        assert DSN not in str(caught.value)
        assert DSN not in repr(caught.value.context)
        assert str(caught.value) == "tokenizer 'test_leaky' could not be built (ValidationError)"
        assert DSN in str(caught.value.__cause__.__cause__)

    def test_a_built_in_factorys_parameter_refusal_reaches_the_author_whole(self) -> None:
        with pytest.raises(ValidationError) as caught:
            tokenizer_from_config({"name": "words", "apostrophe": "keep"})

        assert (
            str(caught.value) == "tokenizer 'words' takes only 'apostrophes'; given ['apostrophe']"
        )
        assert _is_written_by_dataknobs(caught.value)

    def test_an_unknown_apostrophe_mode_reaches_the_author_whole(self) -> None:
        with pytest.raises(ValidationError) as caught:
            tokenizer_from_config({"name": "words", "apostrophes": "bogus"})

        assert "apostrophes must be one of ['keep', 'clitic']; given 'bogus'" in str(caught.value)


class TestRulesFromConfig:
    def test_a_consumer_factorys_words_are_named_by_type(self, leaky_rule: str) -> None:
        with pytest.raises(ValidationError) as caught:
            rules_from_config([leaky_rule])

        assert DSN not in str(caught.value)
        assert str(caught.value) == "variant rule 'test_leaky' could not be built (ValueError)"

    def test_a_built_in_rules_parameter_refusal_reaches_the_author_whole(self) -> None:
        with pytest.raises(ValidationError) as caught:
            rules_from_config([{"rule": "split", "seperator": "/"}])

        assert str(caught.value).startswith("variant rule 'split' does not take ['seperator']")
        assert _is_written_by_dataknobs(caught.value)


def _scan_document(mammals_path: Path, tokenizer: Any) -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load(mammals_path.read_text())
    document["ontology"]["resolver"] = {"rungs": [{"kind": "scan", "tokenizer": tokenizer}]}
    return document


class TestTheRungChain:
    """A tokenizer's refusal travels tokenizer door → rung registry → ``_rung_fault``."""

    def test_a_tokenizer_parameter_typo_reaches_the_document_author_whole(
        self, mammals_path: Path
    ) -> None:
        document = _scan_document(mammals_path, {"name": "words", "apostrophes": "bogus"})

        with pytest.raises(ValidationError) as caught:
            build_resolver(document, load_ontology(document))

        message = str(caught.value)
        assert message.startswith("`resolver.rungs[0]` names kind 'scan'")
        assert "apostrophes must be one of ['keep', 'clitic']; given 'bogus'" in message

    def test_a_consumer_tokenizers_words_never_reach_the_document_author(
        self, mammals_path: Path, leaky_tokenizer: str
    ) -> None:
        document = _scan_document(mammals_path, leaky_tokenizer)

        with pytest.raises(ValidationError) as caught:
            build_resolver(document, load_ontology(document))

        message = str(caught.value)
        assert DSN not in message
        assert message.endswith(
            "which this door could not build: tokenizer 'test_leaky' could not be built "
            "(ValidationError)"
        )

    def test_a_consumer_rung_factorys_words_never_reach_the_document_author(
        self, mammals_path: Path
    ) -> None:
        from dataknobs_common.entity_resolution import signal_backends

        def factory(config: Any) -> Any:
            raise ValidationError(f"cannot open the rung's index at {DSN}")

        signal_backends.register("test_leaky_rung", factory)
        try:
            document: dict[str, Any] = yaml.safe_load(mammals_path.read_text())
            document["ontology"]["resolver"] = {"rungs": [{"kind": "test_leaky_rung"}]}
            with pytest.raises(ValidationError) as caught:
                build_resolver(document, load_ontology(document))
        finally:
            signal_backends.unregister("test_leaky_rung")

        message = str(caught.value)
        assert DSN not in message
        assert message.endswith(
            "could not build: Failed to create plugin 'test_leaky_rung' (ValidationError)"
        )
