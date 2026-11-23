"""What ``create_intent_classifier`` shows of a classifier factory's failure.

The shims used to unwrap any ``ValueError`` off the registry's bounded error
and raise it as itself, so a consumer classifier factory's text -- a
connection URL to the embedding service it warms up, say -- reached the
caller. A configuration problem still arrives as ``ValueError``, as the shims
have always promised; it carries a factory's words only when dataknobs wrote
them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from dataknobs_common.exceptions import FactoryValueError, OperationError, ResourceError
from dataknobs_llm.intent import (
    create_intent_classifier,
    create_intent_classifier_async,
    intent_classifier_backends,
)

DSN = "postgres://admin:hunter2@db.internal:5432/prod"


async def _create(lane: str, name: str, config: dict[str, Any] | None = None) -> Any:
    if lane == "sync":
        return create_intent_classifier(name, config)
    return await create_intent_classifier_async(name, config)


@pytest.fixture
def leaky() -> Iterator[str]:
    def factory(config: dict[str, Any]) -> Any:
        error_type = config.get("raise", "value")
        if error_type == "resource":
            raise ResourceError(f"embedding service unreachable at {DSN}")
        if error_type == "runtime":
            raise RuntimeError(f"warm-up failed against {DSN}")
        raise ValueError(f"cannot load the vocabulary from {DSN}")

    intent_classifier_backends.register("test_leaky", factory)
    try:
        yield "test_leaky"
    finally:
        intent_classifier_backends.unregister("test_leaky")


@pytest.mark.parametrize("lane", ["sync", "async"])
class TestAFactorysOwnWordsStayBounded:
    async def test_a_value_error_is_still_a_value_error_but_bounded(
        self, lane: str, leaky: str
    ) -> None:
        with pytest.raises(ValueError) as caught:
            await _create(lane, leaky)

        assert isinstance(caught.value, FactoryValueError)
        assert DSN not in str(caught.value)
        assert str(caught.value) == "Failed to create intent classifier 'test_leaky' (ValueError)"
        assert DSN in str(caught.value.__cause__)

    async def test_a_resource_error_keeps_its_family(self, lane: str, leaky: str) -> None:
        with pytest.raises(ResourceError) as caught:
            await _create(lane, leaky, {"raise": "resource"})
        assert DSN not in str(caught.value)

    async def test_anything_else_is_a_bounded_operation_error(self, lane: str, leaky: str) -> None:
        with pytest.raises(OperationError) as caught:
            await _create(lane, leaky, {"raise": "runtime"})
        assert DSN not in str(caught.value)


@pytest.mark.parametrize("lane", ["sync", "async"])
class TestDataknobsOwnRefusalsKeepTheirWords:
    """The census rows for this helper: each raises what it raised before, word for word."""

    async def test_an_unknown_name(self, lane: str) -> None:
        with pytest.raises(ValueError, match=r"^Unknown intent_classifier: nope\. "):
            await _create(lane, "nope")

    async def test_a_composite_child_that_is_not_a_mapping(self, lane: str) -> None:
        with pytest.raises(ValueError) as caught:
            await _create(lane, "composite", {"classifiers": [3]})
        assert str(caught.value) == "composite classifier child spec must be a mapping, got int: 3"
        assert not isinstance(caught.value, FactoryValueError)

    async def test_a_composite_child_naming_no_classifier(self, lane: str) -> None:
        with pytest.raises(ValueError, match=r"^composite classifier child spec missing"):
            await _create(lane, "composite", {"classifiers": [{"config": {}}]})

    async def test_an_empty_composite(self, lane: str) -> None:
        with pytest.raises(ValueError) as caught:
            await _create(lane, "composite", {"classifiers": []})
        assert str(caught.value) == (
            "CompositeIntentClassifier requires at least one inner classifier"
        )

    async def test_a_composite_with_an_unknown_child(self, lane: str) -> None:
        with pytest.raises(ValueError, match=r"^Unknown intent_classifier: nope\. "):
            await _create(lane, "composite", {"classifiers": [{"classifier": "nope"}]})

    async def test_a_composite_whose_child_factory_leaks_is_bounded(
        self, lane: str, leaky: str
    ) -> None:
        with pytest.raises(FactoryValueError) as caught:
            await _create(lane, "composite", {"classifiers": [{"classifier": leaky}]})
        assert DSN not in str(caught.value)
        assert "'test_leaky'" in str(caught.value)
