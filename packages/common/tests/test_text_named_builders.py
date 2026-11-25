# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""One rule for building the callable a document names, at both doors.

``tokenizer_from_config`` and ``rules_from_config`` share one builder, so every
test here runs against both. What a consumer factory's own words may reach is
``test_door_cause_bounds.py``'s subject; this file is about the type a caller
catches, whichever way a name cannot be built, and what the error says of it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from dataknobs_common.exceptions import (
    NotFoundError,
    OperationError,
    ValidationError,
    _is_written_by_dataknobs,
)
from dataknobs_common.registry import PluginRegistry
from dataknobs_common.text import (
    TOKENIZERS,
    VARIANT_RULES,
    rules_from_config,
    tokenizer_from_config,
)

NAME = "test_named_builder"
DSN = "postgres://reader:hunter2@db.internal:5432/forms"


@dataclass(frozen=True)
class Door:
    """One door, and the registry, noun and context its refusals use."""

    registry: PluginRegistry[Any]
    build: Callable[[], object]
    noun: str
    context: Mapping[str, Any]


DOORS = {
    "tokenizer": Door(
        TOKENIZERS, lambda: tokenizer_from_config(NAME), "tokenizer", {"tokenizer": NAME}
    ),
    "rule": Door(
        VARIANT_RULES,
        lambda: rules_from_config([NAME]),
        "variant rule",
        {"rule": NAME, "position": 0},
    ),
}


@pytest.fixture(params=sorted(DOORS))
def door(request: pytest.FixtureRequest) -> Iterator[Door]:
    """A door whose registry forgets ``NAME`` afterwards, however it was declared."""
    chosen = DOORS[request.param]
    yield chosen
    if chosen.registry.is_known(NAME):
        chosen.registry.unregister(NAME)


def _raising(error: Exception) -> Callable[[Any], Any]:
    def factory(config: Any) -> Any:
        raise error

    return factory


def _refusal(door: Door) -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        door.build()
    return caught.value


@pytest.mark.parametrize(
    "raised",
    [RuntimeError, OSError, KeyError, ValueError, ValidationError, NotFoundError, OperationError],
)
def test_any_failure_of_a_consumers_factory_is_a_validation_error_naming_its_type(
    door: Door, raised: type[Exception]
) -> None:
    original = raised(f"could not reach {DSN}")
    door.registry.register(NAME, _raising(original))

    refusal = _refusal(door)

    assert str(refusal) == f"{door.noun} {NAME!r} could not be built ({raised.__name__})"
    assert refusal.context == door.context
    assert _is_written_by_dataknobs(refusal)
    # The registry's bounded error first, then what the factory raised.
    assert refusal.__cause__ is not None
    assert refusal.__cause__.__cause__ is original


def test_a_miss_in_a_registry_the_factory_consults_reaches_the_author(door: Door) -> None:
    """The inner registry wrote those words, so they are shown."""
    inner: PluginRegistry[Any] = PluginRegistry("inner", not_found_kind="dictionary")
    door.registry.register(NAME, lambda config: inner.create("absent", config={}))

    refusal = _refusal(door)

    assert str(refusal).startswith(f"{door.noun} {NAME!r} Unknown dictionary: absent")
    assert isinstance(refusal.__cause__, NotFoundError)


def test_a_name_declared_unavailable_is_refused_with_its_reason(door: Door) -> None:
    """It is spelled right and cannot be built here, so it is not reported as a typo."""
    door.registry.declare_unavailable(NAME, reason="needs `pip install segmenter`")

    refusal = _refusal(door)

    assert str(refusal) == (
        f"{door.noun} {NAME!r} is not available here: needs `pip install segmenter`"
    )
    assert refusal.context == {**door.context, "unavailable": "needs `pip install segmenter`"}
    assert _is_written_by_dataknobs(refusal)


def test_an_unknown_name_lists_the_registered_ones(door: Door) -> None:
    refusal = _refusal(door)

    assert str(refusal).startswith(f"unknown {door.noun} {NAME!r}. Registered ")
    assert str(sorted(door.registry.list_keys())) in str(refusal)
    assert refusal.context == door.context


def test_a_factory_result_that_is_not_callable_is_named_by_type(door: Door) -> None:
    class Handle:
        def __repr__(self) -> str:
            return DSN

    door.registry.register(NAME, lambda config: Handle())

    refusal = _refusal(door)

    assert f"{door.noun} {NAME!r}'s factory returned a Handle, which is not callable" in str(
        refusal
    )
    assert "hunter2" not in str(refusal)
    assert refusal.context == door.context


@pytest.mark.parametrize("spec", [7, {}, {"name": ""}, {"rule": 3}])
def test_a_spec_naming_nothing_is_refused_by_both(spec: Any) -> None:
    with pytest.raises(ValidationError, match=r"^tokenizer must be a tokenizer name"):
        tokenizer_from_config(spec)
    with pytest.raises(ValidationError, match=r"^variant rule 1 must be a rule name") as caught:
        rules_from_config(["drop_parentheticals", spec])
    assert caught.value.context == {"position": 1, "spec": spec}
