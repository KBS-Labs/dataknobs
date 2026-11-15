"""Building a named thing from a document: one rule for tokenizers and variant rules.

``tokenizer_from_config`` and ``rules_from_config`` both turn a configuration's
name into a built callable through a :class:`~dataknobs_common.registry.PluginRegistry`.
What reaches the caller when the named factory fails is decided once, for both:

- a factory refusing its parameters (``ValueError`` or ``ValidationError``) is
  quoted, because that is the documented way a factory says what is wrong;
- any other exception is named by its type alone, chained with ``from``,
  because its text is a driver's or a service's and can carry a URL or a path;
- a ``NotFoundError`` raised inside a factory is wrapped like any other,
  rather than escaping as a type the builder's docstring does not promise.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest

from dataknobs_common.exceptions import NotFoundError, OperationError, ValidationError
from dataknobs_common.text import (
    TOKENIZERS,
    VARIANT_RULES,
    rules_from_config,
    tokenizer_from_config,
)

_NAME = "test_failing_factory"
_SECRET = "postgres://reader:hunter2@db.internal:5432/forms"


def _build_tokenizer() -> object:
    return tokenizer_from_config(_NAME)


def _build_rule() -> object:
    return rules_from_config([_NAME])


_BUILDERS = {
    "tokenizer": (TOKENIZERS, _build_tokenizer),
    "variant_rule": (VARIANT_RULES, _build_rule),
}


@pytest.fixture(params=sorted(_BUILDERS))
def failing(
    request: pytest.FixtureRequest,
) -> Iterator[Callable[[BaseException], Callable[[], object]]]:
    """Register a factory under ``_NAME`` that raises what the test hands it."""
    registry, build = _BUILDERS[request.param]

    def arm(error: BaseException) -> Callable[[], object]:
        def factory(config: Mapping[str, Any]) -> Any:
            raise error

        registry.register(_NAME, factory)
        return build

    yield arm
    if registry.is_registered(_NAME):
        registry.unregister(_NAME)


def test_a_factory_refusing_its_parameters_is_quoted(
    failing: Callable[[BaseException], Callable[[], object]],
) -> None:
    build = failing(ValueError("takes no 'colour'"))
    with pytest.raises(ValidationError, match=rf"{_NAME}.*takes no 'colour'"):
        build()


def test_a_factory_validation_error_is_quoted(
    failing: Callable[[BaseException], Callable[[], object]],
) -> None:
    build = failing(ValidationError("needs 'field'"))
    with pytest.raises(ValidationError, match=rf"{_NAME}.*needs 'field'"):
        build()


@pytest.mark.parametrize("raised", [RuntimeError, OSError, KeyError])
def test_any_other_factory_failure_is_named_by_type_not_quoted(
    failing: Callable[[BaseException], Callable[[], object]], raised: type[Exception]
) -> None:
    original = raised(f"could not reach {_SECRET}")
    build = failing(original)
    with pytest.raises(ValidationError) as caught:
        build()
    message = str(caught.value)
    assert _NAME in message
    assert raised.__name__ in message
    assert "hunter2" not in message and "db.internal" not in message
    assert caught.value.__cause__ is original


def test_an_operation_error_from_a_factory_is_named_by_type(
    failing: Callable[[BaseException], Callable[[], object]],
) -> None:
    original = OperationError(f"pool exhausted for {_SECRET}")
    build = failing(original)
    with pytest.raises(ValidationError) as caught:
        build()
    assert "hunter2" not in str(caught.value)
    assert "OperationError" in str(caught.value)
    assert caught.value.__cause__ is original


def test_a_not_found_error_from_a_factory_is_wrapped(
    failing: Callable[[BaseException], Callable[[], object]],
) -> None:
    original = NotFoundError(f"no segmenter dictionary at {_SECRET}")
    build = failing(original)
    with pytest.raises(ValidationError) as caught:
        build()
    assert not isinstance(caught.value, NotFoundError)
    assert "NotFoundError" in str(caught.value)
    assert "hunter2" not in str(caught.value)
    assert caught.value.__cause__ is original
