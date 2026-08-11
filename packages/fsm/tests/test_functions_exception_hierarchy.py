"""The functions-layer exceptions, against the shared hierarchy.

``dataknobs_fsm.functions.base`` predates the migration of this package's
exceptions onto ``dataknobs_common`` and was left behind by it: a second
hierarchy rooted at a plain ``Exception``, reusing four names that
``dataknobs_fsm.core.exceptions`` also defines as unrelated types.

That had two consequences these tests pin. Catching ``DataknobsError`` --
the whole point of a shared root -- missed every error the resource
backends, the transform library, and the validators raise, which is 60
raise sites. And because the types said nothing about what went wrong, a
caller that renders an exception (an HTTP boundary mapping types onto
statuses, retry logic keyed on a base) had nothing to read: a resource
acquisition failure and a validation failure were indistinguishable from
each other and from a bug.
"""

import warnings

import pytest

from dataknobs_common.exceptions import (
    ConfigurationError as CommonConfigurationError,
    DataknobsError,
    OperationError,
    ResourceError as CommonResourceError,
    ValidationError as CommonValidationError,
)
from dataknobs_fsm.functions.base import (
    ConfigurationError,
    FSMError,
    FunctionError,
    ResourceError,
    StateTransitionError,
    TransformError,
    ValidationError,
)


def _build(cls):
    """Construct one of these with whatever its signature requires.

    Deprecation notices are suppressed: the tests that assert on them do so
    explicitly, and every other test here is about classification.
    """
    args = {
        FSMError: ("boom",),
        ValidationError: ("boom",),
        TransformError: ("boom",),
        StateTransitionError: ("boom", "draft"),
        ResourceError: ("boom", "db", "connect"),
        ConfigurationError: ("boom",),
    }[cls]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return cls(*args)


ALL_TYPES = [
    FSMError,
    ValidationError,
    TransformError,
    StateTransitionError,
    ResourceError,
    ConfigurationError,
]


class TestTheyReachTheSharedRoot:
    """The defect: ``except DataknobsError`` did not reach any of these."""

    @pytest.mark.parametrize("cls", ALL_TYPES, ids=lambda c: c.__name__)
    def test_each_is_a_dataknobs_error(self, cls) -> None:
        assert isinstance(_build(cls), DataknobsError)

    def test_the_alias_reaches_it_too(self) -> None:
        assert FunctionError is StateTransitionError
        assert isinstance(_build(FunctionError), DataknobsError)


class TestEachSaysWhatWentWrong:
    """Classification, which is what a renderer or retry policy reads."""

    @pytest.mark.parametrize(
        ("cls", "common"),
        [
            (ValidationError, CommonValidationError),
            (ResourceError, CommonResourceError),
            (TransformError, OperationError),
            (StateTransitionError, OperationError),
            (ConfigurationError, CommonConfigurationError),
        ],
        ids=lambda v: getattr(v, "__name__", v),
    )
    def test_it_is_the_common_type_for_its_condition(self, cls, common) -> None:
        assert isinstance(_build(cls), common)

    def test_a_resource_failure_is_not_a_validation_failure(self) -> None:
        # The two conditions a caller most needs to tell apart: one is the
        # caller's fault and one is the deployment's.
        assert not isinstance(_build(ResourceError), CommonValidationError)
        assert not isinstance(_build(ValidationError), CommonResourceError)


class TestTheOldBaseStillCatchesWhatItCaught:
    """Back-compat: the rebase must not narrow an existing ``except``."""

    @pytest.mark.parametrize("cls", ALL_TYPES, ids=lambda c: c.__name__)
    def test_except_fsm_error_still_catches_it(self, cls) -> None:
        try:
            raise _build(cls)
        except FSMError:
            pass
        else:  # pragma: no cover - only reached on a regression
            pytest.fail(f"{cls.__name__} escaped `except FSMError`")


class TestTheDetailIsReadableAsContext:
    """The per-class attributes, under the name the shared hierarchy reads."""

    def test_validation_errors_reach_the_context(self) -> None:
        err = ValidationError("invalid", ["name is required", "age must be > 0"])
        assert err.validation_errors == ["name is required", "age must be > 0"]
        assert err.context == {"validation_errors": ["name is required", "age must be > 0"]}

    def test_no_validation_errors_leaves_the_context_empty(self) -> None:
        err = ValidationError("invalid")
        assert err.validation_errors == []
        assert not err.context

    def test_the_resource_and_operation_reach_the_context(self) -> None:
        err = ResourceError("connect failed", "primary-db", "acquire")
        assert err.resource_name == "primary-db"
        assert err.operation == "acquire"
        assert err.context == {
            "resource_name": "primary-db",
            "operation": "acquire",
        }

    def test_the_states_reach_the_context(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            err = StateTransitionError("not allowed", "draft", "shipped")
        assert err.from_state == "draft"
        assert err.to_state == "shipped"
        assert err.context == {"from_state": "draft", "to_state": "shipped"}


class TestTheNamesWithNoRaiserAreDeprecated:
    """Three of these are raised nowhere in the package and duplicate a name."""

    @pytest.mark.parametrize(
        "cls",
        [FSMError, StateTransitionError, FunctionError, ConfigurationError],
        ids=["FSMError", "StateTransitionError", "FunctionError", "ConfigurationError"],
    )
    def test_constructing_one_warns(self, cls) -> None:
        with pytest.warns(DeprecationWarning):
            _ = cls(
                *{
                    FSMError: ("boom",),
                    StateTransitionError: ("boom", "draft"),
                    ConfigurationError: ("boom",),
                }[cls]
            )

    @pytest.mark.parametrize(
        "cls",
        [ValidationError, TransformError, ResourceError],
        ids=lambda c: c.__name__,
    )
    def test_the_types_with_raisers_do_not_warn(self, cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            _build(cls)

    def test_subclassing_the_deprecated_base_does_not_warn(self) -> None:
        # `FSMError` stays as the base of the live types, so constructing one
        # of those must not fire the notice meant for direct use.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            ValidationError("boom")
            TransformError("boom")
            ResourceError("boom", "db", "connect")

    @pytest.mark.parametrize(
        ("base", "args"),
        [
            (FSMError, ("boom",)),
            (ConfigurationError, ("boom",)),
            (StateTransitionError, ("boom", "draft")),
        ],
        ids=["FSMError", "ConfigurationError", "StateTransitionError"],
    )
    def test_a_consumer_subclass_is_not_warned_about(self, base, args) -> None:
        """The notice is about the *name*, so only direct use should fire it.

        A deployment that subclasses one of these has stopped using the
        deprecated name — it is using its own. Warning there tells someone to
        migrate off something they are not on, points ``stacklevel`` at the
        wrong frame, and cannot be silenced except by silencing the notice
        that still matters.

        ``FSMError`` guarded this from the start because it stays as the base
        of the live types; the other two were written without the guard, which
        is the whole difference this pins.
        """
        subclass = type("ConsumerError", (base,), {})

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            subclass(*args)

    def test_the_alias_names_the_trap_it_creates(self) -> None:
        # `FunctionError` here is an alias of `StateTransitionError`, while
        # `core.exceptions.FunctionError` is about a failed user function.
        # The notice has to say so, or a reader follows the name.
        with pytest.warns(DeprecationWarning) as caught:
            FunctionError("boom", "draft")
        message = str(caught[0].message)
        assert "FunctionError" in message
        assert "TransitionError" in message
