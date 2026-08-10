"""Tests for API exceptions and dependencies."""

import json
import warnings
from collections.abc import Callable
from pathlib import Path

import pytest

from dataknobs_bots.api import exceptions as api_exceptions
from dataknobs_bots.api.exceptions import (
    DEFAULT_ERROR_POLICY,
    APIError,
    BotCreationError,
    BotNotFoundError,
    ConfigurationError,
    ConversationNotFoundError,
    ErrorPolicy,
    RateLimitError,
    ValidationError,
    register_exception_handlers,
    resolve_error_policy,
)
from dataknobs_bots.api.dependencies import (
    _BotManagerSingleton,
    get_bot_manager,
    init_bot_manager,
    reset_bot_manager,
)
import dataknobs_common
from dataknobs_common import exceptions as common_exceptions
from dataknobs_common.exceptions import (
    ConfigurationError as CommonConfigurationError,
)
from dataknobs_common.exceptions import (
    DataknobsError,
)
from dataknobs_common.exceptions import (
    NotFoundError as CommonNotFoundError,
)
from dataknobs_common.exceptions import (
    OperationError as CommonOperationError,
)
from dataknobs_common.exceptions import (
    RateLimitError as CommonRateLimitError,
)
from dataknobs_common.exceptions import (
    ResourceError as CommonResourceError,
)
from dataknobs_common.exceptions import (
    ValidationError as CommonValidationError,
)


class TestAPIError:
    """Tests for APIError base class."""

    def test_basic_error(self):
        """Test creating a basic API error."""
        error = APIError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.status_code == 500
        assert error.error_code == "APIError"

    def test_error_with_status_code(self):
        """Test creating an error with custom status code."""
        error = APIError("Not found", status_code=404)

        assert error.status_code == 404

    def test_error_with_detail(self):
        """Test creating an error with detail."""
        detail = {"field": "email", "reason": "invalid format"}
        error = APIError("Validation failed", detail=detail)

        assert error.detail == detail
        # Also accessible via context (inherited from DataknobsError)
        assert error.context == detail

    def test_error_with_error_code(self):
        """Test creating an error with custom error code."""
        error = APIError("Custom error", error_code="CUSTOM_ERROR")

        assert error.error_code == "CUSTOM_ERROR"

    def test_to_dict(self):
        """Test converting error to dictionary."""
        error = APIError(
            "Test error",
            status_code=400,
            detail={"key": "value"},
            error_code="TEST_ERROR",
        )

        error_dict = error.to_dict()

        assert error_dict["error"] == "TEST_ERROR"
        assert error_dict["message"] == "Test error"
        assert error_dict["detail"] == {"key": "value"}
        assert "timestamp" in error_dict

    def test_inherits_from_dataknobs_error(self):
        """Test that APIError inherits from DataknobsError."""
        error = APIError("Test")
        assert isinstance(error, DataknobsError)


class TestBotNotFoundError:
    """Tests for BotNotFoundError."""

    def test_basic_error(self):
        """Test creating a bot not found error."""
        error = BotNotFoundError("my-bot-id")

        assert "my-bot-id" in str(error)
        assert error.status_code == 404
        assert error.detail["bot_id"] == "my-bot-id"

    def test_inherits_from_common_not_found(self):
        """Test that BotNotFoundError inherits from CommonNotFoundError."""
        error = BotNotFoundError("my-bot-id")
        assert isinstance(error, CommonNotFoundError)


class TestBotCreationError:
    """Tests for BotCreationError."""

    def test_basic_error(self):
        """Test creating a bot creation error."""
        error = BotCreationError("my-bot-id", "Invalid configuration")

        assert "my-bot-id" in str(error)
        assert "Invalid configuration" in str(error)
        assert error.status_code == 500
        assert error.detail["bot_id"] == "my-bot-id"
        assert error.detail["reason"] == "Invalid configuration"

    def test_catchable_as_the_common_operation_error(self):
        """Failing to create a bot is an operation failure.

        This is the same reasoning that makes the API ``RateLimitError`` a
        ``CommonRateLimitError`` — and therefore an ``OperationError``. A
        consumer catching ``OperationError`` to mean "something DataKnobs
        attempted did not succeed" has no way to tell that bot creation is
        excluded from that set, because nothing about the name suggests it.

        Unlike its twinned siblings there is no same-named counterpart to
        fall back on, so the derived same-name rule cannot reach this class;
        only the expectation table can, which is why recording it as
        API-only had to be an assertion rather than a skip.
        """
        error = BotCreationError("my-bot-id", "Invalid configuration")

        assert isinstance(error, CommonOperationError)


class TestConversationNotFoundError:
    """Tests for ConversationNotFoundError."""

    def test_basic_error(self):
        """Test creating a conversation not found error."""
        error = ConversationNotFoundError("conv-123")

        assert "conv-123" in str(error)
        assert error.status_code == 404
        assert error.detail["conversation_id"] == "conv-123"

    def test_inherits_from_common_not_found(self):
        """Test that ConversationNotFoundError inherits from CommonNotFoundError."""
        error = ConversationNotFoundError("conv-123")
        assert isinstance(error, CommonNotFoundError)


class TestValidationError:
    """Tests for ValidationError."""

    def test_basic_error(self):
        """Test creating a validation error."""
        error = ValidationError("Invalid input")

        assert str(error) == "Invalid input"
        assert error.status_code == 422

    def test_error_with_detail(self):
        """Test creating a validation error with detail."""
        detail = {"field": "email", "constraint": "must be valid email"}
        error = ValidationError("Validation failed", detail=detail)

        assert error.detail == detail

    def test_inherits_from_common_validation(self):
        """Test that ValidationError inherits from CommonValidationError."""
        error = ValidationError("Test")
        assert isinstance(error, CommonValidationError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_basic_error(self):
        """Test creating a configuration error."""
        error = ConfigurationError("Invalid config")

        assert str(error) == "Invalid config"
        assert error.status_code == 500

    def test_error_with_config_key(self):
        """Test creating a configuration error with config key."""
        error = ConfigurationError("Invalid value", config_key="llm.model")

        assert error.detail["config_key"] == "llm.model"

    def test_inherits_from_common_configuration(self):
        """Test that ConfigurationError inherits from CommonConfigurationError."""
        error = ConfigurationError("Test")
        assert isinstance(error, CommonConfigurationError)


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_basic_error(self):
        """Test creating a rate limit error."""
        error = RateLimitError()

        assert "Rate limit exceeded" in str(error)
        assert error.status_code == 429

    def test_error_with_custom_message(self):
        """Test creating a rate limit error with custom message."""
        error = RateLimitError("Too many requests")

        assert str(error) == "Too many requests"

    def test_error_with_retry_after(self):
        """Test creating a rate limit error with retry_after."""
        error = RateLimitError(retry_after=60)

        assert error.detail["retry_after"] == 60

    def test_inherits_from_common_rate_limit(self):
        """Test that RateLimitError inherits from CommonRateLimitError."""
        error = RateLimitError("Test")
        assert isinstance(error, CommonRateLimitError)

    def test_common_catch_catches_both_variants(self):
        """``except`` on the common name must catch the API variant too.

        This is the defect in full: DK's own ``RateLimitMiddleware`` raises
        the *common* ``RateLimitError``, so a consumer writes one ``except``
        against the common name. Before this became a subclass, that block
        silently never fired for the API variant.
        """
        caught = []
        for error in (CommonRateLimitError("common"), RateLimitError("api")):
            try:
                raise error
            except CommonRateLimitError as exc:
                caught.append(str(exc))

        assert caught == ["common", "api"]

    def test_retry_after_is_both_an_attribute_and_a_detail_field(self):
        """``retry_after`` survives the widened MRO.

        Ordering guard. ``APIError.__init__`` now reaches
        ``CommonRateLimitError.__init__`` through the MRO, which assigns
        ``self.retry_after`` from *its own* default of ``None``. If the
        assignment in ``RateLimitError.__init__`` were moved above that
        call, this reads back ``None`` — a null-valued ``Retry-After``,
        which is worse than the original bug because it looks correct.
        """
        error = RateLimitError(retry_after=30.0)

        assert error.retry_after == 30.0
        assert error.detail["retry_after"] == 30.0

    def test_a_zero_retry_after_still_reaches_the_response_body(self):
        """``retry_after=0`` is a value, not an absence.

        The two views of it are populated by different rules — the attribute
        by direct assignment, the ``detail`` entry by a conditional — so a
        falsy-but-present value can appear in one and not the other. Zero is
        the reachable case: ``PyrateRateLimiter.get_status`` reports
        ``reset_after=0.0`` unconditionally and ``InMemoryRateLimiter`` does
        so whenever the window has just drained, and the re-raise recipe in
        the API docs forwards that straight into this constructor.

        The consequence is not a wrong number but a missing one: the 429 body
        carries no ``retry_after`` at all, so a client that reads it back
        learns nothing, while server-side code reading the attribute sees
        ``0.0``. "Retry immediately" and "no guidance offered" are different
        answers, and only one of them is true.
        """
        error = RateLimitError(retry_after=0.0)

        assert error.retry_after == 0.0
        assert error.detail["retry_after"] == 0.0

    def test_retry_after_defaults_to_none_as_an_attribute(self):
        """The common hierarchy's attribute exists even when unset."""
        error = RateLimitError()

        assert error.retry_after is None
        assert error.detail == {}


#: Every exception class this module defines, mapped to the
#: ``dataknobs_common`` class it is expected to subclass. Written out rather
#: than derived so it also covers the ones whose common base has a
#: *different* name — the derived same-name rule below cannot see those.
#:
#: An API-layer-only concept records a rationale string instead of a class.
#: A bare ``None`` would be indistinguishable from "nobody thought about it",
#: and because the value is what the check dispatches on, that is precisely
#: the entry a new class is most likely to be given by default.
_EXPECTED_COMMON_BASE: dict[str, type[BaseException] | str] = {
    "APIError": "the API layer's own base — it is what the common hierarchy "
    "is being extended *into*, so it has no counterpart to inherit",
    "BotCreationError": CommonOperationError,
    "BotNotFoundError": CommonNotFoundError,
    "ConversationNotFoundError": CommonNotFoundError,
    "ValidationError": CommonValidationError,
    "ConfigurationError": CommonConfigurationError,
    "RateLimitError": CommonRateLimitError,
}


def _api_exception_classes() -> dict[str, type[BaseException]]:
    """The exception classes defined by ``dataknobs_bots.api.exceptions``."""
    return {
        name: obj
        for name, obj in vars(api_exceptions).items()
        if isinstance(obj, type)
        and issubclass(obj, BaseException)
        and obj.__module__ == api_exceptions.__name__
    }


#: The HTTP status and disclosure each ``dataknobs_common`` exception type must
#: produce, as ``(status, message, context)``, written out independently of
#: ``DEFAULT_ERROR_POLICY``. Asserting the table against itself would pass for
#: any value it happened to hold; this is the contract, and changing a row in
#: the source has to fail here first.
_EXPECTED_POLICY: dict[str, tuple[int, bool, bool]] = {
    "ValidationError": (422, True, True),
    "ConsentRequiredError": (403, True, True),
    "ConcurrencyError": (409, True, True),
    # Overrides an inherited 500 rather than declaring a base's status; lives
    # in dataknobs_common.transitions, not the exception module.
    "InvalidTransitionError": (409, True, True),
    "RateLimitError": (429, True, True),
    # Message only: the context enumerates a registry keyspace.
    "NotFoundError": (404, True, False),
    # Message only: the type's documented example puts a SQL query in context.
    "TimeoutError": (504, True, False),
    "ConfigurationError": (500, False, False),
    # Bounded messages, unlike their parent — but the missing-attribute text
    # enumerates the target module's public callables, which is a deployment
    # inventory to an HTTP caller.
    "DottedPathError": (500, False, False),
    "DottedPathTypeError": (500, False, False),
    "ResourceError": (503, False, False),
    "SerializationError": (500, False, False),
    "OperationError": (500, False, False),
    "DataknobsError": (500, False, False),
}


#: The published copy of the same contract. Transcluded into the site, so this
#: is what a deployment reads when deciding whether a route can be public.
_POLICY_DOC = Path(__file__).resolve().parents[1] / "docs" / "MULTI_TENANT.md"

#: The header cell that identifies the status table among the doc's several
#: tables. Matching on content rather than position, so an added table above it
#: does not silently redirect the parse.
_POLICY_TABLE_HEADER = "DataKnobs error"


def _documented_policy() -> dict[str, tuple[int, bool, bool]]:
    """Parse the status/disclosure table out of ``MULTI_TENANT.md``.

    Raises rather than returning something partial for every shape it does not
    recognise. A parser that quietly yields an empty dict on a reworded header
    would let the comparison below pass forever while the two copies drifted —
    which is the failure mode the guard exists to prevent, reintroduced inside
    the guard.
    """
    assert _POLICY_DOC.is_file(), f"{_POLICY_DOC} is missing"
    lines = _POLICY_DOC.read_text(encoding="utf-8").splitlines()

    starts = [
        i for i, line in enumerate(lines) if line.startswith("|") and _POLICY_TABLE_HEADER in line
    ]
    assert len(starts) == 1, (
        f"expected exactly one table headed {_POLICY_TABLE_HEADER!r} in "
        f"{_POLICY_DOC.name}, found {len(starts)}"
    )

    documented: dict[str, tuple[int, bool, bool]] = {}
    # +2: past the header and the `|---|` separator beneath it.
    for line in lines[starts[0] + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        assert len(cells) == 4, f"unexpected row shape in {_POLICY_DOC.name}: {line!r}"
        name, status, *disclosure = cells
        # "no — masked, see below" and friends: the answer is the first word.
        for cell in disclosure:
            assert cell.split()[0] in {"yes", "no"}, (
                f"disclosure columns must begin yes/no, got {cell!r}"
            )
        documented[name.strip("`")] = (
            int(status),
            disclosure[0].startswith("yes"),
            disclosure[1].startswith("yes"),
        )

    assert documented, f"parsed no rows from the table in {_POLICY_DOC.name}"
    return documented


async def _response_for(exc, *, error_policy=None, raise_app_exceptions=True):
    """Raise ``exc`` from a route and return the response the handlers built.

    End-to-end through the real ASGI stack rather than by calling a handler
    directly, because which handler runs is the thing under test: Starlette
    picks it by walking ``type(exc).__mro__``, and calling one by hand would
    assume the answer.

    ``raise_app_exceptions`` stays at httpx's strict default so that an
    exception reaching the ``Exception`` catch-all surfaces here instead of
    being quietly absorbed — Starlette's ``ServerErrorMiddleware`` re-raises
    after calling that handler, which is the distinction
    :class:`TestAsgiPropagation` is about. Pass ``False`` for a case that is
    *meant* to reach the catch-all.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    app = FastAPI()
    register_exception_handlers(app, error_policy=error_policy)

    @app.get("/boom")
    async def boom():
        raise exc

    transport = ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/boom")


# Subclasses defined outside `dataknobs_common/exceptions.py`, each reached by
# MRO rather than by a row of its own. Built here rather than in the
# parametrize list so their differing constructor shapes stay out of the case
# table, which is about statuses.


def _record_not_found():
    from dataknobs_data.exceptions import RecordNotFoundError

    return RecordNotFoundError("rec-1")


def _context_length_exceeded():
    from dataknobs_llm.exceptions import ContextLengthExceededError

    return ContextLengthExceededError("prompt too long")


def _pack_resolution_error():
    from dataknobs_common.packs import PackResolutionError

    return PackResolutionError("pack 'audit' is not registered", reason="unknown_pack")


def _capability_not_supported():
    from dataknobs_common import Capability, CapabilityNotSupportedError

    return CapabilityNotSupportedError(Capability.CONDITIONAL_WRITE, object())


class TestTwinHierarchy:
    """Recurrence guard for the API/common exception pairing.

    ``RateLimitError`` was a *sibling* of its common counterpart rather than
    a subclass, so ``except`` on the common name silently missed it. Nothing
    tested the family as a family, so the divergence survived alongside four
    correct twins. These tests are that missing check.
    """

    @pytest.mark.parametrize(
        ("name", "expected_base"),
        sorted(_EXPECTED_COMMON_BASE.items()),
        ids=sorted(_EXPECTED_COMMON_BASE),
    )
    def test_declared_common_base(self, name, expected_base):
        """Each API exception subclasses the common class it should.

        Both branches assert. Skipping the API-only ones would make the
        cheapest possible table entry — the one a new class gets when nobody
        decides — also the one nothing verifies, so a class that *should*
        have a common base could be waved through by recording that it
        doesn't.
        """
        cls = _api_exception_classes()[name]

        if isinstance(expected_base, str):
            inherited = [
                base.__name__
                for base in cls.__mro__
                if base.__module__ == common_exceptions.__name__ and base is not DataknobsError
            ]
            assert not inherited, (
                f"{name} is recorded as API-only ({expected_base}), but it "
                f"inherits {', '.join(inherited)} from the common hierarchy. "
                f"Either record that base in _EXPECTED_COMMON_BASE or drop it "
                f"from the class"
            )
            return

        assert issubclass(cls, expected_base), (
            f"{name} must subclass {expected_base.__module__}."
            f"{expected_base.__name__} so that catching the common type "
            f"also catches the API variant"
        )

    def test_every_api_exception_is_accounted_for(self):
        """A newly added exception must declare its intent above.

        Without this, a future class could be added with no common base and
        no test would notice — which is exactly how the original divergence
        went unremarked.
        """
        assert set(_api_exception_classes()) == set(_EXPECTED_COMMON_BASE)

    @pytest.mark.parametrize(
        "name", sorted(_EXPECTED_COMMON_BASE), ids=sorted(_EXPECTED_COMMON_BASE)
    )
    def test_same_named_common_class_is_always_a_base(self, name):
        """Derived rule: a same-named common class must be inherited.

        Independent of the table above, so a class added with the wrong
        expectation recorded there is still caught.
        """
        cls = _api_exception_classes()[name]
        twin = getattr(common_exceptions, name, None)

        if twin is None:
            pytest.skip(f"{name} has no same-named counterpart in common")
        assert issubclass(cls, twin), (
            f"dataknobs_bots.api.exceptions.{name} shares a name with "
            f"dataknobs_common.exceptions.{name} but does not subclass it"
        )

    def test_api_error_still_precedes_the_common_base_in_the_mro(self):
        """``APIError`` must win handler dispatch over the common base.

        Starlette resolves handlers by walking ``type(exc).__mro__`` and
        taking the first registered match, so this ordering is what keeps
        ``api_error_handler`` responsible for the API variants.
        """
        for name, expected_base in _EXPECTED_COMMON_BASE.items():
            if isinstance(expected_base, str):
                continue
            mro = _api_exception_classes()[name].__mro__
            assert mro.index(APIError) < mro.index(expected_base), name

    @pytest.mark.parametrize(
        "name", sorted(_EXPECTED_COMMON_BASE), ids=sorted(_EXPECTED_COMMON_BASE)
    )
    def test_api_error_precedes_dataknobs_error_in_the_mro(self, name):
        """``DataknobsError`` is registered too, and must not win.

        Every API exception is a ``DataknobsError``, so once that type has a
        handler of its own the two registrations compete for all seven
        classes. ``APIError`` preceding it is the whole reason they still
        reach ``api_error_handler`` and keep their per-class status codes
        instead of resolving through the policy table.
        """
        mro = _api_exception_classes()[name].__mro__
        assert mro.index(APIError) < mro.index(DataknobsError), name


class TestExceptionHandlerDispatch:
    """The registered handlers still own the API exception variants."""

    async def test_rate_limit_error_still_routes_to_api_error_handler(self):
        """Widening the base list must not change the HTTP response.

        ``RateLimitError`` gained ``OperationError`` and a second route to
        ``DataknobsError`` in its MRO. Neither is registered, and ``APIError``
        still precedes them, so the response is byte-identical to before.
        """
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/throttled")
        async def throttled():
            raise RateLimitError("Too many requests", retry_after=30.0)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/throttled")

        assert response.status_code == 429
        body = response.json()
        assert body["error"] == "RateLimitError"
        assert body["message"] == "Too many requests"
        assert body["detail"] == {"retry_after": 30.0}

    @pytest.mark.parametrize(
        ("retry_after", "expected_header"),
        [
            (30.0, "30"),
            (0.0, "0"),
            # delay-seconds is an integer, so a fractional wait must round
            # *up* — rounding down tells the client to retry while still
            # throttled, which earns it a second 429.
            (2.4, "3"),
            # Nonsense in, nothing dangerous out: a negative wait is clamped
            # rather than emitted, since `Retry-After: -5` is unparseable and
            # a client may fall back to retrying immediately.
            (-5.0, "0"),
        ],
        ids=["whole", "zero", "fractional-rounds-up", "negative-clamped"],
    )
    # NOTE: the non-finite and non-numeric cases are NOT in this list —
    # `math.ceil` raises on them rather than producing a header, so there is
    # no `expected_header` to assert. They live in the class below.
    async def test_a_429_carries_a_retry_after_header(self, retry_after, expected_header):
        """The retry hint must reach the client, not just the response body.

        ``detail.retry_after`` is DataKnobs' own JSON shape; nothing outside
        this codebase knows to look for it. ``Retry-After`` is the field HTTP
        clients, proxies, and SDK retry policies already read, and RFC 6585
        says a 429 SHOULD carry it. Without the header the server knows
        exactly how long the client should wait and declines to say so in the
        one place it would be acted on automatically.

        RFC 7231 defines the value as delay-seconds — a non-negative integer
        — so the float the rate limiters report has to be converted, not
        stringified.
        """
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/throttled")
        async def throttled():
            raise RateLimitError("Too many requests", retry_after=retry_after)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/throttled")

        assert response.status_code == 429
        assert response.headers["retry-after"] == expected_header

    async def test_an_error_without_a_retry_hint_sends_no_header(self):
        """``Retry-After`` is omitted rather than guessed at.

        Every API error flows through the same handler, so the header has to
        be driven by the exception actually carrying a hint. Emitting a
        default would assert a wait the server never computed.
        """
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/missing")
        async def missing():
            raise BotNotFoundError("unknown-bot")

        @app.get("/unhinted")
        async def unhinted():
            raise RateLimitError("Too many requests")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            assert "retry-after" not in (await client.get("/missing")).headers
            assert "retry-after" not in (await client.get("/unhinted")).headers

    @pytest.mark.parametrize(
        ("exc", "status", "message", "detail"),
        [
            (APIError("boom"), 500, "boom", {}),
            (
                BotNotFoundError("bot-1"),
                404,
                "Bot with ID 'bot-1' not found",
                {"bot_id": "bot-1"},
            ),
            (
                # Masked by its own `client_safe = False`, not by the table --
                # which is the distinction this case now also pins. Reaching
                # `api_error_handler` is still what is under test: the table
                # would give it the same 500 but a different `error` code.
                BotCreationError("bot-1", "disk full"),
                500,
                api_exceptions.MASKED_MESSAGE,
                {},
            ),
            (
                ConversationNotFoundError("conv-1"),
                404,
                "Conversation with ID 'conv-1' not found",
                {"conversation_id": "conv-1"},
            ),
            (
                ValidationError("bad field", {"field": "name"}),
                422,
                "bad field",
                {"field": "name"},
            ),
            (
                ConfigurationError("bad config", config_key="llm.model"),
                500,
                "bad config",
                {"config_key": "llm.model"},
            ),
            (
                RateLimitError("slow down", retry_after=5.0),
                429,
                "slow down",
                {"retry_after": 5.0},
            ),
        ],
        ids=[
            "APIError",
            "BotNotFoundError",
            "BotCreationError",
            "ConversationNotFoundError",
            "ValidationError",
            "ConfigurationError",
            "RateLimitError",
        ],
    )
    async def test_every_api_class_still_reaches_api_error_handler(
        self, exc, status, message, detail
    ):
        """Registering ``DataknobsError`` must not take any API class with it.

        Every one of the seven is a ``DataknobsError``, so the new handler
        competes for all of them; ``APIError`` precedes it in each MRO, which
        is what this asserts through the response rather than through the MRO.

        One row would visibly change if that regressed: ``APIError`` resolves
        through the table to a *masked* 500, so its real message would
        disappear. The other six produce the same body either way. Five do so
        by design — the table gives the common types the same statuses the API
        twins already used — and they pin that agreement.

        ``BotCreationError`` is the sixth, and it agrees for a different
        reason: it is masked here by its own ``client_safe = False``, and the
        table would mask it too via ``OperationError``. So this case no longer
        detects the regression for it, and the MRO assertions in
        :class:`TestTwinHierarchy` are what stand behind that class now.
        """
        response = await _response_for(exc)

        assert response.status_code == status
        body = response.json()
        assert body["message"] == message
        assert body["detail"] == detail
        assert body["error"] == type(exc).__name__
        assert "timestamp" in body


class TestApiErrorDisclosure:
    """The API family carries its own disclosure gate.

    ``APIError`` precedes every common base in these classes' MROs, so they
    reach ``api_error_handler`` and the policy table's ``client_safe`` bit
    never governs them. That exemption is deliberate — these are the one
    family designed for the HTTP boundary, carrying a per-instance
    ``status_code`` and a public, overridable ``to_dict()``. The
    ``client_safe`` class attribute is the same bit for this family, decided
    per class rather than per type-table row.
    """

    async def test_bot_creation_error_does_not_echo_its_reason(self):
        """``reason`` is free text and the taught pattern filled it from ``str(e)``.

        ``BotCreationError`` is the only class in the family whose entire
        payload is unstructured — the others put the authored part in
        ``detail``/``config_key`` and the caller's own input in the message.
        Bots are built lazily on the request path (a ``get_bot`` cache miss
        calls ``DynaBot.from_config``), and the tool and middleware factories
        wrap ``except Exception`` into a message ending in ``{e}``, so a tool
        whose constructor opens a database rendered the driver's error —
        connection URL and credentials included — into the response body.
        """
        exc = BotCreationError(
            "bot-1",
            "Could not parse SQLAlchemy URL from string "
            "'postgresql://svc:hunter2@db.internal:5432/prod'",
        )

        response = await _response_for(exc)

        assert response.status_code == 500
        assert "hunter2" not in response.text
        body = response.json()
        assert body["message"] == api_exceptions.MASKED_MESSAGE
        assert body["detail"] == {}
        assert body["error"] == "BotCreationError"

    async def test_a_masked_api_error_still_logs_the_diagnostic(self, caplog):
        """Masking moves the diagnostic to the log; it must not delete it.

        ``api_error_handler`` logged nothing at all before this gate existed,
        so without a log line here the reason would simply vanish.
        """
        import logging

        exc = BotCreationError("bot-1", "disk full on /var/lib/bots")

        with caplog.at_level(logging.ERROR, logger=api_exceptions.__name__):
            await _response_for(exc)

        assert "disk full on /var/lib/bots" in caplog.text

    @pytest.mark.parametrize(
        "factory,expected_message",
        [
            (lambda: BotNotFoundError("bot-1"), "Bot with ID 'bot-1' not found"),
            (
                lambda: ConversationNotFoundError("conv-1"),
                "Conversation with ID 'conv-1' not found",
            ),
            (lambda: ValidationError("bad field"), "bad field"),
            (lambda: ConfigurationError("bad config"), "bad config"),
            (lambda: RateLimitError("slow down"), "slow down"),
            (lambda: APIError("plain"), "plain"),
        ],
        ids=[
            "BotNotFoundError",
            "ConversationNotFoundError",
            "ValidationError",
            "ConfigurationError",
            "RateLimitError",
            "APIError",
        ],
    )
    async def test_every_other_api_class_stays_disclosed(self, factory, expected_message):
        """Only ``BotCreationError`` is gated; the gate must not spread.

        ``client_safe`` defaults to ``True`` on ``APIError`` precisely so that
        adding it changes nothing for the other six — or for a consumer's own
        subclass, which would otherwise start masking silently.
        """
        response = await _response_for(factory())

        assert response.json()["message"] == expected_message

    async def test_a_consumer_subclass_defaults_to_disclosed(self):
        """A consumer writing an ``APIError`` subclass wrote it to be shown."""

        class TenantQuotaError(APIError):
            def __init__(self, tenant: str):
                super().__init__(
                    message=f"Quota exhausted for tenant '{tenant}'",
                    status_code=402,
                    detail={"tenant": tenant},
                )

        response = await _response_for(TenantQuotaError("acme"))

        assert response.status_code == 402
        body = response.json()
        assert body["message"] == "Quota exhausted for tenant 'acme'"
        assert body["detail"] == {"tenant": "acme"}

    def test_the_gate_is_declared_on_the_base_not_only_the_gated_class(self):
        """Reading ``APIError`` must answer "is this disclosed?" for the family.

        A gate that existed only as an attribute on ``BotCreationError`` would
        make the other six answer by ``getattr`` default, which is the shape
        that lets a typo silently disclose.
        """
        assert APIError.client_safe is True
        assert BotCreationError.client_safe is False


def _common_exception_names() -> list[str]:
    """Exception classes exported by ``dataknobs_common.exceptions``.

    The filter matters: that module's ``__all__`` held nothing but exception
    classes until ``DottedPathReason`` — the ``reason`` vocabulary that lives
    beside the error carrying it — was added. Without the filter, the guards
    below demand an HTTP status for an enum, which is not a question with an
    answer. Filtering on ``BaseException`` keeps their teeth: a new *error*
    type still has to be given a row.
    """
    return [
        name
        for name in common_exceptions.__all__
        if isinstance(getattr(common_exceptions, name), type)
        and issubclass(getattr(common_exceptions, name), BaseException)
    ]


#: Types in the common hierarchy whose constructor is not the uniform
#: ``(message, context=...)``. Each builds an instance carrying the same
#: ``"diagnostic detail"`` message and a ``probe`` context key, so the
#: end-to-end disclosure assertions read identically whichever way the
#: instance was made.
_NON_UNIFORM_CTORS: dict[str, Callable[[type], BaseException]] = {
    "DottedPathError": lambda cls: cls(
        "diagnostic detail",
        ref="a.module:name",
        reason=common_exceptions.DottedPathReason.MALFORMED,
        probe="value",
    ),
    "DottedPathTypeError": lambda cls: cls(
        "diagnostic detail",
        ref="a.module:name",
        expected=ValueError,
        probe="value",
    ),
}


class TestErrorPolicyTable:
    """The policy table covers the hierarchy and resolves by MRO."""

    @pytest.mark.parametrize("name", sorted(_EXPECTED_POLICY), ids=sorted(_EXPECTED_POLICY))
    def test_table_matches_the_declared_contract(self, name):
        """Each entry has the status and disclosure recorded above."""
        # Resolved against the package namespace, not ``.exceptions``: a row
        # may name a type that lives in the module it serves rather than in the
        # exception hierarchy's own module.
        cls = getattr(dataknobs_common, name)
        status, message, context = _EXPECTED_POLICY[name]

        assert cls in DEFAULT_ERROR_POLICY, f"{name} has no policy entry"
        assert DEFAULT_ERROR_POLICY[cls] == ErrorPolicy(status, message, context)

    def test_every_common_exception_has_a_policy(self):
        """A new common error type must be given a row, not defaulted.

        Without this, adding one to ``dataknobs_common.exceptions`` would
        silently resolve it through ``DataknobsError`` to a masked 500 — the
        exact behaviour this handler exists to stop, reintroduced one type at
        a time.
        """
        missing = [
            name
            for name in _common_exception_names()
            if getattr(common_exceptions, name) not in DEFAULT_ERROR_POLICY
        ]
        assert not missing, (
            f"{', '.join(missing)} in dataknobs_common.exceptions.__all__ but "
            f"absent from DEFAULT_ERROR_POLICY — add a row, deciding its "
            f"status and whether its message and context are client-safe"
        )

    def test_no_row_ships_without_being_recorded(self):
        """The direction the guard was missing.

        The two tests above run outward from
        ``dataknobs_common.exceptions.__all__``, so they catch a new common
        type left without a row. Neither catches the reverse: a row added for
        a type from anywhere else ships an undocumented status, because both
        the contract above and the published table are keyed off that
        ``__all__``. Not hypothetical — the ``InvalidTransitionError`` row
        (whose type lives in ``dataknobs_common.transitions``, not in the
        exception module) landed through exactly this gap, and the whole
        module stayed green.
        """
        unrecorded = sorted(
            cls.__name__ for cls in DEFAULT_ERROR_POLICY if cls.__name__ not in _EXPECTED_POLICY
        )
        assert not unrecorded, (
            f"{', '.join(unrecorded)} has a DEFAULT_ERROR_POLICY row but is "
            f"absent from _EXPECTED_POLICY — record it there and in the "
            f"published table, so a deployment can see the status it ships"
        )

    def test_the_expected_policy_table_covers_the_common_hierarchy(self):
        """The test's own expectations must not go stale either.

        A superset rather than an equality: a row may also be declared for a
        type outside ``dataknobs_common.exceptions``, which the test above
        holds to the same recording requirement.
        """
        assert set(_EXPECTED_POLICY) >= set(_common_exception_names())

    def test_the_documented_table_matches_the_declared_contract(self):
        """The published table is the third copy, and the one nobody runs.

        ``MULTI_TENANT.md`` states each type's status and whether it is
        disclosed, and that section is transcluded into the site, so it is
        what a deployment reads when deciding whether a route can be public.
        Flipping a row in the source without editing it would leave the
        documentation confidently wrong about a security property.

        Chained through ``_EXPECTED_POLICY`` rather than read straight off
        ``DEFAULT_ERROR_POLICY``: that constant is asserted against the
        shipped table one case above, so tying the prose to the same
        hand-written contract makes all three agree rather than making the
        docs agree with whatever the code happens to say.
        """
        assert _documented_policy() == _EXPECTED_POLICY

    def test_the_row_count_in_the_module_comment_is_right(self):
        """A number written in prose beside a table it describes goes stale.

        The comment over ``DEFAULT_ERROR_POLICY`` said eleven while the table
        held twelve, and repeated the wrong figure twice more — the row added
        for ``InvalidTransitionError`` did not come with a re-count. Nothing
        depends on the number, which is exactly why nobody noticed; asserting
        it is what makes the next row's author notice.
        """
        raw = Path(api_exceptions.__file__).read_text(encoding="utf-8")
        # Comment continuation lines carry a `#:` marker, so a phrase that
        # wraps is not contiguous in the file. Flatten before matching.
        source = " ".join(raw.replace("#:", " ").split())
        count = len(DEFAULT_ERROR_POLICY)
        spelled = {11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen"}[count]

        assert f"{spelled.capitalize()} entries govern" in source, (
            f"the table has {count} rows; update the comment over "
            f"DEFAULT_ERROR_POLICY, which spells a different number"
        )
        assert f"cover the {spelled}" in source

        # The split the comment claims: all but one come from the exceptions
        # module, and the odd one out is named.
        from_exceptions = [
            t for t in DEFAULT_ERROR_POLICY if t.__module__ == common_exceptions.__name__
        ]
        assert len(from_exceptions) == count - 1
        from dataknobs_common.transitions import InvalidTransitionError

        assert InvalidTransitionError in DEFAULT_ERROR_POLICY
        assert InvalidTransitionError not in from_exceptions

    def test_dataknobs_error_is_the_terminal_entry(self):
        """The fallback is a row, so a consumer can override it."""
        assert DEFAULT_ERROR_POLICY[DataknobsError] == ErrorPolicy(500, False)

    def test_resolution_walks_the_mro(self):
        """A subclass with no row of its own inherits its nearest ancestor's."""
        from dataknobs_data.exceptions import RecordNotFoundError

        assert resolve_error_policy(RecordNotFoundError("rec-1")) == ErrorPolicy(404, True, False)

    def test_a_subclass_row_beats_its_base(self):
        """``RateLimitError`` is an ``OperationError``; 429 must win over 500.

        Resolution is by MRO, not by dict order, so this holds however the
        table happens to be sorted in the source.
        """
        assert resolve_error_policy(CommonRateLimitError("nope")) == ErrorPolicy(429, True, True)

    def test_a_table_without_a_terminal_entry_fails_closed(self):
        """A table that omits ``DataknobsError`` masks, not discloses.

        Reachable only by calling this function directly — the registration
        path merges over the defaults — which is why the guard is here rather
        than trusted to be unreachable. Spelled with the keyword to exercise
        the parameter name: ``table`` *replaces* the defaults, where
        ``register_exception_handlers``' ``error_policy`` merges over them,
        and this case is exactly what the difference costs.
        """
        assert resolve_error_policy(CommonValidationError("bad"), table={}) == ErrorPolicy(
            500, False
        )


class TestDataknobsErrorHandling:
    """DataKnobs' own errors reach the client with a meaningful status."""

    @pytest.mark.parametrize(
        "name",
        sorted(_common_exception_names()),
        ids=sorted(_common_exception_names()),
    )
    async def test_each_common_type_end_to_end(self, name):
        """Raised from a route, each type returns its status and disclosure.

        Before this handler existed, every row here returned
        ``500 / "An unexpected error occurred"``.

        Driven off the common hierarchy rather than off ``_EXPECTED_POLICY``,
        because it relies on that hierarchy's uniform
        ``(message, context=...)`` constructor. A row declared for a type
        outside it — `InvalidTransitionError`, whose constructor takes the
        transition — is covered by its own case instead.

        Two types *inside* the hierarchy no longer take that constructor:
        the dotted-path pair require ``ref=`` and their own discriminator,
        following ``PackResolutionError``. They are built by
        ``_NON_UNIFORM_CTORS`` rather than excluded, so every common error
        type still gets an end-to-end case — an exclusion list is the thing
        that quietly grows until the guard covers half the hierarchy.
        """
        status, disclose_message, disclose_context = _EXPECTED_POLICY[name]
        cls = getattr(common_exceptions, name)
        build = _NON_UNIFORM_CTORS.get(name)
        exc = (
            build(cls)
            if build is not None
            else cls("diagnostic detail", context={"probe": "value"})
        )

        response = await _response_for(exc)

        assert response.status_code == status
        body = response.json()
        assert body["error"] == name
        if disclose_message:
            assert body["message"] == "diagnostic detail"
        else:
            assert body["message"] == api_exceptions.MASKED_MESSAGE
        if disclose_context:
            assert body["detail"]["probe"] == "value"
        else:
            assert body["detail"] == {}

    async def test_the_configuration_diagnostic_reaches_the_log_by_default(self, caplog):
        """The case the gap was reported for, and why it is masked by default.

        A config diagnostic is generated by DataKnobs and was then discarded by
        DataKnobs one layer later, leaving the deployment a 500 saying nothing
        about which key was wrong. It is no longer discarded — but it goes to
        the log rather than the response, because ``ConfigurationError`` is
        also where the funnels that wrap third-party constructor and import
        failures land, and a deployment cannot audit its consumers' raise
        sites. See the opt-in below for the other half.
        """
        import logging

        exc = CommonConfigurationError(
            "embedding: no variant registered for 'ollamaa'",
            context={"config_key": "embedding"},
        )

        with caplog.at_level(logging.ERROR, logger=api_exceptions.__name__):
            response = await _response_for(exc)

        assert response.status_code == 500
        body = response.json()
        assert body["message"] == api_exceptions.MASKED_MESSAGE
        assert body["detail"] == {}

        assert "no variant registered for 'ollamaa'" in caplog.text
        assert "config_key" in caplog.text

    async def test_the_configuration_diagnostic_is_one_line_from_disclosure(self):
        """Masking is the default, not the only option.

        A deployment whose config routes are not public — an admin API, an
        internal control plane — turns the diagnostic back on per app, and the
        override merges over the defaults rather than replacing them.
        """
        response = await _response_for(
            CommonConfigurationError(
                "embedding: no variant registered for 'ollamaa'",
                context={"config_key": "embedding"},
            ),
            error_policy={CommonConfigurationError: ErrorPolicy(500, True, True)},
        )

        assert response.status_code == 500
        body = response.json()
        assert body["message"] == "embedding: no variant registered for 'ollamaa'"
        assert body["detail"] == {"config_key": "embedding"}

    @pytest.mark.parametrize(
        ("factory", "status", "expect_message"),
        [
            # data: NotFoundError -> 404
            (_record_not_found, 404, "Record with ID 'rec-1' not found"),
            # llm: ValidationError -> 422
            (_context_length_exceeded, 422, "prompt too long"),
            # common, defined outside exceptions.py: OperationError -> masked 500
            (_capability_not_supported, 500, None),
        ],
        ids=["data", "llm", "common-capabilities"],
    )
    async def test_subclasses_outside_the_table_resolve_by_mro(
        self, factory, status, expect_message
    ):
        """More than forty subclasses live outside ``common/exceptions.py``.

        None of them appear in the table, and an exact-type lookup would 500
        every one. Four packages' worth are checked here; the MRO walk is what
        makes the rest work too.
        """
        response = await _response_for(factory())

        assert response.status_code == status
        body = response.json()
        if expect_message is None:
            assert body["message"] == api_exceptions.MASKED_MESSAGE
        else:
            assert body["message"] == expect_message

    async def test_a_common_subclass_resolves_to_the_configuration_row(self):
        """``PackResolutionError`` is a ``ConfigurationError`` defined elsewhere.

        Driven through an override rather than the default, because
        ``ConfigurationError``, ``OperationError`` and ``DataknobsError`` are
        all masked 500s now — so against the defaults this class would look
        identical whichever of the three it resolved to, and the case would
        pin nothing. Disclosing *only* the configuration row makes the
        resolution observable again: a regression that sent this class to
        ``OperationError`` or the terminal fallback would leave it masked.
        """
        response = await _response_for(
            _pack_resolution_error(),
            error_policy={CommonConfigurationError: ErrorPolicy(500, True, True)},
        )

        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "PackResolutionError"
        assert body["message"] == "pack 'audit' is not registered"

    async def test_a_not_found_keeps_its_message_and_drops_its_context(self):
        """The case that split disclosure into two bits.

        ``Registry.get`` raises with ``available_keys`` in ``context`` — the
        whole registered keyspace, which is a "did you mean" for a library
        caller and an inventory listing for an HTTP one. The message is the
        caller's own key echoed back, so it discloses nothing new. One bit
        forced a choice between publishing the inventory and returning a 404
        that will not say what was not found; two bits do not.
        """
        from dataknobs_common.registry import Registry

        registry: Registry[str] = Registry("tools")
        registry.register("internal_admin_export", "x")
        registry.register("billing_reconciliation", "y")

        with pytest.raises(CommonNotFoundError) as caught:
            registry.get("nope")

        response = await _response_for(caught.value)

        assert response.status_code == 404
        assert response.json()["message"] == "Item not found: nope"
        assert response.json()["detail"] == {}
        for key in ("internal_admin_export", "billing_reconciliation"):
            assert key not in response.text, f"{key!r} leaked into response"

    async def test_a_timeout_keeps_its_message_and_drops_its_context(self):
        """``TimeoutError``'s own docstring example puts the SQL query in context.

        A raiser following the documented shape put the query in a disclosed
        504. The message — that something timed out, and after how long if the
        raiser said so — is the actionable half and stays.
        """
        response = await _response_for(
            common_exceptions.TimeoutError(
                "Database query timed out",
                context={"query": "SELECT * FROM tenant_billing", "timeout_seconds": 30},
            )
        )

        assert response.status_code == 504
        assert response.json()["message"] == "Database query timed out"
        assert response.json()["detail"] == {}
        assert "tenant_billing" not in response.text

    async def test_a_masked_error_discloses_neither_message_nor_context(self):
        """Security assertion: a masked type leaks nothing to the caller.

        ``ResourceError`` is raised on failed connects, so its message is the
        one most likely to carry a DSN — credentials included. This fails
        loudly if someone flips that row to ``client_safe=True``.
        """
        dsn = "postgresql://svc_user:hunter2@db.internal:5432/prod"
        response = await _response_for(
            CommonResourceError(
                f"could not connect: {dsn}",
                context={"dsn": dsn, "password": "hunter2"},
            )
        )

        assert response.status_code == 503
        body = response.json()
        assert body["message"] == api_exceptions.MASKED_MESSAGE
        assert body["detail"] == {}
        for secret in ("hunter2", "svc_user", "db.internal", dsn):
            assert secret not in response.text, f"{secret!r} leaked into response"

    async def test_a_non_dataknobs_exception_is_untouched(self):
        """The catch-all still owns everything that is not a DataKnobs error.

        ``raise_app_exceptions=False`` because this one is *supposed* to reach
        the catch-all, and reaching it means the ASGI server sees the failure
        too — see :class:`TestAsgiPropagation`.
        """
        response = await _response_for(ValueError("nope"), raise_app_exceptions=False)

        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "InternalServerError"
        assert body["message"] == api_exceptions.MASKED_MESSAGE
        assert body["detail"] == {"exception_type": "ValueError"}


class TestResponseEncoding:
    """A value the JSON encoder rejects must not undo the handler's work.

    ``context`` is a free ``dict[str, Any]`` and raise sites fill it with
    whatever the failure was about — a ``Path``, a timeout that came back
    infinite, an object. Starlette renders with
    ``json.dumps(..., allow_nan=False)``, so a value it cannot encode raises
    *inside* the handler; Starlette's error middleware then catches that and
    the caller gets exactly the ``500 / "An unexpected error occurred"`` this
    handler exists to replace, with the real status and message lost.
    """

    async def test_a_path_in_context_does_not_become_a_500(self):
        """A ``Path`` is the likeliest way this happens.

        Anything that reads a file — a config loader, a knowledge backend —
        has one to hand when it raises, and it is the natural thing to put in
        ``context``.

        Driven with a type whose context is disclosed, since a masked one
        would pass whether the encoder coped or not.
        """
        response = await _response_for(
            CommonValidationError(
                "bot config is malformed",
                context={"path": Path("/var/lib/bots/acme.yaml")},
            )
        )

        assert response.status_code == 422
        assert response.json()["detail"]["path"] == "/var/lib/bots/acme.yaml"

    async def test_a_non_finite_float_does_not_become_a_500(self):
        """``allow_nan=False`` rejects the infinities, not just NaN.

        A limit or a timeout is a plausible source: ``float("inf")`` is how a
        few config paths spell "no ceiling".
        """
        response = await _response_for(
            CommonValidationError("limit out of range", context={"limit": float("inf")})
        )

        assert response.status_code == 422
        assert response.json()["detail"]["limit"] == "inf"

    async def test_an_api_error_detail_is_encoded_too(self):
        """The ``to_dict()`` path needs the same treatment as the table path.

        Encoding in the response builder rather than in one handler is what
        makes that true, including for a consumer's overridden ``to_dict``.
        """
        from pathlib import Path

        response = await _response_for(
            ValidationError("bad path", detail={"path": Path("/etc/bots.yaml")})
        )

        assert response.status_code == 422
        assert response.json()["detail"]["path"] == "/etc/bots.yaml"

    async def test_an_unencodable_object_is_not_expanded_into_its_attributes(self):
        """Not ``vars(value)`` — this is a disclosure decision.

        ``fastapi.encoders.jsonable_encoder`` falls back to ``dict(obj)`` and
        then ``vars(obj)``, which would put an object's whole attribute dict
        into a response body the raiser only meant to carry the object.
        """

        class _Connection:
            def __init__(self):
                self.dsn = "postgresql://svc:hunter2@db.internal/prod"

            def __str__(self):
                return "<connection to prod>"

        response = await _response_for(
            CommonValidationError("row rejected", context={"conn": _Connection()})
        )

        assert response.status_code == 422
        assert "hunter2" not in response.text
        assert "dsn" not in response.text

    async def test_a_third_party_objects_repr_is_not_disclosed(self):
        """``str(obj)`` is the object's choice, and it is not ours to publish.

        The previous rule — coerce anything unknown with ``str`` — was argued
        from a ``StructuredConfig``, whose repr redacts its own secrets. That
        generalised from one cooperative type. The objects a raise site
        actually has to hand at the moment of failure do the opposite: a
        SQLAlchemy ``Engine`` renders as
        ``Engine(postgresql://svc:hunter2@db/prod)`` and a psycopg2
        connection as ``<connection object ...; dsn: '... password=hunter2
        ...'>``. Both put the credential in the repr *by design*, because it
        is a debugging aid — for a log, not a response body.

        Five rows in the default policy disclose ``context``, so this is the
        live path, not a hypothetical one.
        """

        class Engine:
            def __repr__(self):  # what SQLAlchemy actually does
                return "Engine(postgresql://svc:hunter2@db.internal:5432/prod)"

        response = await _response_for(
            CommonValidationError("write rejected", context={"bind": Engine()})
        )

        assert response.status_code == 422
        assert "hunter2" not in response.text
        # The key survives, and the caller is told what kind of thing it was.
        assert response.json()["detail"]["bind"] == "<Engine>"

    async def test_the_types_whose_text_is_their_value_still_render(self):
        """Fail-closed must not mean uninformative.

        A ``Path``, a ``UUID``, a timestamp, a ``Decimal``, an enum member —
        for these ``str`` *is* the value, and they are what raise sites
        legitimately put in ``context``. Withholding them would trade a real
        disclosure for a real loss of diagnostic.
        """
        from decimal import Decimal
        from uuid import UUID

        response = await _response_for(
            CommonValidationError(
                "record is invalid",
                context={
                    "path": Path("/etc/bots.yaml"),
                    "id": UUID("12345678-1234-5678-1234-567812345678"),
                    "amount": Decimal("10.25"),
                },
            )
        )

        detail = response.json()["detail"]
        assert detail["path"] == "/etc/bots.yaml"
        assert detail["id"] == "12345678-1234-5678-1234-567812345678"
        assert detail["amount"] == "10.25"

    async def test_a_value_whose_str_raises_does_not_become_a_500(self):
        """The coercion is the last thing standing between here and the 500.

        ``__str__`` is arbitrary code. An object that raises when rendered —
        a lazy proxy whose backing resource has closed, a dataclass with a
        broken ``__repr__`` — would take the whole response with it.
        """

        class Hostile:
            def __str__(self):
                raise RuntimeError("cannot render")

            __repr__ = __str__

        response = await _response_for(
            CommonValidationError("row rejected", context={"thing": Hostile()})
        )

        assert response.status_code == 422
        assert response.json()["message"] == "row rejected"

    async def test_a_mapping_whose_items_raises_does_not_become_a_500(self):
        """``isinstance(value, Mapping)`` does not promise ``items()`` works.

        A dict subclass over a closed cursor, or a lazily-populated config
        proxy, satisfies the check and then raises when walked.
        """

        class HostileMapping(dict):
            def items(self):
                raise RuntimeError("backing store is gone")

        response = await _response_for(
            CommonValidationError("row rejected", context={"m": HostileMapping()})
        )

        assert response.status_code == 422
        assert response.json()["message"] == "row rejected"

    async def test_a_self_referential_context_does_not_become_a_500(self):
        """``json.dumps`` raises on a cycle; an unbounded walk would hang.

        The depth bound handles both, and is why the walk is bounded rather
        than merely type-aware.
        """
        node: dict[str, object] = {"kind": "stage"}
        node["parent"] = node

        response = await _response_for(
            CommonValidationError("stage is invalid", context={"node": node})
        )

        assert response.status_code == 422
        assert response.json()["detail"]["node"]["kind"] == "stage"


class TestHandlerLogging:
    """Severity follows the status class; disclosure decides only content.

    Every handled error is logged, because the handlers absorb errors that
    used to propagate to the ASGI server. What level they log at is therefore
    load-bearing: a 404 is the caller's problem and a routine outcome of
    serving traffic, and logging one at ``warning`` makes a working service
    look like a failing one — the 5xx that need attention get buried in
    not-founds.
    """

    def _records(self, caplog):
        return [r for r in caplog.records if r.name == api_exceptions.__name__]

    async def test_a_4xx_is_logged_at_info(self, caplog):
        """The routine case: a caller asked for something that is not there."""
        import logging

        with caplog.at_level(logging.INFO, logger=api_exceptions.__name__):
            await _response_for(CommonNotFoundError("no such bot"))

        assert [r.levelname for r in self._records(caplog)] == ["INFO"]

    async def test_a_5xx_is_logged_at_error_with_the_traceback(self, caplog):
        """The server's own problem: worth ERROR, and worth the traceback.

        ``exc_info`` is what distinguishes ``logger.exception`` from
        ``logger.error`` here, and it is the half that makes the line
        actionable — Starlette calls handlers from inside its ``except``
        block, so the traceback is live.
        """
        import logging

        with caplog.at_level(logging.INFO, logger=api_exceptions.__name__):
            await _response_for(CommonResourceError("connect failed"))

        records = self._records(caplog)
        assert [r.levelname for r in records] == ["ERROR"]
        assert records[0].exc_info is not None

    async def test_a_disclosed_5xx_is_still_logged_at_error(self, caplog):
        """Level is the status class, not the disclosure bit.

        A 504 is client-safe, and it is still a server-side fault.
        """
        import logging

        with caplog.at_level(logging.INFO, logger=api_exceptions.__name__):
            await _response_for(common_exceptions.TimeoutError("upstream timed out"))

        assert [r.levelname for r in self._records(caplog)] == ["ERROR"]

    async def test_a_masked_4xx_is_logged_above_info(self, caplog):
        """The one case where disclosure does move the level.

        Masking a 404 does not make it an incident, so this is not ``error``
        — but it is the single combination where the log is the *only* record
        of the failure, since the caller was told nothing. ``info`` is a level
        production deployments routinely filter, which would discard the
        diagnostic this design promises to relocate rather than discard. So
        ``warning`` is the floor for a masked 4xx and ``info`` still covers
        the disclosed ones.

        No default row reaches here — every masked default is a 5xx — so this
        is for a consumer override, or an ``APIError`` subclass that sets
        ``client_safe = False`` at a 4xx.
        """
        import logging

        with caplog.at_level(logging.INFO, logger=api_exceptions.__name__):
            await _response_for(
                CommonNotFoundError("no such bot", context={"bot_id": "acme"}),
                error_policy={CommonNotFoundError: ErrorPolicy(404, False)},
            )

        records = self._records(caplog)
        assert [r.levelname for r in records] == ["WARNING"]
        # Masked, so the log line is the only place the context survives.
        assert "acme" in caplog.text

    async def test_a_disclosed_4xx_stays_at_info(self, caplog):
        """The rule above must not quietly promote ordinary traffic.

        A 404 the caller can read is a routine outcome of serving requests;
        logging it at ``warning`` is what made a working service look like a
        failing one.
        """
        import logging

        with caplog.at_level(logging.INFO, logger=api_exceptions.__name__):
            await _response_for(CommonNotFoundError("no such bot"))

        assert [r.levelname for r in self._records(caplog)] == ["INFO"]

    async def test_the_api_family_uses_the_same_rule(self, caplog):
        """One helper for both handlers, so the two cannot drift apart.

        ``api_error_handler`` logged nothing at all until the disclosure gate
        landed, and a second severity policy written beside the first is how
        it would end up disagreeing.
        """
        import logging

        with caplog.at_level(logging.INFO, logger=api_exceptions.__name__):
            await _response_for(BotNotFoundError("bot-1"))

        assert [r.levelname for r in self._records(caplog)] == ["INFO"]


class TestStatusesThatContradictedTheCondition:
    """Two types whose resolved status told the caller the wrong thing.

    Most subclasses inherit a sensible status from their base — a
    ``DuplicateRecordError`` really is a 409, a ``DatabaseConnectionError``
    really is a 503. These two were not: each picked its base for a reason
    unrelated to HTTP, and nothing re-read the choice from the caller's side.
    Both conditions are the *caller's* to act on, and both were rendered as
    something the caller could do nothing with.
    """

    async def test_an_invalid_transition_is_a_conflict_not_a_server_fault(self):
        """``InvalidTransitionError`` is an ``OperationError``, hence 500-masked.

        The base is right for a library: an invalid transition is a permanent
        failure, so retry logic keyed on ``OperationError`` correctly declines
        to re-attempt it. But over HTTP, "you cannot go from ``draft`` to
        ``shipped``" is the textbook 409 — the request conflicts with the
        resource's current state and would succeed in another one. It was
        answered with ``500 / "An unexpected error occurred"``, which blames
        the server for the caller's mistake and hands back nothing to fix.
        """
        response = await _response_for(
            dataknobs_common.InvalidTransitionError(
                "artifact_status", "draft", "shipped", allowed={"review"}
            )
        )

        assert response.status_code == 409

    async def test_the_allowed_targets_reach_the_caller(self):
        """The context is the remedy, so a masked row wasted it."""
        response = await _response_for(
            dataknobs_common.InvalidTransitionError(
                "artifact_status", "draft", "shipped", allowed={"review"}
            )
        )
        body = response.json()

        assert body["detail"]["allowed"] == ["review"]
        assert body["detail"]["current_status"] == "draft"
        assert "shipped" in body["message"]

    async def test_an_open_breaker_is_unavailable_not_conflicting(self):
        """``CircuitBreakerError`` was a ``ConcurrencyError``, hence 409.

        409 says the request conflicts with the resource's current state, so a
        client is told to change the request. Nothing about the request is
        wrong — the dependency is down and we declined to call it. The fix is
        in the FSM package, where the base class was the thing that was wrong;
        this pins the boundary behaviour it produces.
        """
        from dataknobs_fsm.core.exceptions import CircuitBreakerError

        response = await _response_for(CircuitBreakerError(wait_time=12.0))

        assert response.status_code == 503

    async def test_an_open_breaker_tells_the_caller_when_to_come_back(self):
        """The wait is a flow-control signal, so it survives the masked row.

        ``ResourceError`` is masked, so the body says nothing — correct, since
        "circuit breaker is open" is internal architecture. The header is not a
        diagnostic and is emitted regardless, which is the whole reason the
        breaker's wait had to answer to ``retry_after``.
        """
        from dataknobs_fsm.core.exceptions import CircuitBreakerError

        response = await _response_for(CircuitBreakerError(wait_time=12.0))

        assert response.headers["retry-after"] == "12"
        assert response.json()["message"] == api_exceptions.MASKED_MESSAGE


class TestARetryHintThatIsNotANumberOfSeconds:
    """A bad ``retry_after`` must cost the header, not the response.

    ``math.ceil`` raises on the non-finite floats and on anything that is not
    a real number, and it runs *inside* the handler — so Starlette's error
    middleware catches it and returns exactly the ``500 / "An unexpected error
    occurred"`` these handlers exist to replace. The status, the message, and
    the retry hint are all lost to a malformed hint about how long to wait.

    This is reachable rather than theoretical: a provider parses the value out
    of the upstream ``Retry-After`` header with ``float()``, and ``float()``
    accepts ``"inf"``, ``"Infinity"``, and ``"nan"``. Any consumer-configured
    endpoint — a self-hosted inference server, a gateway, a proxy — can send
    one, so the input is not under this codebase's control.
    """

    @pytest.mark.parametrize(
        "retry_after",
        [
            float("inf"),
            float("-inf"),
            float("nan"),
            "soon",
            None.__class__,  # a type, not a number
        ],
        ids=["inf", "-inf", "nan", "string", "non-numeric"],
    )
    async def test_the_status_survives_a_malformed_hint(self, retry_after):
        error = CommonRateLimitError("Too many requests")
        error.retry_after = retry_after

        response = await _response_for(error)

        assert response.status_code == 429
        assert response.json()["error"] == "RateLimitError"

    @pytest.mark.parametrize(
        "retry_after",
        [float("inf"), float("nan"), "soon"],
        ids=["inf", "nan", "string"],
    )
    async def test_the_header_is_omitted_rather_than_guessed(self, retry_after):
        """No header beats a wrong one.

        ``Retry-After: inf`` is unparseable, and a client that gives up on
        parsing may retry at once — the opposite of what a 429 is asking for.
        Omitting it lets the client apply its own backoff.
        """
        error = CommonRateLimitError("Too many requests")
        error.retry_after = retry_after

        response = await _response_for(error)

        assert "retry-after" not in response.headers

    async def test_a_usable_hint_is_unaffected(self):
        """The guard must not cost the header in the ordinary case."""
        error = CommonRateLimitError("Too many requests")
        error.retry_after = 30.0

        response = await _response_for(error)

        assert response.headers["retry-after"] == "30"


class TestTheFunctionsLayerReachesTheTable:
    """The FSM functions layer was outside the shared hierarchy entirely.

    ``dataknobs_fsm.functions.base`` predates the migration of that package's
    exceptions onto ``dataknobs_common`` and was left behind by it, rooted at
    a plain ``Exception``. So the 60 raise sites in the resource backends, the
    transform library, and the validators never reached
    ``dataknobs_error_handler`` at all — they fell through to the ``Exception``
    catch-all, which is where the docstring on that handler says DataKnobs'
    own errors no longer arrive. Every one of them rendered as an
    indistinguishable ``500 / InternalServerError``, whatever had happened.
    """

    async def test_the_old_shape_still_lands_on_the_catch_all(self):
        """The control, and what all of these used to do.

        A plain-``Exception`` error is not a ``DataknobsError``, so Starlette
        walks its MRO, finds no more specific handler, and lands on the
        catch-all. This is the pre-fix behaviour, kept executable so the two
        cases below are read against it rather than against a claim.
        """

        class LegacyShapedError(Exception):
            """Rooted where every functions-layer error used to be rooted."""

        response = await _response_for(
            LegacyShapedError("connect failed"), raise_app_exceptions=False
        )
        body = response.json()

        assert response.status_code == 500
        assert body["error"] == "InternalServerError"
        assert body["message"] == api_exceptions.MASKED_MESSAGE

    async def test_a_resource_failure_is_unavailable_not_a_bug(self):
        """24 raise sites across the six resource backends.

        The deployment could not reach something. That is a 503 — and masked,
        because the message on these carries whatever the driver said, which
        for a failed connect is a connection string.
        """
        from dataknobs_fsm.functions.base import ResourceError

        response = await _response_for(
            ResourceError(
                "could not connect to postgres://user:pw@db:5432", "primary-db", "acquire"
            )
        )
        body = response.json()

        assert response.status_code == 503
        assert body["error"] == "ResourceError"
        assert body["message"] == api_exceptions.MASKED_MESSAGE
        assert "postgres://" not in json.dumps(body)

    async def test_a_validation_failure_is_the_callers_to_fix(self):
        """11 raise sites in the validator library.

        The caller sent something that did not validate, which is a 422, and
        the list of what failed is the caller's own input described back — the
        one thing they need in order to retry successfully.
        """
        from dataknobs_fsm.functions.base import ValidationError

        response = await _response_for(
            ValidationError(
                "Missing required fields: name, age", ["name is required", "age is required"]
            )
        )
        body = response.json()

        assert response.status_code == 422
        assert body["error"] == "ValidationError"
        assert body["detail"]["validation_errors"] == [
            "name is required",
            "age is required",
        ]

    async def test_a_transform_failure_stays_a_server_fault(self):
        """25 raise sites, and the one where 500 was already the right answer.

        Pinned anyway: it now arrives at the table rather than the catch-all,
        so it is 500 by a policy row that says so, and is catchable as an
        ``OperationError`` by anything deciding whether to retry.
        """
        from dataknobs_fsm.functions.base import TransformError

        response = await _response_for(TransformError("transform failed"))

        assert response.status_code == 500
        assert response.json()["error"] == "TransformError"


def _chained(outer: Exception, inner: Exception) -> Exception:
    """Return *outer* raised ``from`` *inner*, so ``__cause__`` is really set."""
    try:
        raise inner
    except Exception as cause:
        try:
            raise outer from cause
        except Exception as chained:
            return chained


class TestTheCauseReachesTheLog:
    """Whatever the response drops, the log line has to keep.

    Masking and trimming both move detail out of the response on the premise
    that an operator can still get it. For a 5xx that holds — ``exception()``
    prints the whole chain. For a 4xx it did not: the level is ``info``, there
    is no traceback, and ``%s`` on the exception renders only its own message.

    That is precisely the case a library raising wrapped errors produces. A
    provider translating a vendor failure raises a ``ValidationError`` whose
    message names the provider and the status, deliberately nothing more, and
    puts the vendor's rendering on ``__cause__``. Log only the outer message at
    422 and every such failure reads identically — same line for a bad request,
    a broken gateway, and a wrong endpoint.
    """

    def _records(self, caplog):
        return [r for r in caplog.records if r.name == api_exceptions.__name__]

    async def test_a_4xx_reports_what_it_was_raised_from(self, caplog):
        import logging

        exc = _chained(
            CommonValidationError("ollama API error (HTTP 400)"),
            RuntimeError("400, url='http://ollama.internal:11434/api/chat'"),
        )

        with caplog.at_level(logging.INFO, logger=api_exceptions.__name__):
            await _response_for(exc)

        records = self._records(caplog)
        assert [r.levelname for r in records] == ["INFO"]
        assert "ollama.internal:11434" in caplog.text
        assert "RuntimeError" in caplog.text
        # And it stays out of the response — that is the whole arrangement.
        assert "ollama.internal" not in str((await _response_for(exc)).json())

    async def test_an_unchained_error_gets_no_empty_cause_field(self, caplog):
        """No ``cause=None`` noise on the overwhelmingly common case."""
        import logging

        with caplog.at_level(logging.INFO, logger=api_exceptions.__name__):
            await _response_for(CommonNotFoundError("no such bot"))

        assert "cause=" not in caplog.text

    async def test_the_api_family_reports_its_cause_too(self, caplog):
        """One helper for both handlers, so this cannot land on only one."""
        import logging

        exc = _chained(BotNotFoundError("bot-1"), RuntimeError("registry lookup exploded"))

        with caplog.at_level(logging.INFO, logger=api_exceptions.__name__):
            await _response_for(exc)

        assert "registry lookup exploded" in caplog.text


class TestAsgiPropagation:
    """Handled DataKnobs errors no longer reach the ASGI server.

    Starlette routes the ``Exception`` catch-all through
    ``ServerErrorMiddleware``, which calls the handler *and then re-raises* so
    the server sees the failure. Handlers for narrower types go through
    ``ExceptionMiddleware``, which does not. So a ``ConfigurationError`` used
    to be both returned as a 500 and propagated to uvicorn — a server-level
    error log, and a tick on whatever the deployment counts as an unhandled
    exception.

    That signal now drops for DataKnobs errors. It is the intended semantics —
    a config typo is not a server fault — but it is a monitoring behaviour
    change, so it is pinned here rather than left to be noticed in production.
    """

    async def test_a_dataknobs_error_is_handled_without_propagating(self):
        """Handled cleanly: no exception escapes to the transport."""
        response = await _response_for(
            CommonConfigurationError("bad config"), raise_app_exceptions=True
        )

        assert response.status_code == 500

    async def test_a_plain_exception_still_propagates(self):
        """The catch-all's re-raise is unchanged for everything else."""
        with pytest.raises(ValueError, match="nope"):
            await _response_for(ValueError("nope"), raise_app_exceptions=True)

    async def test_an_error_raised_in_user_middleware_reaches_the_catch_all(self):
        """These handlers cover the app, not the whole ASGI stack.

        Starlette builds ``ServerErrorMiddleware`` → user middleware →
        ``ExceptionMiddleware`` → router, and only ``ExceptionMiddleware``
        consults the per-type handler map. An error raised in middleware is
        therefore *above* the layer that would give it a status: it goes to
        ``ServerErrorMiddleware``, which holds the ``Exception`` handler only,
        and comes back as a generic 500.

        Not a defect in this module — it is where Starlette puts the layer —
        but it bounds the claim, and the shape it bites is a plausible one: a
        tenant-resolving middleware raising ``BotNotFoundError`` returns 500,
        not 404. Pinned here so the documented remedy stays true.
        """
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient
        from starlette.middleware.base import BaseHTTPMiddleware

        class _TenantMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                raise BotNotFoundError("acme")

        app = FastAPI()
        register_exception_handlers(app)
        app.add_middleware(_TenantMiddleware)

        @app.get("/bots")
        async def bots():  # pragma: no cover - middleware raises first
            return {}

        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/bots")

        assert response.status_code == 500
        assert response.json()["error"] == "InternalServerError"

    async def test_converting_in_the_middleware_is_the_remedy(self):
        """The documented workaround, pinned rather than asserted in prose.

        A middleware that returns a response instead of raising keeps the
        status it meant, because it never has to cross the layer boundary.
        """
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient
        from starlette.middleware.base import BaseHTTPMiddleware

        from dataknobs_bots.api.exceptions import api_error_handler

        class _TenantMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                try:
                    raise BotNotFoundError("acme")
                except APIError as exc:
                    return await api_error_handler(request, exc)

        app = FastAPI()
        register_exception_handlers(app)
        app.add_middleware(_TenantMiddleware)

        @app.get("/bots")
        async def bots():  # pragma: no cover - middleware returns first
            return {}

        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/bots")

        assert response.status_code == 404
        assert response.json()["message"] == "Bot with ID 'acme' not found"


class TestRateLimitBodyParity:
    """Both ``RateLimitError`` variants describe one condition the same way."""

    @pytest.mark.parametrize(
        ("retry_after", "expected_header"),
        [(30.0, "30"), (0.0, "0"), (2.4, "3")],
        ids=["whole", "zero", "fractional-rounds-up"],
    )
    async def test_both_variants_agree_on_status_header_and_body(
        self, retry_after, expected_header
    ):
        """Same status, same header, and ``detail.retry_after`` in both.

        The two classes store the hint in different places: the API twin
        writes it into ``context``, which ``to_dict`` serializes, while the
        common one keeps it as an attribute only. So the same condition
        returned ``detail.retry_after`` from one variant and an empty
        ``detail`` from the other — and ``detail.retry_after`` is the shape
        this project's own docs tell callers to read.
        """
        api = await _response_for(RateLimitError("slow down", retry_after))
        common = await _response_for(CommonRateLimitError("slow down", retry_after=retry_after))

        for response in (api, common):
            assert response.status_code == 429
            assert response.headers["retry-after"] == expected_header
            assert response.json()["detail"]["retry_after"] == retry_after

    async def test_a_raiser_supplied_context_value_is_not_overwritten(self):
        """``setdefault``: an explicit ``context`` entry wins over the attribute."""
        response = await _response_for(
            CommonRateLimitError("slow down", retry_after=30.0, context={"retry_after": 99})
        )

        assert response.json()["detail"]["retry_after"] == 99

    async def test_no_hint_means_no_field_and_no_header(self):
        """Absent stays absent — nothing invents a wait the server never set."""
        response = await _response_for(CommonRateLimitError("slow down"))

        assert response.status_code == 429
        assert "retry-after" not in response.headers
        assert "retry_after" not in response.json()["detail"]


class _TenantQuotaError(CommonOperationError):
    """Stand-in for a deployment's own ``DataknobsError`` subclass."""


class TestErrorPolicyOverride:
    """``error_policy=`` is what makes the permissive defaults defensible."""

    async def test_a_consumer_type_can_be_given_its_own_policy(self):
        """A subclass DataKnobs has never heard of gets a first-class row."""
        exc = _TenantQuotaError("quota exhausted for tenant acme")

        default = await _response_for(exc)
        assert default.status_code == 500
        assert default.json()["message"] == api_exceptions.MASKED_MESSAGE

        overridden = await _response_for(
            exc, error_policy={_TenantQuotaError: ErrorPolicy(402, True)}
        )
        assert overridden.status_code == 402
        assert overridden.json()["message"] == "quota exhausted for tenant acme"

    async def test_a_default_row_can_be_masked(self):
        """The documented opt-out from the client-safe ``ConfigurationError``.

        A deployment serving unauthenticated traffic from a route that can
        raise a config error needs one line, not a fork of the handler.
        """
        exc = CommonConfigurationError("bad config", context={"config_key": "llm.api_base"})

        response = await _response_for(
            exc, error_policy={CommonConfigurationError: ErrorPolicy(500, False)}
        )

        assert response.status_code == 500
        body = response.json()
        assert body["message"] == api_exceptions.MASKED_MESSAGE
        assert body["detail"] == {}

    async def test_an_override_is_merged_over_the_defaults_not_replacing_them(self):
        """Rows the consumer did not mention keep working."""
        response = await _response_for(
            CommonNotFoundError("missing"),
            error_policy={CommonConfigurationError: ErrorPolicy(500, False)},
        )

        assert response.status_code == 404
        assert response.json()["message"] == "missing"

    def test_a_policy_for_an_api_error_is_rejected_not_ignored(self):
        """The table never sees an ``APIError``, so a row for one is a no-op.

        Starlette dispatches by walking ``type(exc).__mro__`` and taking the
        first registered handler, and ``APIError`` precedes ``DataknobsError``
        in every class in that family — so ``api_error_handler`` wins and
        decides disclosure from ``client_safe``, without consulting the table
        at all.

        Accepting the row anyway is the worst of both: the deployment writes
        what it believes is a disclosure policy, gets no error, and ships
        something with no effect. Since the mistake is not detectable from the
        response either — the default and the override look identical — it has
        to be caught at registration.
        """
        pytest.importorskip("fastapi")

        from fastapi import FastAPI

        with pytest.raises(CommonConfigurationError) as excinfo:
            register_exception_handlers(
                FastAPI(),
                error_policy={BotNotFoundError: ErrorPolicy(418, True, True)},
            )

        message = str(excinfo.value)
        assert "BotNotFoundError" in message
        assert "client_safe" in message

    async def test_the_two_configuration_errors_disagree_deliberately(self):
        """Same name, opposite disclosure — pinned so it stays a decision.

        ``api.ConfigurationError`` subclasses the common one, so the pair is
        easy to read as a drift. It is not: this one takes its message from a
        raise site writing for the caller, while the common one is also where
        the funnels wrapping a third-party constructor land, and their text is
        unbounded. The API class wins dispatch because ``APIError`` precedes
        ``DataknobsError`` in its MRO, so the two never contend for the same
        exception instance.
        """
        api_level = await _response_for(
            api_exceptions.ConfigurationError("llm.api_base is unset", "llm.api_base")
        )
        library_level = await _response_for(CommonConfigurationError("llm.api_base is unset"))

        assert api_level.json()["message"] == "llm.api_base is unset"
        assert api_level.json()["detail"] == {"config_key": "llm.api_base"}

        assert library_level.json()["message"] == api_exceptions.MASKED_MESSAGE
        assert library_level.json()["detail"] == {}

    def test_a_policy_for_a_non_dataknobs_type_is_rejected(self):
        """Same reasoning: the handler is registered for ``DataknobsError``."""
        pytest.importorskip("fastapi")

        from fastapi import FastAPI

        class NotOursError(Exception):
            pass

        with pytest.raises(CommonConfigurationError) as excinfo:
            register_exception_handlers(
                FastAPI(), error_policy={NotOursError: ErrorPolicy(418, True)}
            )

        assert "NotOursError" in str(excinfo.value)

    async def test_registration_returns_the_effective_table_for_middleware(self):
        """The footgun this closes: ``dataknobs_error_handler`` defaults.

        Middleware cannot raise to reach a handler — Starlette consults the
        per-type handlers only below the middleware stack — so the documented
        pattern is to *call* one. Calling ``dataknobs_error_handler`` without
        a ``table=`` silently applies ``DEFAULT_ERROR_POLICY`` rather than the
        table registered on the app, and the two differ exactly when someone
        bothered to pass ``error_policy=``. Naming cannot prevent an omitted
        argument; handing back the table gives it something to be passed.
        """
        pytest.importorskip("fastapi")

        from fastapi import FastAPI

        table = register_exception_handlers(
            FastAPI(), error_policy={_TenantQuotaError: ErrorPolicy(402, True)}
        )

        assert table[_TenantQuotaError] == ErrorPolicy(402, True)
        # Rows not mentioned come along, so the returned table is usable on
        # its own rather than only for the overridden types.
        assert table[CommonNotFoundError] == DEFAULT_ERROR_POLICY[CommonNotFoundError]

        with pytest.raises(TypeError):
            table[CommonNotFoundError] = ErrorPolicy(418, True)  # type: ignore[index]

    def test_the_default_table_is_not_mutated_by_an_override(self):
        """Registration merges into a copy; the module default is shared state."""
        pytest.importorskip("fastapi")

        from fastapi import FastAPI

        before = dict(DEFAULT_ERROR_POLICY)
        register_exception_handlers(
            FastAPI(), error_policy={CommonConfigurationError: ErrorPolicy(418, False)}
        )

        assert before == DEFAULT_ERROR_POLICY

    def test_the_default_table_cannot_be_assigned_into(self):
        """The copy above is the discipline; this is what enforces it.

        ``DEFAULT_ERROR_POLICY`` is process-global and read per request, so a
        consumer assigning a row into it would change the disclosure policy of
        every app in the process — including ones registered before the
        assignment, which no amount of care at registration time would catch.
        """
        with pytest.raises(TypeError):
            DEFAULT_ERROR_POLICY[CommonNotFoundError] = ErrorPolicy(200, True)  # type: ignore[index]

    async def test_masking_a_rate_limit_still_sends_retry_after(self):
        """The header is flow control, not disclosure, so masking keeps it.

        A deployment that masks a rate limit wants its *message* withheld, not
        its client's back-off broken — and the wait is the one thing a 429 has
        to say for the client to behave. So ``Retry-After`` survives
        ``client_safe=False`` even though ``detail.retry_after`` does not.
        """
        response = await _response_for(
            CommonRateLimitError("internal quota name leaks here", retry_after=30.0),
            error_policy={CommonRateLimitError: ErrorPolicy(429, False)},
        )

        assert response.status_code == 429
        assert response.headers["retry-after"] == "30"
        assert response.json()["message"] == api_exceptions.MASKED_MESSAGE
        assert response.json()["detail"] == {}


@pytest.mark.filterwarnings("ignore:BotManager is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*is deprecated.*Use.*instead:DeprecationWarning")
class TestBotManagerSingleton:
    """Tests for BotManager singleton management (deprecated)."""

    def setup_method(self):
        """Reset singleton before each test."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            reset_bot_manager()

    def teardown_method(self):
        """Reset singleton after each test."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            reset_bot_manager()

    def test_get_creates_default(self):
        """Test that get creates a default BotManager if none exists."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            manager = get_bot_manager()

        assert manager is not None
        # Check class name rather than importing BotManager
        assert type(manager).__name__ == "BotManager"

    def test_get_returns_same_instance(self):
        """Test that get returns the same instance."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            manager1 = get_bot_manager()
            manager2 = get_bot_manager()

        assert manager1 is manager2

    def test_init_with_config_loader(self):
        """Test initializing with a config loader."""

        def loader(bot_id: str) -> dict:
            return {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            manager = init_bot_manager(config_loader=loader)

            assert manager is not None
            assert get_bot_manager() is manager

    def test_reset_clears_instance(self):
        """Test that reset clears the singleton."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            manager1 = get_bot_manager()
            reset_bot_manager()
            manager2 = get_bot_manager()

        assert manager1 is not manager2

    def test_singleton_class_get(self):
        """Test _BotManagerSingleton.get()."""
        manager = _BotManagerSingleton.get()
        assert type(manager).__name__ == "BotManager"

    def test_singleton_class_init(self):
        """Test _BotManagerSingleton.init()."""
        manager = _BotManagerSingleton.init()
        assert type(manager).__name__ == "BotManager"
        assert _BotManagerSingleton.get() is manager

    def test_singleton_class_reset(self):
        """Test _BotManagerSingleton.reset()."""
        _BotManagerSingleton.get()  # Create instance
        _BotManagerSingleton.reset()
        assert _BotManagerSingleton._instance is None
