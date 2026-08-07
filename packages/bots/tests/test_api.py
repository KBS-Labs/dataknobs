"""Tests for API exceptions and dependencies."""

import warnings

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
#: produce, written out independently of ``DEFAULT_ERROR_POLICY``. Asserting the
#: table against itself would pass for any value it happened to hold; this is
#: the contract, and changing a row in the source has to fail here first.
_EXPECTED_POLICY: dict[str, tuple[int, bool]] = {
    "ValidationError": (422, True),
    "NotFoundError": (404, True),
    "ConsentRequiredError": (403, True),
    "ConcurrencyError": (409, True),
    "RateLimitError": (429, True),
    "TimeoutError": (504, True),
    "ConfigurationError": (500, True),
    "ResourceError": (503, False),
    "SerializationError": (500, False),
    "OperationError": (500, False),
    "DataknobsError": (500, False),
}


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
                BotCreationError("bot-1", "disk full"),
                500,
                "Failed to create bot 'bot-1': disk full",
                {"bot_id": "bot-1", "reason": "disk full"},
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

        Two of the rows would visibly change if that regressed: ``APIError``
        and ``BotCreationError`` resolve through the table to *masked* 500s,
        so their real messages would disappear. The other five happen to
        produce the same body either way, which is by design — the table gives
        the common types the same statuses the API twins already used — and
        they are here to pin that agreement rather than to detect the
        regression.
        """
        response = await _response_for(exc)

        assert response.status_code == status
        body = response.json()
        assert body["message"] == message
        assert body["detail"] == detail
        assert body["error"] == type(exc).__name__
        assert "timestamp" in body


class TestErrorPolicyTable:
    """The policy table covers the hierarchy and resolves by MRO."""

    @pytest.mark.parametrize("name", sorted(_EXPECTED_POLICY), ids=sorted(_EXPECTED_POLICY))
    def test_table_matches_the_declared_contract(self, name):
        """Each entry has the status and disclosure recorded above."""
        cls = getattr(common_exceptions, name)
        status, client_safe = _EXPECTED_POLICY[name]

        assert cls in DEFAULT_ERROR_POLICY, f"{name} has no policy entry"
        assert DEFAULT_ERROR_POLICY[cls] == ErrorPolicy(status, client_safe)

    def test_every_common_exception_has_a_policy(self):
        """A new common error type must be given a row, not defaulted.

        Without this, adding one to ``dataknobs_common.exceptions`` would
        silently resolve it through ``DataknobsError`` to a masked 500 — the
        exact behaviour this handler exists to stop, reintroduced one type at
        a time.
        """
        missing = [
            name
            for name in common_exceptions.__all__
            if getattr(common_exceptions, name) not in DEFAULT_ERROR_POLICY
        ]
        assert not missing, (
            f"{', '.join(missing)} in dataknobs_common.exceptions.__all__ but "
            f"absent from DEFAULT_ERROR_POLICY — add a row, deciding its "
            f"status and whether its message and context are client-safe"
        )

    def test_the_expected_policy_table_itself_is_complete(self):
        """The test's own expectations must not go stale either."""
        assert set(_EXPECTED_POLICY) == set(common_exceptions.__all__)

    def test_dataknobs_error_is_the_terminal_entry(self):
        """The fallback is a row, so a consumer can override it."""
        assert DEFAULT_ERROR_POLICY[DataknobsError] == ErrorPolicy(500, False)

    def test_resolution_walks_the_mro(self):
        """A subclass with no row of its own inherits its nearest ancestor's."""
        from dataknobs_data.exceptions import RecordNotFoundError

        assert resolve_error_policy(RecordNotFoundError("rec-1")) == ErrorPolicy(404, True)

    def test_a_subclass_row_beats_its_base(self):
        """``RateLimitError`` is an ``OperationError``; 429 must win over 500.

        Resolution is by MRO, not by dict order, so this holds however the
        table happens to be sorted in the source.
        """
        assert resolve_error_policy(CommonRateLimitError("nope")) == ErrorPolicy(429, True)

    def test_a_table_without_a_terminal_entry_fails_closed(self):
        """An override table that omits ``DataknobsError`` masks, not discloses.

        Reachable only by calling this function directly — the registration
        path merges over the defaults — which is why the guard is here rather
        than trusted to be unreachable.
        """
        assert resolve_error_policy(CommonValidationError("bad"), {}) == ErrorPolicy(500, False)


class TestDataknobsErrorHandling:
    """DataKnobs' own errors reach the client with a meaningful status."""

    @pytest.mark.parametrize("name", sorted(_EXPECTED_POLICY), ids=sorted(_EXPECTED_POLICY))
    async def test_each_common_type_end_to_end(self, name):
        """Raised from a route, each type returns its status and disclosure.

        Before this handler existed, every row here returned
        ``500 / "An unexpected error occurred"``.
        """
        status, client_safe = _EXPECTED_POLICY[name]
        cls = getattr(common_exceptions, name)
        exc = cls("diagnostic detail", context={"probe": "value"})

        response = await _response_for(exc)

        assert response.status_code == status
        body = response.json()
        assert body["error"] == name
        if client_safe:
            assert body["message"] == "diagnostic detail"
            assert body["detail"]["probe"] == "value"
        else:
            assert body["message"] == api_exceptions.MASKED_MESSAGE
            assert body["detail"] == {}

    async def test_the_configuration_diagnostic_survives(self):
        """The case the gap was reported for.

        A config diagnostic is generated by DataKnobs and was then discarded
        by DataKnobs one layer later, leaving the deployment a 500 saying
        nothing about which key was wrong.
        """
        response = await _response_for(
            CommonConfigurationError(
                "embedding: no variant registered for 'ollamaa'",
                context={"config_key": "embedding"},
            )
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
            # common, defined outside exceptions.py: ConfigurationError -> 500
            (_pack_resolution_error, 500, "pack 'audit' is not registered"),
            # common, defined outside exceptions.py: OperationError -> masked 500
            (_capability_not_supported, 500, None),
        ],
        ids=["data", "llm", "common-packs", "common-capabilities"],
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

    def test_the_default_table_is_not_mutated_by_an_override(self):
        """Registration merges into a copy; the module default is shared state."""
        pytest.importorskip("fastapi")

        from fastapi import FastAPI

        before = dict(DEFAULT_ERROR_POLICY)
        register_exception_handlers(
            FastAPI(), error_policy={CommonConfigurationError: ErrorPolicy(418, False)}
        )

        assert before == DEFAULT_ERROR_POLICY

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
