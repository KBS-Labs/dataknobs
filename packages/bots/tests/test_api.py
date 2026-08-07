"""Tests for API exceptions and dependencies."""

import warnings

import pytest

from dataknobs_bots.api import exceptions as api_exceptions
from dataknobs_bots.api.exceptions import (
    APIError,
    BotCreationError,
    BotNotFoundError,
    ConfigurationError,
    ConversationNotFoundError,
    RateLimitError,
    ValidationError,
    register_exception_handlers,
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
                if base.__module__ == common_exceptions.__name__
                and base is not DataknobsError
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
    async def test_a_429_carries_a_retry_after_header(
        self, retry_after, expected_header
    ):
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
