"""What a failed resource acquisition is allowed to say.

The resource providers wrap ``except Exception`` around a driver, a session
factory, or a pool — code whose failure text they do not write and cannot
bound. A database client that cannot parse its DSN says so *by quoting the
DSN*, credentials included, which is demonstrated below rather than asserted:
:class:`DatabaseResourceAdapter` is given a real malformed connection string
and the password comes back in the message.

Since the exceptions in ``dataknobs_fsm.functions.base`` now reach the shared
hierarchy, this text is what a boundary rendering a ``ResourceError`` would
have to decide about. It resolves to a masked 503 under the bots API layer's
default policy, so nothing is disclosed today — but the message is the wrong
place to keep a credential regardless of who is currently choosing not to
print it, and that row is one ``error_policy=`` entry away from flipping.

The fix at each site is the one the rest of the branch uses: name the resource
and the exception type, and let ``raise ... from exc`` carry the original to
the logs.
"""

from __future__ import annotations

import pytest

from dataknobs_fsm.functions.base import ResourceError
from dataknobs_fsm.resources.database import DatabaseResourceAdapter

#: A DSN malformed in a way the postgres config normalizer reports by quoting
#: the offending value back. The password is the part that must not survive.
_MALFORMED_DSN = "::::svc:hunter2@@@"
_PASSWORD = "hunter2"


class TestADriversTextStaysOutOfTheMessage:
    """The provider's message is assembled from what the provider knows."""

    def test_a_malformed_dsn_does_not_echo_its_password(self):
        """The demonstration: real backend, real config, real failure path."""
        with pytest.raises(ResourceError) as excinfo:
            DatabaseResourceAdapter(
                name="orders",
                backend="postgres",
                connection_string=_MALFORMED_DSN,
                table="t",
            )

        assert _PASSWORD not in str(excinfo.value)

    def test_the_message_still_says_what_failed(self):
        """Bounding it must not empty it — a masked 503 logs this line."""
        with pytest.raises(ResourceError) as excinfo:
            DatabaseResourceAdapter(
                name="orders",
                backend="postgres",
                connection_string=_MALFORMED_DSN,
                table="t",
            )

        message = str(excinfo.value)
        assert "postgres" in message
        assert "ValueError" in message

    def test_the_original_is_still_reachable(self):
        """``__cause__`` is where the DSN-quoting text is supposed to go.

        Withholding it from the message only relocates it; a library caller
        reads it off the traceback and the API layer logs it.
        """
        with pytest.raises(ResourceError) as excinfo:
            DatabaseResourceAdapter(
                name="orders",
                backend="postgres",
                connection_string=_MALFORMED_DSN,
                table="t",
            )

        cause = excinfo.value.__cause__
        assert cause is not None
        assert _PASSWORD in str(cause)

    def test_the_resource_attributes_survive(self):
        """``resource_name``/``operation`` are authored, and stay."""
        with pytest.raises(ResourceError) as excinfo:
            DatabaseResourceAdapter(
                name="orders",
                backend="postgres",
                connection_string=_MALFORMED_DSN,
                table="t",
            )

        assert excinfo.value.resource_name == "orders"
        assert excinfo.value.operation == "initialize"

    def test_an_unknown_backend_names_the_backend_not_the_registry(self):
        """The other natural failure through the same wrap.

        The factory's own text here is bounded — a sorted list of registered
        backend keys — but the site cannot tell one wrapped exception from
        another, so it declines to relay any of them.
        """
        with pytest.raises(ResourceError) as excinfo:
            DatabaseResourceAdapter(name="orders", backend="no-such-backend")

        message = str(excinfo.value)
        assert "no-such-backend" in message
        assert "Available backends" not in message
