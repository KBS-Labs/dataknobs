"""``dataknobs_llm.conversations`` must import without ``dataknobs-fsm``.

``dataknobs-fsm`` is an optional dependency: it backs
:mod:`dataknobs_llm.fsm_integration` and ``ConversationFlowAdapter``, and
nothing else. But ``conversations/manager.py`` imports the FSM-free leaf
``conversations.flow.flow``, and importing a submodule runs its parent
package's ``__init__`` — so an eager re-export of ``.adapter`` there pulls an
FSM engine into ``import dataknobs_llm.conversations``, and with it
``ConversationManager``, on a base install that has no ``dataknobs-fsm``.

The absence of a distribution is the whole variable, so these tests reproduce
it directly: a ``sys.meta_path`` finder refuses ``dataknobs_fsm`` and the
package under test runs unmodified.

Each probe runs in **its own interpreter**. A failed import leaves enough
behind that a second probe in the same process reports a false OK.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_BLOCKER = """
import sys

class _Refuse:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "dataknobs_fsm" or fullname.startswith("dataknobs_fsm."):
            raise ModuleNotFoundError(f"No module named {fullname!r}")
        return None

sys.meta_path.insert(0, _Refuse())
"""


def _run(body: str, *, block_fsm: bool = True) -> subprocess.CompletedProcess[str]:
    """Run *body* in a fresh interpreter, optionally with ``dataknobs_fsm`` refused."""
    script = (_BLOCKER if block_fsm else "") + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,  # a non-zero exit IS the measurement; the caller asserts on it
    )


def _assert_ok(result: subprocess.CompletedProcess[str], what: str) -> None:
    assert result.returncode == 0, (
        f"{what} failed without dataknobs-fsm.\n--- stderr ---\n{result.stderr}"
    )


def test_conversations_imports_without_fsm() -> None:
    """The headline failure: a base install could not import this subpackage."""
    _assert_ok(_run("import dataknobs_llm.conversations"), "import of conversations")


def test_conversation_manager_is_reachable_without_fsm() -> None:
    """``ConversationManager`` is the package's headline surface."""
    _assert_ok(
        _run("from dataknobs_llm.conversations import ConversationManager"),
        "import of ConversationManager",
    )


def test_conversation_storage_imports_without_fsm() -> None:
    """``manager`` is reached through ``conversations/__init__``, ``storage`` beside it."""
    _assert_ok(
        _run("import dataknobs_llm.conversations.storage"),
        "import of conversations.storage",
    )


def test_fsm_free_flow_names_still_export_without_fsm() -> None:
    """The three names in ``flow`` that need no FSM must survive the split."""
    _assert_ok(
        _run(
            """
            from dataknobs_llm.conversations.flow import (
                ConversationFlow,
                FlowState,
                TransitionCondition,
            )
            """
        ),
        "import of the FSM-free flow names",
    )


def test_adapter_still_importable_when_fsm_is_present() -> None:
    """The public API guarantee — deferring the import must not remove the name.

    ``manager.py`` imports ``ConversationFlowAdapter`` by this path at call
    time, and it is in ``flow/__init__.__all__``. This is what stops "make it
    lazy" from becoming "delete the export".
    """
    result = _run(
        """
        from dataknobs_llm.conversations.flow import (
            ConversationFlowAdapter,
            FlowExecutionState,
        )
        assert ConversationFlowAdapter.__name__ == "ConversationFlowAdapter"
        assert FlowExecutionState.__name__ == "FlowExecutionState"
        """,
        block_fsm=False,
    )
    _assert_ok(result, "import of ConversationFlowAdapter with fsm present")


def test_adapter_raises_a_named_import_error_without_fsm() -> None:
    """Asking for the FSM-backed name without the extra must say so, not AttributeError."""
    result = _run(
        """
        try:
            from dataknobs_llm.conversations.flow import ConversationFlowAdapter
        except ModuleNotFoundError as exc:
            assert "dataknobs_fsm" in str(exc), str(exc)
        else:
            raise AssertionError("expected ModuleNotFoundError")
        """
    )
    _assert_ok(result, "the named failure for the FSM-backed export")


def test_flow_all_is_unchanged() -> None:
    """Pin the exported contract: laziness must not shrink ``__all__``."""
    from dataknobs_llm.conversations import flow

    assert set(flow.__all__) >= {
        "ConversationFlow",
        "FlowState",
        "TransitionCondition",
        "ConversationFlowAdapter",
        "FlowExecutionState",
    }


def test_unknown_attribute_still_raises_attribute_error() -> None:
    """A module ``__getattr__`` must not swallow genuine typos."""
    from dataknobs_llm.conversations import flow

    with pytest.raises(AttributeError, match="NoSuchName"):
        _ = flow.NoSuchName
