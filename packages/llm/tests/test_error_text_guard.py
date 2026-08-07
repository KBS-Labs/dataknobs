"""No unbounded vendor text in this package's error messages.

Provider translation is guarded separately and more precisely in
``test_vendor_error_disclosure.py``, which asserts on the messages the
dispatcher actually produces. This is the source-level companion: it covers
the rest of the package, where the same shape can reappear without going
through the dispatcher at all.

The FSM integration layer is the reason it is not redundant. Those transforms
wrap ``except Exception`` around a live provider call, so a failure there
carries the same vendor rendering the dispatcher exists to withhold — an
endpoint URL, a relayed response body, an AWS operation name — into a
``TransformError`` that never passes through the translation path.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.testing import (
    GUARDED_ERROR_NAMES,
    assert_no_broad_except_in_error_text,
)

_SRC = Path(__file__).resolve().parents[1] / "src"

_GUARDED = GUARDED_ERROR_NAMES | {"TransformError", "FSMError", "LLMError"}


def test_no_broad_except_feeds_a_rendered_error_message():
    assert_no_broad_except_in_error_text(_SRC, error_names=_GUARDED)
