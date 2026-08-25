"""A ``keywords:`` field means the same thing wherever it is written.

``keywords`` is authored as a list of strings and is *iterated* by every
reader, so a bare string is the one wrong value that raises nothing: it
is iterable, and iterating it yields one keyword per character. A stage
declaring ``keywords: "done"`` arms ``d``, ``o``, ``n`` and ``e``, and a
user answering ``d`` triggers the command meant for ``done``.

There are four readers of that field in this package, and the guard
reached two of them:

* ``NavigationConfig`` / a stage's ``navigation:`` block, through
  ``NavigationCommandConfig.normalize_raw`` -- covered by
  ``tests/unit/test_wizard_navigation_config.py``;
* ``intent_confirm:``'s synthesizer, which does ``list(intent["keywords"])``
  at load time; and
* a hand-rolled ``intent_detection:`` block, which does
  ``tuple(i["keywords"])`` at classification time.

The two intent readers are covered here. They share the predicate with
the navigation readers -- what the field *means* must not depend on
which block it was written in -- but not the response: the synthesizer
has a ``validate()`` pass that raises at load, which is the right answer
for a wizard that has not started yet, while the runtime reader takes
the documented default and says so once, because there is a live
conversation to keep serving.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dataknobs_bots.testing import BotTestHarness


# ---------------------------------------------------------------------------
# intent_confirm: -- rejected at load
# ---------------------------------------------------------------------------


def _intent_confirm_wizard(keywords: Any) -> dict[str, Any]:
    """An ``intent_confirm:`` stage whose ``accept`` declares *keywords*."""
    return {
        "name": "intent-keywords",
        "version": "1.0",
        "stages": [
            {
                "name": "propose",
                "is_start": True,
                "intent_confirm": {
                    "proposal_template": "Use the default?",
                    "intents": {
                        "accept": {"target": "accepted", "keywords": keywords},
                        "decline": {"target": "declined"},
                    },
                },
            },
            {"name": "accepted", "is_end": True, "response_template": "Activated."},
            {"name": "declined", "is_end": True, "response_template": "Skipped."},
        ],
    }


def test_load_rejects_intent_keywords_written_as_a_string() -> None:
    """``keywords: "affirm"`` synthesized five one-letter keywords.

    The synthesizer validates every other shape in this block at load
    time -- a non-mapping ``intents``, a missing ``target``, a reserved
    name -- and then copied ``keywords`` through with ``list()``, which
    is the one call that turns a wrong value into a plausible one.
    """
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
    from dataknobs_common.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="keywords"):
        WizardConfigLoader().load_from_dict(_intent_confirm_wizard("affirm"))


def test_load_rejects_intent_keywords_that_are_not_strings() -> None:
    """``keywords: [1, 2]`` reached the classifier as ints.

    The keyword backend lowercases what it is given, so this failed later
    and elsewhere -- at classification time, on a turn, in a class that
    cannot name the stage that wrote it.
    """
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
    from dataknobs_common.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="keywords"):
        WizardConfigLoader().load_from_dict(_intent_confirm_wizard([1, 2]))


def test_load_accepts_a_proper_intent_keyword_list() -> None:
    """Anti-overreach: the shape the docs show must still load."""
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

    fsm = WizardConfigLoader().load_from_dict(_intent_confirm_wizard(["affirm", "yes"]))

    intents = fsm.stages["propose"]["intent_detection"]["intents"]
    accept = next(intent for intent in intents if intent["id"] == "accept")
    assert accept["keywords"] == ["affirm", "yes"]


# ---------------------------------------------------------------------------
# A hand-rolled intent_detection: block -- defaulted at runtime
# ---------------------------------------------------------------------------


def _hand_rolled_wizard(keywords: Any) -> dict[str, Any]:
    """A stage carrying an ``intent_detection:`` block written by hand.

    No synthesizer runs over this shape, so nothing has validated it by
    the time the classifier reads it -- which is why the runtime reader
    needs the guard rather than inheriting one.
    """
    return {
        "name": "hand-rolled-intents",
        "version": "1.0",
        "stages": [
            {
                "name": "propose",
                "is_start": True,
                "mode": "conversation",
                "response_template": "Use the default?",
                "confirm_first_render": False,
                "intent_detection": {
                    "classifier": "keyword",
                    "intents": [
                        {"id": "accept", "target": "accepted", "keywords": keywords},
                    ],
                    "per_intent_booleans": True,
                },
                "transitions": [
                    {"target": "accepted", "condition": "data.get('accept')"},
                ],
            },
            {"name": "accepted", "is_end": True, "response_template": "Activated."},
        ],
    }


@pytest.mark.asyncio
async def test_a_hand_rolled_keywords_string_does_not_arm_single_letters() -> None:
    """``keywords: "done"`` matched a user who typed ``d``.

    ``per_intent_booleans`` then wrote ``data["accept"] = True`` and the
    transition fired: a one-character message advanced the wizard past a
    confirmation the user never gave.
    """
    async with await BotTestHarness.create(
        wizard_config=_hand_rolled_wizard("done"),
    ) as harness:
        await harness.greet()

        await harness.chat("d")

        assert harness.wizard_stage == "propose", (
            "a single letter matched a keyword meant for 'done' and advanced the wizard"
        )
        assert "accept" not in harness.wizard_data


@pytest.mark.asyncio
async def test_a_hand_rolled_keywords_string_is_reported_once() -> None:
    """The value is unusable, so the reader says so rather than guessing.

    Once per intent: this runs on every turn the stage is current, so an
    unthrottled line would repeat for the life of the conversation --
    the budget ``_stage_field`` and the navigation reader both keep.
    """
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Collect()
    reader_logger = logging.getLogger("dataknobs_bots.reasoning.wizard_extraction")
    reader_logger.addHandler(handler)
    try:
        async with await BotTestHarness.create(
            wizard_config=_hand_rolled_wizard("done"),
        ) as harness:
            await harness.greet()
            await harness.chat("d")
            await harness.chat("d")
    finally:
        reader_logger.removeHandler(handler)

    reported = [message for message in records if "keywords" in message]
    assert len(reported) == 1, f"expected one report per intent, got {reported}"
    assert "accept" in reported[0]


@pytest.mark.asyncio
async def test_a_hand_rolled_keyword_list_still_routes() -> None:
    """Anti-overreach: the shape the docs show must still classify."""
    async with await BotTestHarness.create(
        wizard_config=_hand_rolled_wizard(["done"]),
    ) as harness:
        await harness.greet()

        await harness.chat("done")

        assert harness.wizard_stage == "accepted"
        assert harness.wizard_data.get("accept") is True
