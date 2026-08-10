"""``WizardReasoning`` closes its wizard FSM — but only if it built it.

The FSM is a *required constructor parameter*, so there are two paths to a
``WizardReasoning``: ``from_config`` builds the FSM via the loader (the
strategy owns it), and the direct constructor receives a pre-built one (the
caller owns it, and may still be stepping it). An unconditional close would
tear down a live FSM at the second, far more numerous, kind of site.
"""

from __future__ import annotations

import threading

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_config import WizardReasoningConfig
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_bots.testing import BotTestHarness
from dataknobs_common.testing import DK_SYNC_BRIDGE_THREAD

_WIZARD_DICT = {
    "name": "ownership-test-wizard",
    "version": "1.0",
    "stages": [
        {
            "name": "start",
            "is_start": True,
            "prompt": "Hello?",
            "transitions": [{"target": "done"}],
        },
        {"name": "done", "is_end": True, "prompt": "Bye."},
    ],
}


def _bridge_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name == DK_SYNC_BRIDGE_THREAD)


async def test_from_config_built_strategy_closes_its_fsm() -> None:
    """The owned path: the strategy built the FSM, so it releases it."""
    before = _bridge_threads()
    reasoning = WizardReasoning.from_config({"wizard_config": _WIZARD_DICT})
    reasoning._fsm.step({})
    assert _bridge_threads() == before + 1

    await reasoning.close()

    assert _bridge_threads() == before


async def test_direct_ctor_strategy_leaves_its_fsm_open() -> None:
    """The ownership gate's guard — delete it and this test fails.

    The caller built this FSM and still holds it. Closing it here would be
    a use-after-close at every direct-construction site.
    """
    before = _bridge_threads()
    fsm = WizardConfigLoader().load_from_dict(_WIZARD_DICT)
    reasoning = WizardReasoning(wizard_fsm=fsm)
    fsm.step({})
    assert _bridge_threads() == before + 1

    await reasoning.close()

    assert _bridge_threads() == before + 1, "a caller-owned FSM was closed"

    # Still usable, which is the point — the caller closes it when done.
    fsm.step({})
    fsm.close()
    assert _bridge_threads() == before


async def test_direct_ctor_with_a_typed_config_still_leaves_the_fsm_open() -> None:
    """Ownership is an explicit flag, not an inference from the sentinel.

    A caller may pass the direct ctor a pre-built FSM *and* a real config,
    which produces an injected FSM carrying no ``<injected-fsm>`` sentinel.
    Deriving ownership from the sentinel would close this one.
    """
    before = _bridge_threads()
    fsm = WizardConfigLoader().load_from_dict(_WIZARD_DICT)
    reasoning = WizardReasoning(
        wizard_fsm=fsm,
        config=WizardReasoningConfig(wizard_config=_WIZARD_DICT, strict_validation=False),
    )
    assert reasoning.config.wizard_config == _WIZARD_DICT
    fsm.step({})

    await reasoning.close()

    assert _bridge_threads() == before + 1
    fsm.close()
    assert _bridge_threads() == before


async def test_close_is_idempotent_on_the_owned_path() -> None:
    before = _bridge_threads()
    reasoning = WizardReasoning.from_config({"wizard_config": _WIZARD_DICT})
    reasoning._fsm.step({})

    await reasoning.close()
    await reasoning.close()

    assert _bridge_threads() == before


async def test_dynabot_close_reaches_the_fsm_end_to_end() -> None:
    """The production path, unbroken from ``DynaBot.close()`` down.

    ``DynaBot`` builds its strategy via ``from_config``, so the whole chain
    is owned: bot → strategy → FSM.
    """
    before = _bridge_threads()
    # ``async with``: a failing assertion below must not leak the harness,
    # which would surface as this file leaking a thread it did not create.
    async with await BotTestHarness.create(
        wizard_config=_WIZARD_DICT,
        main_responses=["Hi there!"],
    ) as harness:
        strategy = harness.bot.reasoning_strategy
        strategy._fsm.step({})
        assert _bridge_threads() == before + 1

        await harness.bot.close()

        assert _bridge_threads() == before
