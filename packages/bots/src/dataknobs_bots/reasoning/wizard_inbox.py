"""Manager-metadata inbox bridge.

The inbox bridge is implemented as a turn-start hook factory:
:func:`make_metadata_inbox_hook` produces an ``on_turn_start`` hook
that pops one or more ``manager.metadata`` keys and merges their
contents into ``wizard_state.data``.

When :attr:`WizardReasoningConfig.manager_metadata_inbox_key
<dataknobs_bots.reasoning.wizard_config.WizardReasoningConfig.manager_metadata_inbox_key>`
is set, the wizard auto-registers a hook produced by this factory.
Consumers needing variations (different merge semantics, additional
side effects, etc.) can call the factory directly with a custom
``merge_fn``, or write their own ``on_turn_start`` hooks from scratch.

Also exports :func:`write_to_inbox` — a small writer helper for
consumer code (pipeline steps, custom logic) that publishes payloads
for the NEXT turn's inbox consumption.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .lifecycle import TurnHookCallback

logger = logging.getLogger(__name__)


def _default_merge_fn(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Shallow merge — the dataknobs default."""
    target.update(source)


def make_metadata_inbox_hook(
    *,
    inbox_keys: list[str],
    merge_fn: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
) -> TurnHookCallback:
    """Build an ``on_turn_start`` hook that drains inbox keys.

    Args:
        inbox_keys: Manager-metadata keys to pop and merge. Drained
            in order on every turn.
        merge_fn: Custom merger ``(state.data, inbox_payload) -> None``.
            Defaults to :func:`dict.update` (shallow merge). Consumers
            supply deep-merge, conflict-resolving, or other semantics.

    Returns:
        Async callable matching the :data:`TurnHookCallback` shape.

    The returned hook:
      * Pops each configured key (consume-on-read; stale signals can't leak).
      * Skips silently when the popped value is empty / ``None``.
      * Logs a WARNING and skips when a payload is not a mapping
        (so a writer-side bug doesn't crash the wizard).
      * Logs a DEBUG line per non-empty merge for traceability.
    """
    merge = merge_fn or _default_merge_fn

    async def _hook(manager: Any, wizard_state: Any, stage_name: str) -> None:
        if not hasattr(manager, "metadata"):
            return
        for key in inbox_keys:
            payload = manager.metadata.pop(key, None)
            if not payload:
                continue
            if not isinstance(payload, dict):
                logger.warning(
                    "Inbox payload at manager.metadata[%r] is not a "
                    "mapping; skipping. (payload type=%s)",
                    key,
                    type(payload).__name__,
                )
                continue
            merge(wizard_state.data, payload)
            logger.debug(
                "Wizard consumed manager.metadata[%r] at stage '%s': keys=%s",
                key,
                stage_name,
                sorted(payload.keys()),
            )

    return _hook


def write_to_inbox(manager: Any, key: str, payload: dict[str, Any]) -> None:
    """Publish a payload to a manager-metadata inbox key.

    Writer-side helper for consumer code (pipeline steps, custom
    logic) that produces signals destined for the NEXT turn's
    ``on_turn_start`` inbox merge.

    Convention: writers should always overwrite (not merge) the inbox
    key — the wizard's consume-on-read pop means a stale write from a
    prior turn cannot survive, so each turn's writer fully owns the
    inbox content for that turn.

    Args:
        manager: ConversationManager (or compatible — must have a
            ``metadata`` dict attribute).
        key: Inbox key (must match the wizard's
            ``manager_metadata_inbox_key`` config).
        payload: Flat dict to publish. ``None`` values within the
            payload act as evictions on the next turn's merge (the
            default :func:`dict.update` merge writes ``None`` into
            the corresponding ``state.data`` key).

    Raises:
        AttributeError: If the manager has no ``metadata`` attribute.
    """
    if not hasattr(manager, "metadata"):
        raise AttributeError(
            f"{type(manager).__name__} has no 'metadata' attribute; "
            f"cannot write inbox payload."
        )
    manager.metadata[key] = payload
