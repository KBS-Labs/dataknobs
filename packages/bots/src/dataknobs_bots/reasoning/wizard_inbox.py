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

The read step delegates through
:class:`~dataknobs_bots.reasoning.state_bridge.InboxOnlyBridge`, the
named-key consume-on-read bridge. Consumers wanting bi-directional,
projected, or observability-aware semantics compose the other
:class:`~dataknobs_bots.reasoning.state_bridge.StateBridge` reference
implementations with a custom ``on_turn_start`` / ``on_turn_end`` hook.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .lifecycle import TurnHookCallback
from .state_bridge import InboxOnlyBridge

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
      * Skips silently when the popped value is ``None`` (key not set this
        turn) or an empty mapping (writer published nothing).
      * Logs a WARNING and skips when a payload is set but not a mapping
        (so a writer-side bug — e.g. publishing a list or a scalar —
        doesn't crash the wizard).
      * Logs a DEBUG line per non-empty merge for traceability.
    """
    merge = merge_fn or _default_merge_fn
    bridge: InboxOnlyBridge[dict[str, Any]] = InboxOnlyBridge()

    async def _hook(event: dict[str, Any]) -> None:
        manager = event.get("manager")
        wizard_state = event.get("state")
        stage_name = event.get("stage", "")
        if manager is None or wizard_state is None:
            return
        if not hasattr(manager, "metadata"):
            return
        for key in inbox_keys:
            payload = bridge.read_inbox(manager, key)
            if payload is None:
                continue
            if not isinstance(payload, dict):
                logger.warning(
                    "Inbox payload at manager.metadata[%r] is not a "
                    "mapping; skipping. (payload type=%s)",
                    key,
                    type(payload).__name__,
                )
                continue
            if not payload:
                # Empty dict — writer published nothing. Silent skip.
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
    inbox content for that turn. This is the writer counterpart to the
    read side of
    :class:`~dataknobs_bots.reasoning.state_bridge.InboxOnlyBridge`; a
    consumer needing assign-or-merge or projected writes uses a
    :class:`~dataknobs_bots.reasoning.state_bridge.BiDirectionalBridge`
    / :class:`~dataknobs_bots.reasoning.state_bridge.SubsetBridge`
    instead.

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
