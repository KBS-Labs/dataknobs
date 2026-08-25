"""The grammar of a stage's skip-time defaults.

``skip_default`` writes values into the wizard's collected data when the
user skips a stage.  Which of those writes may land on a key that is
already set is the stage author's decision, and it is per key: a real
stage needs one option preserved because the user configured it and a
sibling flag clobbered because it guards a branch the skip is trying to
leave.

This module owns the shape and nothing else -- reading the block off a
stage belongs to :class:`~dataknobs_bots.reasoning.wizard_fsm.WizardFSM`,
and applying it to a live state belongs to the navigator.  It is a leaf
so that both of those and the config loader can share one reading of the
grammar; what a field means must not depend on which of them read it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Final

from dataknobs_common.structured_config import StructuredConfig

#: Write the default over whatever is there.  The default mode, because
#: it is what ``dict.update`` has always done and because a stage that
#: relies on the clobber to escape an otherwise-inescapable branch would
#: be silently disarmed by any other choice.
SKIP_DEFAULT_OVERWRITE: Final[str] = "overwrite"

#: Write the default only where the key is absent.
SKIP_DEFAULT_FILL: Final[str] = "fill"

#: Every mode a stage may declare, in the order they are documented.
SKIP_DEFAULT_MODES: Final[tuple[str, ...]] = (SKIP_DEFAULT_OVERWRITE, SKIP_DEFAULT_FILL)

#: Called as ``(key, value, expected)`` when part of a block cannot be
#: read.  ``key`` locates the offending field *within* the block --
#: ``"kb_enabled.mode"`` for one key's own mode -- and is ``""`` for the
#: block-level mode, which is authored beside the block rather than in
#: it, so only the caller knows what that field is called.  The caller
#: supplies the context and decides how loudly to say it -- the same
#: contract ``NavigationCommandConfig.normalize_raw`` uses.
OnInvalid = Callable[[str, Any, str], None]


@dataclass(frozen=True)
class SkipDefaultEntry(StructuredConfig):
    """One key of a ``skip_default`` block, with the mode it resolved to.

    Attributes:
        value: The value to write.
        mode: One of :data:`SKIP_DEFAULT_MODES`.  Already resolved --
            an entry that declared no mode of its own carries the
            block-level one, so a consumer never has to look further up.
    """

    value: Any = None
    mode: str = SKIP_DEFAULT_OVERWRITE


@dataclass(frozen=True)
class SkipDefaults(StructuredConfig):
    """A stage's ``skip_default`` block, resolved into per-key modes.

    Attributes:
        entries: Key to the entry that writes it.  Empty when the stage
            declares no defaults, which is the common case and applies
            nothing.
    """

    entries: dict[str, SkipDefaultEntry] = field(default_factory=dict)

    @classmethod
    def from_stage(
        cls,
        block: Mapping[str, Any] | None,
        block_mode: str = SKIP_DEFAULT_OVERWRITE,
        *,
        on_invalid: OnInvalid | None = None,
    ) -> SkipDefaults:
        """Resolve an authored block against its block-level mode.

        Two shapes are accepted, and the choice between them is per key
        so that one block can hold both::

            skip_default:
              scenario_enabled: false                      # bare value
              kb_enabled: {value: false, mode: fill}       # annotated

        A bare value takes *block_mode*.  A mapping is an annotated entry
        **only when it carries a ``value`` key**; any other mapping is
        itself the value, which is what keeps a nested default such as
        ``llm: {provider: "x"}`` meaning what it reads as.

        An unreadable mode falls back to the documented default for that
        key **alone**, so one typo does not discard the rest of the
        block -- the same contract
        :meth:`WizardFSM._stage_field` gives a wrong-typed stage field.

        Args:
            block: The authored ``skip_default`` mapping, or ``None``.
            block_mode: The stage's ``skip_default_mode``, applied to
                every key that does not name its own.
            on_invalid: Reporter for anything unreadable; see
                :data:`OnInvalid`.

        Returns:
            A resolved block.  Every entry carries a mode from
            :data:`SKIP_DEFAULT_MODES`.
        """
        if not block:
            return cls()

        resolved_block_mode = cls._mode(block_mode, SKIP_DEFAULT_OVERWRITE, "", on_invalid)

        entries: dict[str, SkipDefaultEntry] = {}
        for key, raw in block.items():
            if isinstance(raw, Mapping) and "value" in raw:
                mode = cls._mode(
                    raw.get("mode", resolved_block_mode),
                    resolved_block_mode,
                    f"{key}.mode",
                    on_invalid,
                )
                entries[key] = SkipDefaultEntry(value=raw["value"], mode=mode)
            else:
                entries[key] = SkipDefaultEntry(value=raw, mode=resolved_block_mode)

        return cls(entries=entries)

    @staticmethod
    def _mode(
        raw: Any,
        default: str,
        key: str,
        on_invalid: OnInvalid | None,
    ) -> str:
        """*raw* if it names a mode, else *default*, reported once."""
        if raw in SKIP_DEFAULT_MODES:
            return str(raw)
        if on_invalid is not None:
            on_invalid(key, raw, " or ".join(repr(mode) for mode in SKIP_DEFAULT_MODES))
        return default

    def apply(self, data: MutableMapping[str, Any]) -> list[str]:
        """Write the defaults into *data*.

        Args:
            data: The wizard's collected data, mutated in place.

        Returns:
            The keys whose existing value was **replaced by a different
            one**, sorted.  A default equal to what was already there has
            replaced nothing, and reporting it would train a reader to
            ignore the case that matters.  ``fill`` never replaces, so it
            never appears here.
        """
        replaced = sorted(
            key
            for key, entry in self.entries.items()
            if entry.mode == SKIP_DEFAULT_OVERWRITE and key in data and data[key] != entry.value
        )
        for key, entry in self.entries.items():
            if entry.mode == SKIP_DEFAULT_FILL:
                data.setdefault(key, entry.value)
            else:
                data[key] = entry.value
        return replaced
