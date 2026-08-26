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
Being a leaf is also why the "is this key set?" test below is spelled
out rather than imported from ``wizard_types``, where
:func:`~dataknobs_bots.reasoning.wizard_types.field_is_present` defines
it for the rest of the package.  The two are pinned together by test.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Literal

from dataknobs_common.structured_config import StructuredConfig

#: Write the default over whatever is there.  The default mode, because
#: it is what ``dict.update`` has always done and because a stage that
#: relies on the clobber to escape an otherwise-inescapable branch would
#: be silently disarmed by any other choice.
SKIP_DEFAULT_OVERWRITE: Final[str] = "overwrite"

#: Write the default only where the key is unset.
SKIP_DEFAULT_FILL: Final[str] = "fill"

#: Every mode a stage may declare, in the order they are documented.
SKIP_DEFAULT_MODES: Final[tuple[str, ...]] = (SKIP_DEFAULT_OVERWRITE, SKIP_DEFAULT_FILL)

#: The keys an annotated entry names -- both of them, and nothing else.
#: A mapping naming any other set is a value in its own right.
_ANNOTATION_KEYS: Final[frozenset[str]] = frozenset({"value", "mode"})

#: Called as ``(key, value, requirement, outcome)`` when part of a block
#: cannot be read as written.  ``key`` locates the offending field
#: *within* the block -- ``"kb_enabled.mode"`` for one key's own mode --
#: and is ``""`` for the block-level mode, which is authored beside the
#: block rather than in it, so only the caller knows what that field is
#: called.  ``requirement`` says what would have been readable and
#: ``outcome`` says what was done instead, because the two are not
#: always the same answer: an unreadable mode falls back to another
#: mode, while a mapping that merely *looks* like an annotation is still
#: written as the value it reads as.  The caller supplies the context
#: and decides how loudly to say it.
OnInvalid = Callable[[str, Any, str, str], None]

#: What :func:`SkipDefaults.from_stage` says a mode has to be.
_MODE_REQUIRED: Final[str] = " or ".join(repr(mode) for mode in SKIP_DEFAULT_MODES) + " is required"


@dataclass(frozen=True)
class SkipDefaultEntry(StructuredConfig):
    """One key of a ``skip_default`` block, with the mode it resolved to.

    Attributes:
        value: The value to write.
        mode: One of :data:`SKIP_DEFAULT_MODES`.  Already resolved --
            an entry that declared no mode of its own carries the
            block-level one, so a consumer never has to look further up.
    """

    #: A misspelled field here would read as "this entry writes None",
    #: which is a value, so nothing downstream could notice.
    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"

    value: Any = None
    mode: str = SKIP_DEFAULT_OVERWRITE


@dataclass(frozen=True)
class SkipDefaults(StructuredConfig):
    """A stage's ``skip_default`` block, resolved into per-key modes.

    :meth:`from_stage` is the constructor for *authored* input -- a
    block is ``{key: value}`` and this class is ``{"entries": {...}}``,
    which are different shapes.  ``from_dict`` therefore takes only the
    projected shape, and rejects the authored one rather than quietly
    yielding an empty block that applies nothing.

    Attributes:
        entries: Key to the entry that writes it.  Empty when the stage
            declares no defaults, which is the common case and applies
            nothing.
    """

    #: Every key of an authored block would be an unknown field here,
    #: and ignoring them turns ``SkipDefaults.from_dict(block)`` into a
    #: silent no-op -- see the class docstring.
    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"

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

        A bare value takes *block_mode*.  A mapping is an annotated
        entry **only when it names exactly ``value`` and ``mode``**; any
        other mapping is itself the value, which is what keeps a nested
        default meaning what it reads as.  Three shapes turn on that
        rule: ``llm: {provider: "x"}`` names no ``value``;
        ``field: {value: "", label: "Email"}`` names one but is plainly
        not an annotation, and reading it as one would drop ``label`` on
        the floor; and ``threshold: {value: 3}`` names nothing an
        annotation needs, since an entry declaring no mode takes the
        block's -- which is exactly what the bare value does.  Requiring
        both keys therefore costs an author nothing and keeps a mapping
        that has always been a value from becoming a number.

        One collision is irreducible: a nested default naming *exactly*
        ``value`` and ``mode`` is indistinguishable from an annotation,
        and is read as one.  Wrap it in a real annotation to say
        otherwise::

            knob: {value: {value: 3, mode: "off"}, mode: overwrite}

        A mapping that names one of our modes without being an
        annotation is reported and then written as the value it reads
        as -- ``{values: false, mode: fill}`` is a typo, not a config,
        and storing a truthy mapping where the author wrote ``false`` is
        the silent loss this grammar exists to end.

        An unreadable mode falls back to the mode in force for that key
        **alone** -- the block's -- so one typo neither discards the
        rest of the block nor overrides what the block asked for.  A
        mode authored as an explicit ``null`` is unset rather than
        unreadable, the same reading
        :meth:`WizardFSM._stage_field` gives a stage field.

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
            if isinstance(raw, Mapping) and raw.keys() == _ANNOTATION_KEYS:
                declared = raw["mode"]
                mode = (
                    resolved_block_mode
                    if declared is None
                    else cls._mode(declared, resolved_block_mode, f"{key}.mode", on_invalid)
                )
                entries[key] = SkipDefaultEntry(value=raw["value"], mode=mode)
                continue

            if (
                on_invalid is not None
                and isinstance(raw, Mapping)
                and raw.get("mode") in SKIP_DEFAULT_MODES
            ):
                on_invalid(
                    key,
                    raw,
                    "an entry names exactly 'value' and 'mode'",
                    "Writing the mapping as the value it reads as",
                )
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
            on_invalid(key, raw, _MODE_REQUIRED, f"Using {default!r}")
        return default

    def apply(self, data: MutableMapping[str, Any]) -> list[str]:
        """Write the defaults into *data*.

        ``fill`` writes where the key is **unset**, which this package
        spells ``is None`` -- the reading
        :func:`~dataknobs_bots.reasoning.wizard_types.field_is_present`
        centralises for the ``has()`` condition helper, the confidence gate
        and
        :meth:`~dataknobs_bots.reasoning.wizard_extraction.WizardExtractor.apply_schema_defaults`.
        A key left holding ``None`` by extraction or by an earlier stage is
        one every other reader calls absent, so ``fill`` fills it.

        Each value is copied on the way in.  A mutable default belongs to
        the loaded config, and handing out the object itself lets one
        conversation's transform edit what the next one starts from.

        Args:
            data: The wizard's collected data, mutated in place.

        Returns:
            The keys whose **set** value was replaced by a different
            one, sorted.  A default equal to what was already there has
            replaced nothing, and a key that was unset had nothing to
            replace; reporting either would train a reader to ignore the
            case that matters.  ``fill`` never replaces, so it never
            appears here.
        """
        replaced = sorted(
            key
            for key, entry in self.entries.items()
            if entry.mode == SKIP_DEFAULT_OVERWRITE
            and data.get(key) is not None
            and data[key] != entry.value
        )
        for key, entry in self.entries.items():
            if entry.mode == SKIP_DEFAULT_FILL and data.get(key) is not None:
                continue
            data[key] = deepcopy(entry.value)
        return replaced
