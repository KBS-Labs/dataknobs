"""The shared base of the reasoning-strategy config family.

Its own module rather than the bottom of ``base.py``: every strategy
config imports it, and ``base.py`` imports the strategy machinery, so
putting it there would close an import cycle.  This mirrors
``dataknobs_data.backends.config``, where ``DatabaseConfig`` declares for
fourteen backends what each of them would otherwise re-declare.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataknobs_common.structured_config import StructuredConfig


@dataclass(frozen=True, kw_only=True)
class ReasoningConfig(StructuredConfig):
    """Base configuration for every :class:`ReasoningStrategy`.

    ``greeting_template`` is the one field the whole family shares, and
    :class:`~dataknobs_bots.reasoning.base.ReasoningStrategy` documents it
    as universal.  Declaring it here is what makes that documentation
    true: a strategy config that inherits cannot fail to have the field,
    whereas a family of independent declarations could only be *checked*
    for it -- and the check that existed could not see the gap, because
    ``StructuredConfig`` drops an undeclared key silently rather than
    reporting it.

    Attributes:
        greeting_template: Optional Jinja2 template for the bot-initiated
            greeting, rendered against the initial context.  Read through
            :attr:`ReasoningStrategy.greeting_template`, which is the one
            place any strategy reads it from.

    ``kw_only`` is load-bearing, not stylistic.  The field is defaulted,
    so without it no subclass could declare a *required* field -- the
    dataclass machinery rejects the class at import with ``non-default
    argument follows default argument``, and
    :class:`~dataknobs_bots.reasoning.wizard_config.WizardReasoningConfig`
    has exactly that shape in ``wizard_config``.  Marking only this base's
    field keyword-only leaves each subclass's own fields positional, so
    re-basing the family changed no call site.
    """

    greeting_template: str | None = None
