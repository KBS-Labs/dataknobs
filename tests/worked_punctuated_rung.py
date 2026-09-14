"""The consumer rung from the entity-resolution guide, executed as written.

This module is not collected by pytest.
``test_worked_entity_resolution_call_site.py`` imports it with
:func:`runpy.run_path` and drives the class against a vocabulary declaring the
punctuated forms the guide names.

**Everything below the blank line after this docstring is the guide's fence,
character for character**, and the test asserts that in both directions --
:mod:`worked_entity_resolution_call_site` carries the same arrangement and the
same reason for it.

Its own reason is narrower and worth stating. The call site one page up shows a
reader what the shipped rung *answers*; this shows them what a rung they write
themselves has to *do*, and its claims are the sharper of the two: that
``(beagle)``, ``C.D.C.`` and ``K-9`` are reachable this way and are not
reachable by the shipped scan. A page that made that claim while nothing ran it
would be asserting the one thing a reader cannot check without writing the
class first.

So this file is where a formatter, a linter and a type checker read that
sample. The fence is written the way ``ruff format`` writes it, for the reason
the sibling gives: a formatting finding here is a finding against a
documentation page, and no ``# noqa`` can answer it without rendering.

Do not add to it.
"""

import re

from dataknobs_common.entity_resolution import DeclaredSignal, FormHit


class PunctuatedFormRung(DeclaredSignal):
    key = "punctuated"

    def _located(self, query: str) -> list[FormHit]:
        found: list[FormHit] = []
        for chunk in re.finditer(r"\S+", query):  # "(beagle)" is one chunk
            hits = self._entities.by_surface_form(self._fold(chunk.group()))
            found += [FormHit(entity_id=i, span=chunk.span()) for i in self._order(hits)]
        return found
