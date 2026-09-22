# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for dataknobs packages.

**The submodules are reached lazily (PEP 562), not imported here.** Every
spelling a caller already uses still works --- ``from dataknobs_utils import
json_utils``, ``import dataknobs_utils.json_utils``,
``dataknobs_utils.json_utils`` --- and each one imports that module and no
other.

This door used to import all fifteen *"for easy access"*, and the package
declares no optional extras, so a consumer reaching **one** function paid for
all fifteen: measured in a fresh interpreter, that single import line loaded
**1,342 modules** and pulled ``nltk``, ``pandas``, ``numpy``, ``psycopg2``,
``lxml``, ``beautifulsoup4``, ``requests``, ``graphviz``, ``defusedxml``,
``python-dotenv`` and ``json-stream`` behind it. The case that found it: a
consumer adding ``dataknobs-data`` to a package declaring
``dataknobs-common`` alone went from 2 third-party distributions to 34, and
the twenty-eight beyond the four ``dataknobs-data`` declares arrive behind
this door --- for one name, ``quote_ident`` out of ``sql_utils``, reached by
the **in-memory** backend.

**This changes what a process imports, not what a consumer installs.** That
follows from what ``pyproject.toml`` declares, and moving a dependency into
an extra is a separate change with its own compatibility surface.

The pattern is this repository's own, three times over with its reason
written out: ``dataknobs_common.events`` defers ``SqsEventBus`` because
*"importing it eagerly would pull the optional aioboto3 dependency at import
time, putting a backend driver in the base install"*, and
``dataknobs_bots.knowledge`` and ``dataknobs_llm.tools`` carry the same
shape. This door is the one that cost the most and had not adopted it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    # Named eagerly here and nowhere else: a type checker resolves
    # ``dataknobs_utils.json_utils`` from this block, and ``TYPE_CHECKING``
    # is never true at run time, so the deferral is unaffected.
    from dataknobs_utils import (
        elasticsearch_utils,
        emoji_utils,
        file_utils,
        json_extractor,
        json_utils,
        llm_utils,
        pandas_utils,
        requests_utils,
        resource_utils,
        sql_utils,
        stats_utils,
        subprocess_utils,
        sys_utils,
        value_expansion,
        xml_utils,
    )

__version__ = "2.0.2"

__all__ = [
    "elasticsearch_utils",
    "emoji_utils",
    "file_utils",
    "json_extractor",
    "json_utils",
    "llm_utils",
    "pandas_utils",
    "requests_utils",
    "resource_utils",
    "sql_utils",
    "stats_utils",
    "subprocess_utils",
    "sys_utils",
    "value_expansion",
    "xml_utils",
]

#: The submodules this door hands back, as a set for the membership test.
_SUBMODULES = frozenset(__all__)


def __getattr__(name: str) -> object:
    """Import one submodule on first access (PEP 562).

    ``importlib.import_module`` binds the module onto this package as a side
    effect of importing it, so the second access never reaches here --- which
    is what keeps a lazy door from costing anything per call.

    A name that is not a submodule raises :class:`AttributeError` rather than
    letting :class:`ImportError` escape, because a typo at an attribute access
    is an attribute error and reporting it as an import failure sends the
    reader looking for a missing dependency.
    """
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """The submodules included, which a lazy door otherwise hides.

    ``dir()`` on a module answers its ``__dict__`` and nothing has been
    imported yet, so tab completion, ``help()`` and every introspecting tool
    would see an empty package. That is a discoverability regression no import
    test could catch, which is why it is answered here rather than left.
    """
    return sorted(set(globals()) | _SUBMODULES)
