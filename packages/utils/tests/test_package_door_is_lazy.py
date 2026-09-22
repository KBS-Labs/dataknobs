# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The package door imports no submodule until one is asked for.

``dataknobs_utils/__init__.py`` imported all fifteen of its modules *"for
easy access"*, and the package declares no optional extras --- so a consumer
reaching **one** function paid for all fifteen twice over, once in the
resolved dependency closure and once in the process. Measured from a
consumer before this changed: that one import line loaded **1,342 modules**
in a fresh interpreter and pulled ``nltk``, ``pandas``, ``numpy``,
``psycopg2``, ``lxml``, ``beautifulsoup4``, ``requests``, ``graphviz``,
``defusedxml``, ``python-dotenv`` and ``json-stream`` behind it.

The chain that found this ended here. Adding ``dataknobs-data`` to a package
declaring ``dataknobs-common`` alone moved third-party distributions 2 -> 34,
and the twenty-eight beyond the four ``dataknobs-data`` declares arrive
behind this door. The only name taken across that last edge is
``quote_ident``, out of ``sql_utils``, and the backend that reaches it is the
**in-memory** one --- so a consumer who touches no database at all installed
and imported a Postgres driver, an HTML parser and a tokeniser corpus.

**This is the import half only.** A lazy door changes nothing about what a
consumer *installs*: that follows from what ``packages/utils/pyproject.toml``
declares, and moving a dependency into an extra is a separate change with a
separate compatibility surface.

The pattern is already in this repository three times with its reason
written out --- ``dataknobs_common.events``, ``dataknobs_bots.knowledge`` and
``dataknobs_llm.tools`` all defer behind a PEP 562 ``__getattr__``. This door
is the one that costs the most and had not adopted it.
"""

from __future__ import annotations

import subprocess
import sys

#: What the door must not drag in on a bare import.
#:
#: Every one is a third-party root reached only through a module the door used
#: to import eagerly, and each is named rather than counted so a failure says
#: which one came back.
HEAVY = (
    "nltk",
    "pandas",
    "numpy",
    "psycopg2",
    "lxml",
    "bs4",
    "requests",
    "graphviz",
    "defusedxml",
    "dotenv",
    "json_stream",
)


def _in_a_fresh_interpreter(program: str) -> str:
    """Run ``program`` in a subprocess and return its stdout.

    A subprocess rather than ``importlib.reload``: the question is what a
    *cold* process pays, and this test's own interpreter has already imported
    most of the list above through the rest of the suite.
    """
    finished = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )
    return finished.stdout.strip()


def test_the_door_pulls_in_no_third_party_dependency() -> None:
    """A bare ``import dataknobs_utils`` reaches none of the heavy roots."""
    found = _in_a_fresh_interpreter(
        "import sys\n"
        "import dataknobs_utils\n"
        f"print(','.join(n for n in {HEAVY!r} if n in sys.modules))\n"
    )

    assert found == "", f"the package door still imports: {found}"


def test_the_door_costs_a_small_number_of_modules() -> None:
    """A bound rather than a measurement, so the number is allowed to drift.

    1,342 before. The bound is what makes this a guard: a submodule import
    added back to the door lands somewhere between here and there, and the
    exact count moves with the standard library's own version.
    """
    loaded = int(
        _in_a_fresh_interpreter("import sys\nimport dataknobs_utils\nprint(len(sys.modules))\n")
    )

    assert loaded < 200, f"the package door loaded {loaded} modules"


def test_every_submodule_is_still_reachable_through_the_door() -> None:
    """The compatibility surface, asserted rather than assumed.

    ``from dataknobs_utils import <submodule>`` is how every call site in
    this repository and both published READMEs reach these modules, and PEP
    562 keeps it working --- but *keeps working* is the whole claim of this
    change, so it is a test rather than a sentence. ``import
    dataknobs_utils.<submodule>`` never went through the door and is
    included for the same reason.
    """
    import dataknobs_utils

    for name in dataknobs_utils.__all__:
        module = getattr(dataknobs_utils, name)
        assert module.__name__ == f"dataknobs_utils.{name}"

    from dataknobs_utils import json_utils, sql_utils

    assert callable(sql_utils.quote_ident)
    assert json_utils.__name__ == "dataknobs_utils.json_utils"


def test_dir_still_lists_the_submodules() -> None:
    """A lazy door is invisible to ``dir()`` unless it says otherwise.

    Tab completion, ``help()`` and every introspecting tool read ``__dir__``,
    and a door that answered without its submodules would be a discoverability
    regression that no import test could see.

    **In a fresh interpreter, and that is the whole test.** Importing a
    submodule binds it onto the package, so in a process where anything has
    already reached one --- which is every process running the rest of this
    file --- ``dir()`` lists it whether or not ``__dir__`` exists. Asserted
    in-process, this passed with ``__dir__`` deleted.
    """
    missing = _in_a_fresh_interpreter(
        "import dataknobs_utils\n"
        "print(','.join(sorted(set(dataknobs_utils.__all__) - set(dir(dataknobs_utils)))))\n"
    )

    assert missing == "", f"the lazy door hides these from dir(): {missing}"


def test_a_name_the_package_does_not_have_still_raises_attribute_error() -> None:
    """The lazy branch must not swallow a typo into an ImportError."""
    import dataknobs_utils
    import pytest

    with pytest.raises(AttributeError, match="no attribute"):
        _ = dataknobs_utils.no_such_utils
