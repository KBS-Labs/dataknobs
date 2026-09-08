"""A ``Hierarchy`` and its drivers reach no ``dataknobs_data``.

Placement in this project is by **dependency**: a concrete goes where the thing
it opens lives. ``AssertionHierarchy`` is under ``ontology/`` because reading an
edge means asking at runtime whether an assertion's object is an entity or a
literal; a database-backed one belongs in ``dataknobs-data`` because it holds a
handle. What keeps that rule from being a convention is a check that the
*general* layer stands up without the specific ones.

Two halves, and the second is the one that could not be written before a
mapping backing existed. Half (a) asks what importing the module pulls in. Half
(b) asks what **building and walking a real hierarchy** pulls in -- which is a
different question, because the leaks this rule exists to catch are the ones no
import statement shows: a value type constructed lazily, a helper reaching for a
backend on a branch nobody took at import time. ``test_record_core_independence``
is the form, and it exists because exactly that happened one module over.

Fresh interpreters, because ``sys.modules`` in this one has the whole suite in it.
"""

from __future__ import annotations

import subprocess
import sys


def _run(script: str) -> subprocess.CompletedProcess[str]:
    """Run a script in a fresh interpreter, returning the completed process.

    Not ``check=True``: the assertion carries stdout and stderr, which a
    ``CalledProcessError`` would replace with an exit status.
    """
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


_LEAK_CHECK = (
    "leaked = sorted(m for m in sys.modules if m.startswith('dataknobs_data'))\n"
    "assert not leaked, leaked\n"
)


def test_importing_the_structure_axis_does_not_import_dataknobs_data() -> None:
    """Half (a): the modules themselves, including the walk and nesting cores."""
    result = _run(
        "import sys\n"
        "import dataknobs_common.hierarchy\n"
        "import dataknobs_common.taxonomy\n"
        "import dataknobs_common._walk_core\n"
        "import dataknobs_common._nested_core\n" + _LEAK_CHECK
    )

    assert result.returncode == 0, (
        f"dataknobs_data leaked into the hierarchy import path:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_building_and_walking_a_hierarchy_does_not_import_dataknobs_data() -> None:
    """Half (b): a concrete built, driven both flavours, and snapshotted.

    Every entry point in one interpreter -- the two drivers, the public walk
    pair, both constructors and both snapshots -- because a leak on one branch
    is invisible to a probe that takes another.
    """
    result = _run(
        "import asyncio, sys\n"
        "from dataknobs_common.hierarchy import (\n"
        "    AsyncMappingHierarchy, MappingHierarchy, ancestors, async_ancestors,\n"
        ")\n"
        "tree = {'name': 'Mammal', 'children': [{'name': 'Dog',"
        " 'children': [{'name': 'Beagle'}]}]}\n"
        "axis = MappingHierarchy.from_nested(tree)\n"
        "assert ancestors(axis, 'mammal/dog/beagle') == ('mammal/dog', 'mammal')\n"
        "assert MappingHierarchy.snapshot(axis).roots() == ('mammal',)\n"
        "async_axis = AsyncMappingHierarchy.from_nested(tree)\n"
        "assert asyncio.run(async_ancestors(async_axis, 'mammal/dog/beagle')) == (\n"
        "    'mammal/dog', 'mammal',\n"
        ")\n"
        "snap = asyncio.run(AsyncMappingHierarchy.snapshot(async_axis))\n"
        "assert asyncio.run(snap.roots()) == ('mammal',)\n" + _LEAK_CHECK
    )

    assert result.returncode == 0, (
        f"walking a hierarchy reached dataknobs_data:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
