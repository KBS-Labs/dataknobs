"""The smallest FSM that builds, in both forms the API classes accept.

One start→end machine, spelled twice: the dict ``SimpleFSM`` and
``AsyncSimpleFSM`` take, and the :class:`FSMConfig` ``AdvancedFSM`` takes
through ``FSMBuilder``. Tests about *lifetime* --- what ``close()`` tore
down, which members a class exposes --- need an FSM that builds and nothing
else, so neither the machine nor its configuration is the subject.

A module rather than fixtures in ``conftest.py``: these are wanted in places
a fixture cannot reach --- inside a ``pytest.param`` in a parametrisation,
and twice in one call where a sync and an async class are built from the
same shape. ``_resource_fixtures`` is the same pattern for the same reason.

Both return a fresh object per call. The dict form is handed to loaders that
normalise it in place, so a shared instance would carry one test's edits
into the next.
"""

from __future__ import annotations

from dataknobs_fsm.config.schema import (
    ArcConfig,
    FSMConfig,
    NetworkConfig,
    StateConfig,
)


def trivial_dict() -> dict[str, object]:
    """The FSM in the dict form ``SimpleFSM`` accepts."""
    return {
        "name": "trivial",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [{"from": "start", "to": "end", "name": "go"}],
            }
        ],
    }


def trivial_config() -> FSMConfig:
    """The same FSM as a built config (no transforms, no resources)."""
    return FSMConfig(
        name="trivial",
        main_network="main",
        networks=[
            NetworkConfig(
                name="main",
                states=[
                    StateConfig(name="start", is_start=True, arcs=[ArcConfig(target="end")]),
                    StateConfig(name="end", is_end=True),
                ],
            )
        ],
    )
