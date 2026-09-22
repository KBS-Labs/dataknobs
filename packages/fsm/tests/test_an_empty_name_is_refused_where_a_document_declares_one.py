# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A name a configuration document writes is at least one character long.

Every falsy guard downstream reads an empty name as *absent*, and until this
change nothing in the schema stopped a document from declaring one. The two
readings are indistinguishable at the guard and mean opposite things at the
document: "this state needs no resource" against "this state needs the
resource I named", written as ``''`` because a template rendered nothing.

Measured at ``ae014a6f``, one per kind of name, each on a document that loads
and builds today:

- **a resource** declared ``{'name': '', ...}`` and required by a state is
  carried into ``resource_requirements`` and then skipped by both acquisition
  loops in ``AsyncExecutionEngine`` --- each opens on a falsy name and
  ``continue``s --- so the state runs without it and nothing is reported.
- **a state** declared ``{'name': '', 'is_end': True}`` becomes the network's
  only final state (``final_states == {''}``) while
  ``BaseExecutionEngine.is_final_state_common('')`` answers ``False``, so the
  engine's termination check cannot see the FSM's own end.
- **a network** declared ``{'name': ''}`` and named as ``main_network``
  becomes ``fsm.main_network_name == ''``, and ``FSM.get_outgoing_arcs``
  short-circuits on it: ``[]`` where the same FSM under a name answers with
  the arc.

- an **arc** may also require resources, and that list is checked against
  nothing --- ``{'resources': ['']}`` reaches the arc as ``{'': ''}`` and is
  skipped by the same falsy guard. Whether an arc may require a resource the
  document never *declared* is a different question, open elsewhere, and this
  change does not answer it: it refuses the empty name, not the unknown one.

The FSM's own name has no such guard today; it is refused here because this is
one rule about names rather than four accommodations for the places a falsy
check happens to sit. Nothing forbade the rule earlier: ``git log -S'min_length'``
over ``config/schema.py`` returns no commits, and all four ``name: str`` lines
arrive together in ``42119f99``, argued nowhere. The file had already reached
this conclusion once --- ``FunctionReference.validate_reference`` refuses a
builtin or registered function on ``not self.name``, which an empty string is
--- for the one name it happened to think about.

The line is drawn at *empty*, not at *short* or *untidy*: a one-character name
is a name, and a whitespace name is truthy, so no guard downstream mistakes it
for an absent one. Both are pinned below so that moving the line is a decision
someone makes rather than a side effect.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.core.fsm import FSM

#: A document that loads, builds and runs, with one of every kind of name in
#: it. Each test empties exactly one of them.
WELL_NAMED: dict[str, Any] = {
    "name": "named",
    "main_network": "main",
    "resources": [
        {"name": "db", "type": "database", "config": {"backend": "memory"}},
    ],
    "networks": [
        {
            "name": "main",
            "states": [
                {
                    "name": "start",
                    "is_start": True,
                    "resources": ["db"],
                    "arcs": [{"target": "done"}],
                },
                {"name": "done", "is_end": True},
            ],
        }
    ],
}


def document(**edits: Any) -> dict[str, Any]:
    """``WELL_NAMED`` with one name emptied, chosen by keyword."""
    config = copy.deepcopy(WELL_NAMED)
    for what, name in edits.items():
        if what == "fsm":
            config["name"] = name
        elif what == "resource":
            config["resources"][0]["name"] = name
            config["networks"][0]["states"][0]["resources"] = [name]
        elif what == "network":
            config["networks"][0]["name"] = name
            config["main_network"] = name
        elif what == "state":
            config["networks"][0]["states"][1]["name"] = name
            config["networks"][0]["states"][0]["arcs"] = [{"target": name}]
        elif what == "arc_resources":
            config["networks"][0]["states"][0]["arcs"][0]["resources"] = name
        else:  # pragma: no cover - a typo in a test's own keyword
            raise AssertionError(f"no such name in the document: {what}")
    return config


def refused(config: dict[str, Any]) -> set[str]:
    """Load ``config``, require a length refusal, answer the fields refused.

    The field path is the assertion rather than the message: it is what tells
    the author of a document which line to look at, and it survives a change
    to pydantic's wording.

    ``ArcConfig.resources`` is a union --- a list of names or a ``{role: name}``
    map --- so when one branch reports the empty name the other reports the
    shape it did not get. That second error is the union saying so, not a
    second finding, which is why only the length refusals are answered and the
    rest are merely required to be that.
    """
    with pytest.raises(ValidationError) as excinfo:
        ConfigLoader().load_from_dict(config)
    errors = [
        (error["type"], ".".join(str(part) for part in error["loc"]))
        for error in excinfo.value.errors()
    ]
    paths = {path for kind, path in errors if kind == "string_too_short"}
    assert paths, errors
    assert {kind for kind, _ in errors} <= {
        "string_too_short",
        "list_type",
        "dict_type",
    }, errors
    return paths


def built(config: dict[str, Any]) -> FSM:
    """``config`` all the way through the builder, for the names still allowed."""
    return FSMBuilder().build(ConfigLoader().load_from_dict(config))


def test_a_resource_declared_with_an_empty_name_is_refused() -> None:
    """The measured case: accepted, carried, then silently never acquired.

    Both ends of it are named: the declaration, and the state that requires it
    under the same empty name. The second would be refused anyway once the
    first is --- a state may only require a resource the document declares ---
    but it is refused *as a name*, which is what puts the document's own line
    in the message.
    """
    assert refused(document(resource="")) == {
        "resources.0.name",
        "networks.0.states.0.resources.0",
    }


def test_a_state_declared_with_an_empty_name_is_refused() -> None:
    """It became the network's only final state, which the engine could not see.

    The arc pointing at it is refused too. That path says which field of which
    arc but not *which* arc, because ``StateConfig.validate_arcs`` builds each
    one in a ``mode="before"`` validator and the index is lost on the way out.
    That predates this change and is pinned here as it stands rather than
    asserted as if it read better than it does.
    """
    assert refused(document(state="")) == {
        "networks.0.states.1.name",
        "networks.0.states.0.arcs.target",
    }


def test_a_network_declared_with_an_empty_name_is_refused() -> None:
    """``main_network`` pointed at it, and the FSM then reported no arcs."""
    assert refused(document(network="")) == {"networks.0.name", "main_network"}


def test_an_fsm_declared_with_an_empty_name_is_refused() -> None:
    """No falsy guard reads this one today --- it is refused by the same rule.

    ``BaseExecutionEngine`` does look a network up under it
    (``self.fsm.networks.get(self.fsm.name)``), which an empty name can no
    longer match now that no network may carry one either.
    """
    assert refused(document(fsm="")) == {"name"}


def test_an_arc_requiring_an_empty_resource_name_is_refused() -> None:
    """The arc's list is validated against no declared set, so it needs its own.

    A state's resources are checked for membership in the document's declared
    resources, which is what makes an empty one unreachable there once no
    resource may be declared empty. An arc's are checked nowhere, so the same
    empty name reached the arc as ``{'': ''}`` and was skipped by the engine's
    falsy guard.
    """
    (path,) = refused(document(arc_resources=[""]))
    assert path.startswith("networks.0.states.0.arcs"), path
    assert "resources" in path, path


def test_an_arc_binding_a_role_to_an_empty_resource_name_is_refused() -> None:
    """The role-map form of the same field: the *resource* is what must be named.

    The role is the document's own vocabulary for a slot
    (``FunctionContext.resource_for_role``); the value is the resource it
    binds, and it is the value a falsy guard drops.
    """
    (path,) = refused(document(arc_resources={"database": ""}))
    assert path.startswith("networks.0.states.0.arcs"), path
    assert path.endswith("database"), path


def test_a_name_of_one_character_is_a_name() -> None:
    """The rule is about empty, not short."""
    fsm = built(document(fsm="f", resource="d", network="n", state="s"))

    assert fsm.main_network_name == "n"
    assert set(fsm.networks["n"].states) == {"start", "s"}
    assert fsm.networks["n"].final_states == {"s"}


def test_a_name_of_only_whitespace_is_still_accepted() -> None:
    """Where the line is, recorded so that moving it is a decision.

    ``'  '`` is untidy and is almost certainly a typo, but it is *truthy*: no
    guard downstream mistakes it for an absent name, so it is not the defect
    this change answers. Refusing it would be a separate judgement about what
    a name may contain, and this test is what a change of that judgement has
    to come through.
    """
    fsm = built(document(resource="  "))

    requirements = fsm.networks["main"].states["start"].resource_requirements
    assert [rc.name for rc in requirements] == ["  "]
