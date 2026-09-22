# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""``StateNetwork.to_dict`` / ``from_dict`` is a round trip, not a summary.

Serialization is one of the reasons the hand-construction API exists: a
consumer that builds a network from a database table or a generated graph
wants to store it and read it back. ``StateDefinition`` gained ``to_dict`` /
``from_dict`` so that a serialized network stopped carrying Python reprs of its
states --- but the arc half of the same payload has two holes of its own, and
one of them is *newly* reachable.

- **The payload is no longer JSON.** ``to_dict`` writes ``arc.transform``
  whole. Before ``add_arc`` accepted a prepared definition, the builder reduced
  a transform to a string on the way into ``_arcs``; now the builder's own arc
  is the stored arc, so a ``TransformSpec`` --- what any config with transform
  ``params`` produces --- reaches the dictionary as a dataclass instance and
  ``json.dumps`` refuses it.
- **A push arc comes back as an ordinary arc.** Nothing in the payload records
  which class the arc was or any of ``target_network`` / ``return_state`` /
  ``isolation_mode`` / ``data_mapping`` / ``result_mapping``. A round-tripped
  sub-network call silently becomes a plain transition to the state named in
  ``target``: the sub-network is never entered and nothing raises.

Both are losses on the path this API exists to serve, and ``_arcs`` holds real
``PushArc`` instances for the first time, so the second is no longer theoretical.
"""

from __future__ import annotations

import json

from dataknobs_fsm.core.arc import ArcDefinition, DataIsolationMode, PushArc, TransformSpec
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import StateDefinition, StateType


def two_state_network() -> StateNetwork:
    """``start`` and ``end``, no arcs yet."""
    network = StateNetwork(name="main")
    network.add_state(StateDefinition(name="start", type=StateType.START), initial=True)
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)
    return network


# --------------------------------------------------------------------------- #
# the payload is data
# --------------------------------------------------------------------------- #


def test_a_network_carrying_a_transform_spec_serializes_to_json() -> None:
    """A transform with params is a ``TransformSpec``, and it must serialize.

    ``transform: {type: registered, name: scale, params: {...}}`` is a
    documented config form; the builder turns it into a ``TransformSpec`` and
    ``add_arc`` now stores that object. ``to_dict`` writing it whole means the
    dictionary is not a dictionary of data, and the caller that asked for one
    finds out at ``json.dumps``.
    """
    network = two_state_network()
    network.add_arc("start", "end", transform=TransformSpec(name="scale", params={"factor": 2}))

    json.dumps(network.to_dict())


def test_a_transform_spec_survives_the_round_trip() -> None:
    """Serializing it is half the job; reading it back is the other half."""
    network = two_state_network()
    network.add_arc("start", "end", transform=TransformSpec(name="scale", params={"factor": 2}))

    restored = StateNetwork.from_dict(json.loads(json.dumps(network.to_dict())))

    arc = restored.get_arcs_from_state("start")[0]
    assert isinstance(arc.transform, TransformSpec)
    assert arc.transform.name == "scale"
    assert arc.transform.params == {"factor": 2}


def test_a_chain_of_transforms_survives_the_round_trip() -> None:
    """The list shape round-trips too, names and specs mixed."""
    network = two_state_network()
    network.add_arc(
        "start",
        "end",
        transform=["normalize", TransformSpec(name="scale", params={"factor": 2})],
    )

    restored = StateNetwork.from_dict(json.loads(json.dumps(network.to_dict())))

    arc = restored.get_arcs_from_state("start")[0]
    assert isinstance(arc.transform, list)
    assert arc.transform[0] == "normalize"
    assert isinstance(arc.transform[1], TransformSpec)
    assert arc.transform[1].params == {"factor": 2}


def test_a_plain_string_transform_still_round_trips_as_a_string() -> None:
    """The common shape is unchanged: a name goes out and comes back a name."""
    network = two_state_network()
    network.add_arc("start", "end", transform="normalize")

    restored = StateNetwork.from_dict(json.loads(json.dumps(network.to_dict())))

    assert restored.get_arcs_from_state("start")[0].transform == "normalize"


# --------------------------------------------------------------------------- #
# a push arc is still a push arc
# --------------------------------------------------------------------------- #


def push_arc_network() -> StateNetwork:
    """``start`` pushes into a sub-network and returns to ``after``."""
    network = StateNetwork(name="main")
    network.add_state(StateDefinition(name="start", type=StateType.START), initial=True)
    network.add_state(StateDefinition(name="after"))
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)
    network.add_arc(
        "start",
        "after",
        definition=PushArc(
            target_state="after",
            target_network="validation:deep_check",
            return_state="after",
            isolation_mode=DataIsolationMode.REFERENCE,
            data_mapping={"outer": "inner"},
            result_mapping={"inner_result": "outer_result"},
        ),
    )
    network.add_arc("after", "end")
    return network


def test_a_push_arc_comes_back_a_push_arc() -> None:
    """Without a class discriminator the sub-network call is silently dropped.

    The round-tripped arc is an ``ArcDefinition`` to ``after`` --- a perfectly
    ordinary transition --- so the sub-network is never entered and no error is
    raised anywhere. Losing a call by deserializing it is the quietest way to
    lose one.
    """
    restored = StateNetwork.from_dict(json.loads(json.dumps(push_arc_network().to_dict())))

    arc = restored.get_arcs_from_state("start")[0]
    assert isinstance(arc, PushArc), f"came back as {type(arc).__name__}"


def test_a_push_arcs_own_fields_survive_the_round_trip() -> None:
    """Being the right class is not enough; it has to carry its own payload."""
    restored = StateNetwork.from_dict(json.loads(json.dumps(push_arc_network().to_dict())))

    arc = restored.get_arcs_from_state("start")[0]
    assert isinstance(arc, PushArc)
    assert arc.target_network == "validation:deep_check"
    assert arc.parse_target() == ("validation", "deep_check")
    assert arc.return_state == "after"
    assert arc.isolation_mode is DataIsolationMode.REFERENCE
    assert arc.data_mapping == {"outer": "inner"}
    assert arc.result_mapping == {"inner_result": "outer_result"}


def test_an_ordinary_arc_does_not_come_back_a_push_arc() -> None:
    """The discriminator has to discriminate in both directions."""
    network = two_state_network()
    network.add_arc("start", "end", definition=ArcDefinition(target_state="end", priority=3))

    restored = StateNetwork.from_dict(json.loads(json.dumps(network.to_dict())))

    arc = restored.get_arcs_from_state("start")[0]
    assert type(arc) is ArcDefinition
    assert arc.priority == 3


def test_the_round_tripped_network_keeps_the_single_writer_invariant() -> None:
    """``from_dict`` builds through ``add_arc``, so the three indexes agree.

    Worth pinning: a reader that appended to ``_arcs`` and to the state
    separately would rebuild the very split this API was repaired to close,
    and every accessor would still agree with itself.
    """
    restored = StateNetwork.from_dict(json.loads(json.dumps(push_arc_network().to_dict())))

    for source in ("start", "after"):
        indexed = restored.get_arcs_from_state(source)
        on_state = restored.states[source].outgoing_arcs
        assert len(indexed) == 1
        assert indexed[0] is on_state[0], f"{source}: two objects where there should be one"
