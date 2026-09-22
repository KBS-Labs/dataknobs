# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Every writer and every reader of a network's arcs agrees with the others.

``StateNetwork`` holds its arcs three times over --- ``_arcs``, ``_arc_index``
and each source state's ``outgoing_arcs`` --- and making ``add_arc`` the single
writer of all three fixed the way they came apart. It did not fix the ways the
*other* members come apart from each other:

- ``add_state`` accepts a state whose ``outgoing_arcs`` were populated before it
  was added --- a path :meth:`StateDefinition.add_outgoing_arc` documents --- and
  never records those arcs in the network's own two indexes. The engines follow
  the arc; ``validate()`` says the state has none.
- ``remove_arc`` removes by *equality*, so it will take an arc the caller never
  held when a structurally identical one is present.
- The ``arcs`` property is keyed ``"source:target"``, which collapses the
  ordinary branch-on-condition shape --- two arcs between the same pair of
  states --- and three accessors report through it.
- ``_update_resource_requirements`` reads ``state.resource_requirements``, a
  ``List[ResourceConfig]``, as though it were a single object with
  ``.databases`` / ``.filesystems`` attributes. It is not, so the body never
  runs and the network's resource summary is empty for every network ever
  built.

Each of these is the same failure as the one already fixed: a structure
maintained by one writer and read by another that disagrees with it.
"""

from __future__ import annotations

import json

from dataknobs_fsm.core.arc import ArcDefinition
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import StateDefinition, StateType
from dataknobs_fsm.functions.base import ResourceConfig


def two_state_network() -> StateNetwork:
    """``start -> end``, with both states present and nothing else."""
    network = StateNetwork(name="main")
    network.add_state(StateDefinition(name="start", type=StateType.START), initial=True)
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)
    return network


# --------------------------------------------------------------------------- #
# add_state: a state assembled before it joins the network
# --------------------------------------------------------------------------- #


def test_a_state_assembled_before_it_is_added_reaches_every_index() -> None:
    """``add_state`` ingests the arcs the state already carries.

    ``StateDefinition.add_outgoing_arc`` is public and documented as the way to
    assemble a state before adding it. It writes ``outgoing_arcs`` --- the list
    the engines read --- and nothing else, so unless ``add_state`` ingests what
    it finds there, the network ends up in exactly the state this PR set out to
    eliminate: the engine can follow an arc that the network's own accessors
    say does not exist.
    """
    network = StateNetwork(name="main")
    start = StateDefinition(name="start", type=StateType.START)
    start.add_outgoing_arc(ArcDefinition(target_state="end"))

    network.add_state(start, initial=True)
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)

    arc = start.outgoing_arcs[0]
    assert network.get_arcs_from_state("start") == [arc]
    assert network.get_arcs_from_state("start")[0] is arc, "the same object, not a copy"
    assert arc.source_state == "start", "add_state stamps the source it now knows"


def test_a_state_assembled_before_it_is_added_validates() -> None:
    """The accessors agree with the engine, which is what ``validate`` reports.

    Without ingestion this returns two errors --- "Non-final state 'start' has
    no outgoing arcs" and "State 'end' is unreachable from initial state" ---
    about a network that runs correctly.
    """
    network = StateNetwork(name="main")
    start = StateDefinition(name="start", type=StateType.START)
    start.add_outgoing_arc(ArcDefinition(target_state="end"))
    network.add_state(start, initial=True)
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)

    is_valid, errors = network.validate()

    assert is_valid, errors


def test_an_ingested_arc_can_be_removed_like_any_other() -> None:
    """Ingestion puts the arc under the network's management, not beside it."""
    network = StateNetwork(name="main")
    start = StateDefinition(name="start", type=StateType.START)
    start.add_outgoing_arc(ArcDefinition(target_state="end"))
    network.add_state(start, initial=True)
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)

    network.remove_arc(start.outgoing_arcs[0])

    assert network.get_arcs_from_state("start") == []
    assert start.outgoing_arcs == []


def test_an_arc_to_a_state_that_never_arrives_is_reported() -> None:
    """A dangling arc is a broken network, and ``validate`` is where it shows.

    Ingestion cannot check the target the way ``add_arc`` does --- states are
    added in whatever order the caller has them, so an arc's target routinely
    does not exist yet. The check moves to ``validate()``, which runs once the
    network is whole; without it, ingestion would be a way to put an
    unfollowable arc into the network with nothing ever saying so.
    """
    network = StateNetwork(name="main")
    start = StateDefinition(name="start", type=StateType.START)
    start.add_outgoing_arc(ArcDefinition(target_state="nowhere"))
    network.add_state(start, initial=True)
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)

    is_valid, errors = network.validate()

    assert not is_valid
    assert any("nowhere" in error for error in errors), errors


# --------------------------------------------------------------------------- #
# remove_arc: identity, not equality
# --------------------------------------------------------------------------- #


def test_removing_an_equal_arc_does_not_remove_the_one_that_is_held() -> None:
    """Removal is by identity, because the writer's guarantee is identity.

    ``add_arc`` stores *the same object* in all three indexes precisely so two
    arcs cannot drift. Removing by equality gives that up at the other end: a
    freshly built ``ArcDefinition`` with the same fields is ``==`` to the stored
    one, so ``remove_arc`` accepts it and silently removes an arc the caller
    never held.
    """
    network = two_state_network()
    real = network.add_arc("start", "end")
    impostor = ArcDefinition(target_state="end", source_state="start")
    assert impostor == real, "the premise: these are equal but not the same arc"

    try:
        network.remove_arc(impostor)
    except ValueError:
        pass
    else:
        raise AssertionError("remove_arc accepted an arc the network does not hold")

    assert network.get_arcs_from_state("start") == [real]
    assert network.states["start"].outgoing_arcs == [real]


def test_removing_one_of_two_equal_arcs_removes_exactly_one() -> None:
    """Two equal arcs are two arcs; removing one leaves the other."""
    network = two_state_network()
    first = network.add_arc("start", "end")
    second = network.add_arc("start", "end")

    network.remove_arc(second)

    remaining = network.get_arcs_from_state("start")
    assert len(remaining) == 1
    assert remaining[0] is first
    assert network.states["start"].outgoing_arcs[0] is first


def test_removing_a_state_prunes_by_identity_too() -> None:
    """``remove_state`` prunes the same three indexes and has the same hazard."""
    network = StateNetwork(name="main")
    network.add_state(StateDefinition(name="start", type=StateType.START), initial=True)
    network.add_state(StateDefinition(name="middle"))
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)
    kept = network.add_arc("start", "end")
    network.add_arc("start", "middle")
    network.add_arc("middle", "end")

    network.remove_state("middle")

    assert network.get_arcs_from_state("start") == [kept]
    assert network.states["start"].outgoing_arcs == [kept]
    assert "middle" not in network.states


# --------------------------------------------------------------------------- #
# parallel arcs: the branch-on-condition shape
# --------------------------------------------------------------------------- #


def parallel_arc_fsm() -> FSM:
    """``start`` branches to ``end`` on either of two conditions."""
    fsm = FSM(name="branching")
    network = two_state_network()
    network.add_arc("start", "end", pre_test="is_premium")
    network.add_arc("start", "end", pre_test="is_basic")
    fsm.add_network(network)
    return fsm


def test_two_arcs_between_the_same_pair_are_both_reported() -> None:
    """``priority`` and ``definition_order`` exist to order exactly this shape.

    Two arcs from ``start`` to ``end``, taken on different conditions, is the
    ordinary way to branch. The ``arcs`` property keys on ``"source:target"``,
    so it answers with one of them --- and three accessors read through it.
    """
    network = parallel_arc_fsm().main_network
    assert network is not None

    assert len(network.arc_definitions) == 2
    assert {arc.pre_test for arc in network.arc_definitions} == {"is_premium", "is_basic"}


def test_a_branch_does_not_lose_a_function_reference() -> None:
    """Every ``pre_test`` an arc names is a function the FSM must have.

    ``_get_all_function_references`` reads through the collapsing property, so
    the second branch's condition was never counted --- and ``validate()``,
    which checks every referenced function is registered, could not report it
    missing.
    """
    fsm = parallel_arc_fsm()
    try:
        _, errors = fsm.validate()
        reported = " ".join(errors)
        assert "is_premium" in reported, errors
        assert "is_basic" in reported, errors
    finally:
        fsm.close()


def test_the_arc_count_counts_every_arc() -> None:
    """``total_arcs`` and ``get_all_arcs`` report the graph, not the key set."""
    fsm = parallel_arc_fsm()
    try:
        assert fsm.get_resource_summary()["total_arcs"] == 2
        assert len(fsm.get_all_arcs()["main"]) == 2
    finally:
        fsm.close()


# --------------------------------------------------------------------------- #
# resource requirements: a list read as an object
# --------------------------------------------------------------------------- #


def test_a_states_resources_reach_the_networks_summary() -> None:
    """``resource_requirements`` is a list of configs, and is aggregated as one.

    ``_update_resource_requirements`` asked the *list* for ``.databases``,
    ``.filesystems``, ``.http_services`` and ``.llms``. A list has none of them,
    so every branch was skipped and the summary was empty for every network
    ever built --- including every network the config builder produces.
    """
    network = StateNetwork(name="main")
    network.add_state(
        StateDefinition(
            name="start",
            type=StateType.START,
            resource_requirements=[
                ResourceConfig(name="main_db", type="database", connection_params={}),
                ResourceConfig(name="scratch", type="filesystem", connection_params={}),
                ResourceConfig(name="summarizer", type="llm", connection_params={}),
                ResourceConfig(name="embeddings", type="vector_store", connection_params={}),
            ],
        ),
        initial=True,
    )

    reqs = network.get_resource_requirements()

    assert reqs.databases == {"main_db"}
    assert reqs.filesystems == {"scratch"}
    assert reqs.llms == {"summarizer"}
    assert reqs.custom.get("vector_store") == {"embeddings"}


def test_removing_a_state_takes_its_resources_with_it() -> None:
    """The recalculation path aggregates through the same reader."""
    network = StateNetwork(name="main")
    network.add_state(
        StateDefinition(
            name="start",
            type=StateType.START,
            resource_requirements=[
                ResourceConfig(name="main_db", type="database", connection_params={})
            ],
        ),
        initial=True,
    )
    network.add_state(
        StateDefinition(
            name="end",
            type=StateType.END,
            resource_requirements=[
                ResourceConfig(name="archive", type="database", connection_params={})
            ],
        ),
        final=True,
    )
    assert network.get_resource_requirements().databases == {"main_db", "archive"}

    network.remove_state("end")

    assert network.get_resource_requirements().databases == {"main_db"}


# --------------------------------------------------------------------------- #
# find_cycles: named as public API in three places
# --------------------------------------------------------------------------- #


def test_find_cycles_is_reachable_under_the_name_it_is_published_as() -> None:
    """The PR body, the CHANGELOG and a test docstring all name ``find_cycles``.

    It was defined as ``_find_cycles``, so the construction API those three
    documents describe was not the one the class offered.
    """
    network = StateNetwork(name="main")
    network.add_state(StateDefinition(name="a", type=StateType.START), initial=True)
    network.add_state(StateDefinition(name="b"))
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)
    network.add_arc("a", "b")
    network.add_arc("b", "a")
    network.add_arc("b", "end")

    cycles = network.find_cycles()

    assert cycles, "a -> b -> a is a cycle"
    assert any("a" in cycle and "b" in cycle for cycle in cycles)


def test_a_networks_dictionary_form_is_json() -> None:
    """``to_dict`` is a serialization entry point, so its output must serialize."""
    network = two_state_network()
    network.add_arc("start", "end", pre_test="is_ready")

    json.dumps(network.to_dict())


def test_a_network_configured_to_stream_says_so() -> None:
    """``streaming: {enabled: true}`` is a config key, and it now reaches the network.

    Nothing read it. ``supports_streaming`` was fed by a branch asking the
    state's ``resource_requirements`` *list* for a ``streaming_enabled``
    attribute, so the flag had no source and every network answered ``False``
    --- including one whose config asked for streaming in as many words.
    """
    from dataknobs_fsm.config.builder import build_fsm

    fsm = build_fsm(
        {
            "name": "streamer",
            "main_network": "main",
            "networks": [
                {
                    "name": "main",
                    "streaming": {"enabled": True, "chunk_size": 10},
                    "states": [
                        {"name": "start", "is_start": True, "arcs": [{"target": "end"}]},
                        {"name": "end", "is_end": True},
                    ],
                }
            ],
        }
    )
    try:
        assert fsm.networks["main"].supports_streaming is True
        assert fsm.supports_streaming() is True
    finally:
        fsm.close()


def test_a_network_not_configured_to_stream_does_not_claim_to() -> None:
    """The flag has to discriminate, not just be settable."""
    from dataknobs_fsm.config.builder import build_fsm

    fsm = build_fsm(
        {
            "name": "plain",
            "main_network": "main",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {"name": "start", "is_start": True, "arcs": [{"target": "end"}]},
                        {"name": "end", "is_end": True},
                    ],
                }
            ],
        }
    )
    try:
        assert fsm.networks["main"].supports_streaming is False
    finally:
        fsm.close()


def test_the_same_arc_object_cannot_be_added_twice() -> None:
    """One arc, one source. ``add_arc`` stamps ``source_state`` onto the object.

    Adding the same object again re-stamps it and leaves it in two sources'
    indexes carrying one source --- and ``source_state`` participates in
    ``__hash__``, so anything already holding the arc in a set or dict loses
    track of it at the same moment.
    """
    network = StateNetwork(name="main")
    network.add_state(StateDefinition(name="start", type=StateType.START), initial=True)
    network.add_state(StateDefinition(name="other"))
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)

    arc = network.add_arc("start", "end")

    try:
        network.add_arc("other", "end", definition=arc)
    except ValueError:
        pass
    else:
        raise AssertionError("the same arc object was added under a second source")

    assert arc.source_state == "start"
    assert network.get_arcs_from_state("other") == []


def test_a_hand_stamped_arc_that_is_not_in_the_network_is_still_accepted() -> None:
    """The cheap pre-check must not become a refusal of its own.

    ``source_state`` is a public field; an arc carrying one it was given by
    hand has not been added to anything, and the guard has to tell that apart
    from a genuine repeat rather than reading the stamp as proof.
    """
    network = two_state_network()
    prepared = ArcDefinition(target_state="end", source_state="somewhere_else")

    stored = network.add_arc("start", "end", definition=prepared)

    assert stored is prepared
    assert prepared.source_state == "start", "add_arc restamps it with the real source"
