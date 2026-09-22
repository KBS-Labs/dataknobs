# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""State network implementation for FSM."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Set, Tuple

from dataknobs_fsm.core.arc import ArcDefinition, arc_from_dict, TransformSpec
from dataknobs_fsm.core.state import StateDefinition


@dataclass
class NetworkResourceRequirements:
    """Aggregated resource requirements for a network.

    Attributes:
        databases: Set of required database resources.
        filesystems: Set of required filesystem resources.
        http_services: Set of required HTTP service resources.
        llms: Set of required LLM resources.
        custom: Dictionary of custom resource requirements.
        streaming_enabled: Whether any state requires streaming.
        estimated_memory_mb: Estimated memory requirement in MB.
    """

    databases: Set[str] = field(default_factory=set)
    filesystems: Set[str] = field(default_factory=set)
    http_services: Set[str] = field(default_factory=set)
    llms: Set[str] = field(default_factory=set)
    custom: Dict[str, Set[str]] = field(default_factory=dict)
    streaming_enabled: bool = False
    estimated_memory_mb: int = 0

    def merge(self, other: "NetworkResourceRequirements") -> None:
        """Merge another set of requirements into this one.

        Args:
            other: Requirements to merge.
        """
        self.databases.update(other.databases)
        self.filesystems.update(other.filesystems)
        self.http_services.update(other.http_services)
        self.llms.update(other.llms)

        for key, values in other.custom.items():
            if key not in self.custom:
                self.custom[key] = set()
            self.custom[key].update(values)

        self.streaming_enabled = self.streaming_enabled or other.streaming_enabled
        self.estimated_memory_mb = max(self.estimated_memory_mb, other.estimated_memory_mb)

    def is_empty(self) -> bool:
        """Check if there are no resource requirements.

        Returns:
            True if no resources are required.
        """
        return (
            not self.databases
            and not self.filesystems
            and not self.http_services
            and not self.llms
            and not self.custom
        )


class StateNetwork:
    """Represents a network of states and their transitions.

    A state network is a directed graph where nodes are states
    and edges are arcs (transitions) between states.
    """

    def __init__(self, name: str, description: str | None = None):
        """Initialize state network.

        Args:
            name: Network name/identifier.
            description: Optional network description.
        """
        self.name = name
        self.description = description

        # State management
        self._states: Dict[str, StateDefinition] = {}
        self._initial_state: str | None = None
        self._final_states: Set[str] = set()

        # Arc management
        self._arcs: List[ArcDefinition] = []
        self._arc_index: Dict[str, List[ArcDefinition]] = {}  # source_state -> [arcs]

        # Resource tracking
        self._resource_requirements = NetworkResourceRequirements()
        self._streaming_enabled = False

        # Validation cache
        self._validation_cache: Dict[str, Any] | None = None

    @property
    def states(self) -> Dict[str, StateDefinition]:
        """Get all states in the network."""
        return self._states

    @property
    def arc_definitions(self) -> List[ArcDefinition]:
        """Every arc in the network, in the order it was added.

        The accessor to reach for. :attr:`arcs` keys on ``"source:target"``
        and so cannot represent two arcs between the same pair of states ---
        which is the ordinary way to branch, and the shape ``priority`` and
        ``definition_order`` exist to order.
        """
        return list(self._arcs)

    @property
    def arcs(self) -> Dict[str, ArcDefinition]:
        """The network's arcs, keyed ``"source:target"``.

        This used to convert each stored ``Arc`` into a freshly built
        ``ArcDefinition`` "for compatibility", once per call and per arc, and
        the conversion was lossy: ``priority``, ``definition_order`` and
        ``required_resources`` had nowhere to go, so the arcs this returned
        would have been evaluated in a different order than the ones the
        engines hold. There is one arc type now, so this hands back the stored
        arcs themselves.

        **The key still collapses parallel arcs.** Two arcs from ``start`` to
        ``end``, taken on different conditions, share a key and only one
        survives the mapping --- so this is a view of the network's *edges*,
        not of its arcs, and a caller counting arcs or collecting the
        functions they name wants :attr:`arc_definitions` instead.
        """
        return {f"{arc.source_state}:{arc.target_state}": arc for arc in self._arcs}

    @property
    def initial_states(self) -> Set[str]:
        """Get initial states (returns set for compatibility)."""
        if self._initial_state:
            return {self._initial_state}
        return set()

    @property
    def final_states(self) -> Set[str]:
        """Get final states."""
        return self._final_states.copy()

    def is_initial_state(self, state_name: str) -> bool:
        """Check if a state is an initial state.

        Args:
            state_name: Name of the state to check

        Returns:
            True if the state is an initial state
        """
        return self._initial_state == state_name

    def is_final_state(self, state_name: str) -> bool:
        """Check if a state is a final state.

        Args:
            state_name: Name of the state to check

        Returns:
            True if the state is a final state
        """
        return state_name in self._final_states

    @property
    def resource_requirements(self) -> Dict[str, Any]:
        """Get resource requirements."""
        return {
            "databases": self._resource_requirements.databases,
            "filesystems": self._resource_requirements.filesystems,
            "http_services": self._resource_requirements.http_services,
            "llms": self._resource_requirements.llms,
            "custom": self._resource_requirements.custom,
        }

    @property
    def supports_streaming(self) -> bool:
        """Check if network supports streaming."""
        return self._streaming_enabled

    def add_state(self, state: StateDefinition, initial: bool = False, final: bool = False) -> None:
        """Add a state to the network.

        Args:
            state: State to add.
            initial: Mark as initial state.
            final: Mark as final state.

        Raises:
            ValueError: If state with same name already exists.
        """
        if state.name in self._states:
            raise ValueError(f"State '{state.name}' already exists in network")

        self._states[state.name] = state

        # A state may arrive with arcs already on it. ``add_outgoing_arc`` is
        # public and documented as the way to assemble a state before adding
        # it, and it writes ``outgoing_arcs`` --- the list the engines read ---
        # and nothing else. Ingesting them here is what keeps the network's own
        # two indexes from disagreeing with the engine about what this state
        # can do, which is the same split ``add_arc`` was made the single
        # writer to close, arrived at from the other side.
        #
        # The targets are not checked the way ``add_arc`` checks them: states
        # are added in whatever order the caller has them, so an arc's target
        # routinely does not exist yet. ``validate()`` carries that check
        # instead, once the network is whole.
        for arc in list(state.outgoing_arcs):
            self._index_arc(state.name, arc)

        if initial:
            if self._initial_state:
                raise ValueError(f"Initial state already set to '{self._initial_state}'")
            self._initial_state = state.name

        if final:
            self._final_states.add(state.name)

        # Update resource requirements
        self._update_resource_requirements(state)

        # Invalidate validation cache
        self._validation_cache = None

    def remove_state(self, state_name: str) -> None:
        """Remove a state from the network.

        Args:
            state_name: Name of state to remove.

        Raises:
            KeyError: If state doesn't exist.
        """
        if state_name not in self._states:
            raise KeyError(f"State '{state_name}' not found in network")

        # Remove state
        del self._states[state_name]

        # Remove from initial/final if needed
        if self._initial_state == state_name:
            self._initial_state = None
        self._final_states.discard(state_name)

        # Remove arcs involving this state. The state's own outgoing arcs go
        # with it; the arcs *into* it are held by other states, so those lists
        # are pruned too --- an arc surviving on a source state's
        # ``outgoing_arcs`` after its target is gone is one the engine would
        # still offer and then fail to follow.
        removed = [
            arc
            for arc in self._arcs
            if arc.source_state == state_name or arc.target_state == state_name
        ]
        # Identity again: an arc equal to a removed one is not a removed one.
        removed_ids = {id(arc) for arc in removed}
        self._arcs = [arc for arc in self._arcs if id(arc) not in removed_ids]
        for arc in removed:
            source = self._states.get(arc.source_state)
            if source is not None:
                source.outgoing_arcs[:] = [held for held in source.outgoing_arcs if held is not arc]

        # Rebuild arc index
        self._rebuild_arc_index()

        # Recalculate resource requirements
        self._recalculate_resource_requirements()

        # Invalidate validation cache
        self._validation_cache = None

    def add_arc(
        self,
        source_state: str,
        target_state: str,
        pre_test: str | None = None,
        transform: "str | TransformSpec | list[str | TransformSpec] | None" = None,
        metadata: Dict[str, Any] | None = None,
        definition: ArcDefinition | None = None,
    ) -> ArcDefinition:
        """Add an arc between two states, in every index the network keeps.

        **This is the only writer.** The network holds its arcs twice over --- in
        ``_arcs`` / ``_arc_index``, which its own accessors read, and on the
        source state's ``outgoing_arcs``, which is the list the execution
        engines read and the *only* one they read. Maintaining one and not the
        other is how a network could pass ``validate()``, answer
        ``get_arcs_from_state()`` and report no transitions at execution time.
        So one call writes one object into all of them; they are the same
        object, not equal copies, because copies drift the first time one is
        edited.

        Args:
            source_state: Source state name.
            target_state: Target state name.
            pre_test: Optional pre-test function name.
            transform: Optional transform function name.
            metadata: Optional arc metadata.
            definition: An already-built arc to store instead of constructing
                one from the arguments above. A caller that has resolved
                functions, a priority or a definition order --- as
                :class:`~dataknobs_fsm.config.builder.FSMBuilder` has ---
                passes it here, so its arc is *the* arc rather than a second
                one built alongside a lossy copy. ``source_state`` is stamped
                onto it; ``target_state`` must agree with the argument.

        Returns:
            The stored arc: ``definition`` when one was given, else the arc
            built from the arguments.

        Raises:
            ValueError: If either state is unknown, or if ``definition``
                names a different target than ``target_state``.
        """
        if source_state not in self._states:
            raise ValueError(f"Source state '{source_state}' not found")
        if target_state not in self._states:
            raise ValueError(f"Target state '{target_state}' not found")
        if definition is not None and definition.target_state != target_state:
            raise ValueError(
                f"Arc definition targets '{definition.target_state}' but was added "
                f"as an arc to '{target_state}'"
            )

        arc = (
            definition
            if definition is not None
            else ArcDefinition(
                target_state=target_state,
                pre_test=pre_test,
                transform=transform,
                metadata=metadata or {},
            )
        )
        # An arc belongs to one source in one network, because ``_index_arc``
        # stamps ``source_state`` onto it: adding the same object twice
        # re-stamps it and leaves it in both places with one source, and its
        # ``__hash__`` (which ``source_state`` participates in) changes under
        # anything already holding it. An arc that has never been added has an
        # empty ``source_state``, so the common path settles in one comparison
        # and the scan runs only for an arc that might genuinely be a repeat
        # --- worth keeping cheap, since ``from_dict`` calls this once per arc
        # and a generated graph is exactly what this API is for.
        if arc.source_state and any(held is arc for held in self._arcs):
            raise ValueError(
                f"Arc to '{arc.target_state}' is already in this network; an "
                f"ArcDefinition belongs to one source in one network, because "
                f"add_arc stamps source_state onto it"
            )

        self._index_arc(source_state, arc)

        return arc

    def _index_arc(self, source_state: str, arc: ArcDefinition) -> None:
        """Record one arc in all three indexes, as one object.

        The write half of :meth:`add_arc`, separated so :meth:`add_state` can
        ingest a state's pre-existing ``outgoing_arcs`` through it instead of
        repeating it. Two writers of three indexes is what this class had
        before; two *callers* of one writer is not the same thing.

        The state's own list is appended to only if the arc is not already on
        it --- ingestion reaches arcs that are --- and the check is by
        identity, because an equal arc is a different arc.

        Args:
            source_state: The state the arc leaves; stamped onto the arc.
            arc: The arc to record.
        """
        arc.source_state = source_state

        self._arcs.append(arc)
        self._arc_index.setdefault(source_state, []).append(arc)

        state_arcs = self._states[source_state].outgoing_arcs
        if not any(held is arc for held in state_arcs):
            state_arcs.append(arc)

        # Invalidate validation cache
        self._validation_cache = None

    def remove_arc(self, arc: ArcDefinition) -> None:
        """Remove an arc from every index :meth:`add_arc` wrote it to.

        Args:
            arc: Arc to remove.

        Raises:
            ValueError: If arc doesn't exist.
        """
        # By identity, for the same reason ``add_arc`` stores one object in
        # three indexes rather than three equal ones. ``ArcDefinition`` is a
        # dataclass, so two arcs between the same pair of states with the same
        # pre-test are ``==``; removing by equality means a freshly built arc
        # is accepted here and something the caller never held is removed
        # instead --- and where two equal arcs are both present, the survivor
        # is chosen by list order rather than by which one was asked for.
        if not any(held is arc for held in self._arcs):
            raise ValueError("Arc not found in network")

        self._arcs = [held for held in self._arcs if held is not arc]

        # Update arc index
        indexed = self._arc_index.get(arc.source_state)
        if indexed is not None:
            remaining = [held for held in indexed if held is not arc]
            if remaining:
                self._arc_index[arc.source_state] = remaining
            else:
                del self._arc_index[arc.source_state]

        # And the source state's own list, which is what the engines read.
        source = self._states.get(arc.source_state)
        if source is not None:
            source.outgoing_arcs[:] = [held for held in source.outgoing_arcs if held is not arc]

        # Invalidate validation cache
        self._validation_cache = None

    def get_state(self, name: str) -> StateDefinition | None:
        """Get a state by name.

        Args:
            name: State name.

        Returns:
            State if found, None otherwise.
        """
        return self._states.get(name)

    def get_arcs_from_state(self, state_name: str) -> List[ArcDefinition]:
        """Get all arcs originating from a state.

        Args:
            state_name: Source state name.

        Returns:
            List of arcs from the state.
        """
        return self._arc_index.get(state_name, [])

    def get_arcs_to_state(self, state_name: str) -> List[ArcDefinition]:
        """Get all arcs targeting a state.

        Args:
            state_name: Target state name.

        Returns:
            List of arcs to the state.
        """
        return [arc for arc in self._arcs if arc.target_state == state_name]

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate network consistency.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        # Return cached result if available
        if self._validation_cache is not None:
            return self._validation_cache["is_valid"], self._validation_cache["errors"]

        errors = []

        # Check for initial state
        if not self._initial_state:
            errors.append("No initial state defined")
        elif self._initial_state not in self._states:
            errors.append(f"Initial state '{self._initial_state}' not found")

        # Check for final states
        if not self._final_states:
            errors.append("No final states defined")
        else:
            for final_state in self._final_states:
                if final_state not in self._states:
                    errors.append(f"Final state '{final_state}' not found")

        # Check that every arc lands somewhere. ``add_arc`` refuses an unknown
        # target, but a state assembled with ``add_outgoing_arc`` and then
        # added is ingested without that check --- its targets may legitimately
        # not exist yet at ingestion time. This is where that is answered, and
        # without it ingestion would be a way to put an unfollowable arc into a
        # network with nothing ever saying so.
        for arc in self._arcs:
            if arc.target_state not in self._states:
                errors.append(
                    f"Arc from '{arc.source_state}' targets unknown state '{arc.target_state}'"
                )

        # Check for unreachable states
        if self._initial_state:
            reachable = self._find_reachable_states(self._initial_state)
            unreachable = set(self._states.keys()) - reachable
            for state in unreachable:
                errors.append(f"State '{state}' is unreachable from initial state")

        # Check for states with no outgoing arcs (except final states)
        for state_name in self._states:
            if state_name not in self._final_states:
                if state_name not in self._arc_index or not self._arc_index[state_name]:
                    errors.append(f"Non-final state '{state_name}' has no outgoing arcs")

        # Check for cycles that don't include final states
        cycles = self.find_cycles()
        for cycle in cycles:
            if not any(state in self._final_states for state in cycle):
                errors.append(f"Cycle detected without final states: {' -> '.join(cycle)}")

        # Cache validation result
        is_valid = len(errors) == 0
        self._validation_cache = {"is_valid": is_valid, "errors": errors}

        return is_valid, errors

    def get_resource_requirements(self) -> NetworkResourceRequirements:
        """Get aggregated resource requirements for the network.

        Returns:
            Resource requirements.
        """
        return self._resource_requirements

    def set_streaming_enabled(self, enabled: bool) -> None:
        """Declare whether this network streams.

        Args:
            enabled: What :meth:`supports_streaming` should answer.
        """
        self._streaming_enabled = enabled

    def is_streaming_enabled(self) -> bool:
        """Check if any state in the network requires streaming.

        Returns:
            True if streaming is required.
        """
        return self._streaming_enabled

    def analyze_dependencies(self) -> Dict[str, Set[str]]:
        """Analyze resource dependencies between states.

        Returns:
            Dictionary mapping resources to dependent states.
        """
        # Keyed by the resource's *name*, which is what the signature and the
        # docstring both say. It used to key by the ``ResourceConfig`` object,
        # so the returned mapping did not have the shape it declared and no
        # caller could look a resource up by the name it configured it under.
        dependencies: Dict[str, Set[str]] = {}

        for state_name, state in self._states.items():
            for resource in state.resource_requirements:
                dependencies.setdefault(resource.name, set()).add(state_name)

        return dependencies

    def _update_resource_requirements(self, state: StateDefinition) -> None:
        """Fold one state's declared resources into the network's totals.

        ``resource_requirements`` is a ``List[ResourceConfig]``, and this asked
        the *list* for ``.databases`` / ``.filesystems`` / ``.http_services`` /
        ``.llms``. A list has none of them, so every branch was skipped and
        :meth:`get_resource_requirements` answered with an empty set for every
        network ever built --- including every network the config builder
        produces. Retyping this method's parameter from the retired ``State``
        to ``StateDefinition`` is what made the mismatch visible; the method
        directly above it, :meth:`analyze_dependencies`, already read the same
        field correctly as a list.

        Args:
            state: State to analyze.
        """
        totals = self._resource_requirements

        for resource in state.resource_requirements:
            # ``type`` is declared ``str``. The config builder puts the schema
            # ``ResourceConfig`` here, whose ``type`` is a ``ResourceType`` ---
            # the two-``ResourceConfig`` mismatch this package tracks
            # separately. Comparison works across both because ``ResourceType``
            # subclasses ``str``; only the custom bucket's *key* needs the
            # enum's value, since ``str()`` on a member gives its repr.
            kind = resource.type
            if kind in ("database", "async_database"):
                totals.databases.add(resource.name)
            elif kind == "filesystem":
                totals.filesystems.add(resource.name)
            elif kind == "http":
                totals.http_services.add(resource.name)
            elif kind == "llm":
                totals.llms.add(resource.name)
            else:
                key = kind.value if isinstance(kind, Enum) else str(kind)
                totals.custom.setdefault(key, set()).add(resource.name)

    def _recalculate_resource_requirements(self) -> None:
        """Recalculate all resource requirements from scratch.

        ``_streaming_enabled`` is deliberately not reset: it is *declared*
        (by config, through :meth:`set_streaming_enabled`) rather than derived
        from the states, so recalculating the resource totals must not clear
        it. It used to be read off each state's ``resource_requirements``,
        which never had it --- so the flag had no source at all and
        :meth:`supports_streaming` answered ``False`` for every network,
        including one whose config said ``streaming: {enabled: true}``.
        """
        self._resource_requirements = NetworkResourceRequirements()

        for state in self._states.values():
            self._update_resource_requirements(state)

    def _rebuild_arc_index(self) -> None:
        """Rebuild the arc index from scratch."""
        self._arc_index = {}
        for arc in self._arcs:
            if arc.source_state not in self._arc_index:
                self._arc_index[arc.source_state] = []
            self._arc_index[arc.source_state].append(arc)

    def _find_reachable_states(self, start_state: str) -> Set[str]:
        """Find all states reachable from a given state.

        Args:
            start_state: Starting state name.

        Returns:
            Set of reachable state names.
        """
        reachable = set()
        to_visit = [start_state]

        while to_visit:
            current = to_visit.pop()
            if current in reachable:
                continue

            reachable.add(current)

            # Add target states of outgoing arcs
            for arc in self.get_arcs_from_state(current):
                if arc.target_state not in reachable:
                    to_visit.append(arc.target_state)

        return reachable

    def find_cycles(self) -> List[List[str]]:
        """Find all cycles in the network.

        Part of the construction API --- the PR that repaired that API, this
        package's changelog and the test that covers it all name
        ``find_cycles``, while the method was defined as ``_find_cycles`` and
        so was not reachable under the name it was published as.

        Returns:
            List of cycles (each cycle is a list of state names).
        """
        cycles = []
        visited = set()
        rec_stack = []

        def dfs(state: str) -> None:
            visited.add(state)
            rec_stack.append(state)

            for arc in self.get_arcs_from_state(state):
                if arc.target_state not in visited:
                    dfs(arc.target_state)
                elif arc.target_state in rec_stack:
                    # Found a cycle
                    cycle_start = rec_stack.index(arc.target_state)
                    cycle = rec_stack[cycle_start:] + [arc.target_state]
                    cycles.append(cycle)

            rec_stack.pop()

        # Start DFS from all unvisited states
        for state_name in self._states:
            if state_name not in visited:
                dfs(state_name)

        return cycles

    def to_dict(self) -> Dict[str, Any]:
        """Convert network to dictionary representation.

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "description": self.description,
            "initial_state": self._initial_state,
            "final_states": list(self._final_states),
            "states": {name: state.to_dict() for name, state in self._states.items()},
            # Each arc writes its own payload, tagged with its class, so a
            # ``PushArc`` comes back a ``PushArc``. Written inline here, the
            # payload carried neither the push fields nor any record of the
            # class, so a round-tripped sub-network call came back as an
            # ordinary transition to the state named in ``target`` --- the
            # sub-network silently never entered. It also wrote ``transform``
            # whole, which stopped being JSON the moment ``add_arc`` began
            # storing the builder's own ``TransformSpec``.
            "arcs": [arc.to_dict() for arc in self._arcs],
            "resource_requirements": {
                "databases": list(self._resource_requirements.databases),
                "filesystems": list(self._resource_requirements.filesystems),
                "http_services": list(self._resource_requirements.http_services),
                "llms": list(self._resource_requirements.llms),
                "custom": {k: list(v) for k, v in self._resource_requirements.custom.items()},
                "streaming_enabled": self._resource_requirements.streaming_enabled,
                "estimated_memory_mb": self._resource_requirements.estimated_memory_mb,
            },
            "streaming_enabled": self._streaming_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateNetwork":
        """Create network from dictionary representation.

        Args:
            data: Dictionary representation.

        Returns:
            StateNetwork instance.
        """
        network = cls(name=data["name"], description=data.get("description"))

        # Add states. What comes back carries the declarative fields and not
        # the functions, schema or resources --- see StateDefinition.to_dict
        # for why. The old reading built a three-attribute ``State`` from the
        # name alone and discarded the rest of the payload, which produced a
        # network the engines could not execute at all.
        for state_name, state_data in data.get("states", {}).items():
            state = StateDefinition.from_dict(
                state_data if isinstance(state_data, dict) else {"name": state_name}
            )
            is_initial = state_name == data.get("initial_state")
            is_final = state_name in data.get("final_states", [])
            network.add_state(state, initial=is_initial, final=is_final)

        # Add arcs --- through ``add_arc``, so the rebuilt network has the
        # same single-writer guarantee a freshly built one does. A reader that
        # appended to ``_arcs`` and to the state separately would recreate the
        # very split this class was repaired to close, and every accessor
        # would still agree with itself.
        for arc_data in data.get("arcs", []):
            network.add_arc(
                source_state=arc_data["source"],
                target_state=arc_data["target"],
                definition=arc_from_dict(arc_data),
            )

        network.set_streaming_enabled(bool(data.get("streaming_enabled", False)))

        return network
