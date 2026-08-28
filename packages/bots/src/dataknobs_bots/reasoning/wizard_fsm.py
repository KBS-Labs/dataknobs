"""WizardFSM wrapper for wizard-specific FSM operations.

This module provides a thin wrapper around AdvancedFSM that adds
wizard-specific conveniences like stage metadata access, navigation
helpers, and serialization for persistence.
"""

import copy
import logging
from types import TracebackType
from typing import Any, Callable, Self, TypeVar

from dataknobs_fsm.api.advanced import AdvancedFSM, StepResult
from dataknobs_fsm.execution.context import ExecutionContext

from .wizard_skip import SKIP_DEFAULT_OVERWRITE, SkipDefaults

logger = logging.getLogger(__name__)


def _describe_condition(condition: Any) -> str:
    """How a declared transition condition reads in a step log.

    ``None`` is the absence of a ``condition:`` key, and is what
    *unconditional* means.  An empty string is a different thing:
    :meth:`WizardConfigLoader._translate_transition` builds a
    ``FunctionReference`` for any *present* ``condition`` key, so
    ``condition: ""`` compiles to an arc that can never fire rather than
    one that always does.  Reporting both as "unconditional" described a
    dead arc as an open one -- and the shape that did it, ``or``, is the
    same falsy-versus-absent conflation that a ``get`` default made in
    the other direction here.
    """
    if condition is None:
        return "unconditional"
    if isinstance(condition, str) and not condition.strip():
        return "empty (never fires)"
    return str(condition)


#: A stage field's declared type, taken from the default its accessor
#: passes to :meth:`WizardFSM._stage_field`.
_T = TypeVar("_T")


class WizardFSM:
    """Wrapper around AdvancedFSM with wizard-specific conveniences.

    Provides a simplified interface for wizard operations including:
    - Stage metadata access (prompts, schemas, suggestions)
    - Navigation helpers (back, skip, restart)
    - State serialization for persistence
    - Stage-specific tool and configuration access
    - Subflow registry for nested wizard flows

    Lifecycle: the wrapped ``AdvancedFSM`` allocates a daemon event-loop
    thread the first time it is stepped synchronously, so a wizard driven
    through :meth:`step` holds an OS resource. Release it with
    :meth:`close` / :meth:`aclose`, or let ``with`` / ``async with`` do it
    by construction. Closing cascades to every subflow this FSM owns.

    Attributes:
        _fsm: Underlying AdvancedFSM instance
        _stage_metadata: Dict mapping stage names to metadata
        _settings: Wizard-level settings (auto_advance_filled_stages, etc.)
        _context: Current execution context
        _subflow_registry: Dict mapping subflow names to WizardFSM instances
        _owns_subflows: Names in ``_subflow_registry`` whose lifecycle this
            FSM owns, and which its close therefore cascades to
    """

    def __init__(
        self,
        fsm: AdvancedFSM,
        stage_metadata: dict[str, dict[str, Any]],
        settings: dict[str, Any] | None = None,
        subflow_registry: dict[str, "WizardFSM"] | None = None,
        transform_context_factory: Callable[..., Any] | None = None,
    ):
        """Initialize WizardFSM.

        Args:
            fsm: AdvancedFSM instance to wrap
            stage_metadata: Dict mapping stage names to their metadata.
                Normally built by
                :meth:`~dataknobs_bots.reasoning.wizard_loader.WizardConfigLoader._extract_metadata`.
                A hand-built mapping should give each entry of a stage's
                ``transitions`` list a ``name`` and an ``arc_target``
                matching what
                :func:`~dataknobs_bots.reasoning.wizard_loader.arc_identity`
                derives for the compiled arc: ``name`` is how a move is
                matched back to the transition that caused it, so without
                it :meth:`get_transition_condition` reports ``None`` for
                any target two transitions lead to, and ``arc_target``
                is what makes a subflow transition's self-loop match.
                Both are optional -- a stage whose targets are each
                reached by one transition answers correctly without them.
            settings: Wizard-level settings dict (optional)
            subflow_registry: Dict mapping subflow names to WizardFSM
                instances. Subflows passed here are **owned** by this FSM —
                they are built for it by the loader, so :meth:`close`
                cascades to them. Use :meth:`register_subflow` with
                ``owns=False`` to add one whose lifecycle stays with its
                caller. The mapping is **copied**: ownership is recorded
                once, from the contents at construction, so a caller that
                kept its reference and added an entry afterwards would
                otherwise get a subflow this FSM steps but never closes.
            transform_context_factory: Optional callable that receives a
                :class:`FunctionContext` and returns the application-specific
                context for transforms (e.g. :class:`TransformContext`).
                Can also be set later via
                :meth:`set_transform_context_factory`.
        """
        self._fsm = fsm
        self._stage_metadata = stage_metadata
        # (stage, field) pairs already reported by _stage_field, so an
        # ill-typed config is named once rather than once per turn.
        self._reported_stage_field_types: set[tuple[str, str]] = set()
        self._settings = settings or {}
        self._context: ExecutionContext | None = None
        self._subflow_registry: dict[str, WizardFSM] = dict(subflow_registry or {})
        self._owns_subflows: set[str] = set(self._subflow_registry)
        self._transform_context_factory: Callable[..., Any] | None = transform_context_factory

    def set_transform_context_factory(self, factory: Callable[..., Any]) -> None:
        """Register a factory for building transform-level context objects.

        The factory receives a :class:`FunctionContext` and returns the
        application-specific context that transforms should receive (e.g.
        :class:`TransformContext`).  It is applied to the
        :class:`ExecutionContext` before each step executes.

        If the :class:`ExecutionContext` has already been created (i.e.
        after the first step), the factory is propagated to it immediately
        so that the next ``step`` / ``step_async`` call uses the new
        factory without requiring context re-creation.

        Args:
            factory: Callable accepting a FunctionContext and returning
                the desired transform context.
        """
        self._transform_context_factory = factory
        if self._context is not None:
            self._context.transform_context_factory = factory

    def get_transform_context_factory(self) -> Callable[..., Any] | None:
        """Return the currently registered transform context factory.

        Returns:
            The factory callable, or ``None`` if none is registered.
        """
        return self._transform_context_factory

    @property
    def settings(self) -> dict[str, Any]:
        """Get wizard-level settings.

        Returns:
            Dict containing wizard settings like auto_advance_filled_stages
        """
        return self._settings

    @property
    def current_stage(self) -> str:
        """Get current stage name.

        Returns:
            Name of the current stage
        """
        if self._context and self._context.current_state:
            return self._context.current_state
        return self._find_start_stage()

    @property
    def current_metadata(self) -> dict[str, Any]:
        """Get metadata for current stage.

        The stage dict is the live one, not a copy -- see :attr:`stages`
        for the boundary these readers share.

        Returns:
            Dict containing current stage's metadata
        """
        return self._stage_metadata.get(self.current_stage, {})

    def stage_metadata_for(self, stage: str) -> dict[str, Any]:
        """Get metadata for a named stage, or ``{}`` if this FSM has none.

        The empty dict is the same answer :attr:`current_metadata` gives,
        and it is deliberately not distinguished from a stage that
        declared nothing -- callers reading one field with a default get
        the documented default either way. A caller that needs to tell
        "absent" from "empty" apart wants :meth:`has_stage`.

        The stage dict is the live one, not a copy -- see :attr:`stages`
        for the boundary these readers share.

        Args:
            stage: Stage name.

        Returns:
            The stage's metadata dict, or ``{}``.
        """
        return self._stage_metadata.get(stage, {})

    def has_stage(self, stage: str) -> bool:
        """Whether this FSM defines *stage*.

        Args:
            stage: Stage name.

        Returns:
            ``True`` if the stage belongs to this FSM.
        """
        return stage in self._stage_metadata

    def find_stage_owner(self, stage: str) -> "WizardFSM | None":
        """The FSM in this tree that defines *stage*, searched depth-first.

        Answers "does this wizard have such a stage at all", which is a
        property of the configuration rather than of where the user
        currently is.  :meth:`has_stage` answers it for one frame;
        callers that must not depend on the frame -- the section-to-stage
        table, which maps a name a user said onto a name the wizard uses
        -- want this instead.

        A caller that then has to *act* in the owning frame needs more
        than this: two flows may legitimately name a stage the same, and
        this reports the first one found (self, then subflows in
        registration order).  Resolve the frame from the wizard state
        there, and use this only to decide whether the name is a stage.

        Args:
            stage: Stage name.

        Returns:
            The FSM defining *stage*, or ``None`` if no flow in this tree
            does.
        """
        return self._find_stage_owner(stage, set())

    def _find_stage_owner(self, stage: str, seen: set[int]) -> "WizardFSM | None":
        """Recursive half of :meth:`find_stage_owner`.

        ``seen`` holds ``id()`` of the FSMs already visited: a registry
        is built from config and nothing forbids a cycle, and a cycle
        here would be an unbounded recursion rather than a wrong answer.
        """
        if id(self) in seen:
            return None
        seen.add(id(self))
        if self.has_stage(stage):
            return self
        for name in self.subflow_names:
            subflow = self.get_subflow(name)
            if subflow is None:
                continue
            owner = subflow._find_stage_owner(stage, seen)
            if owner is not None:
                return owner
        return None

    @property
    def stages(self) -> dict[str, dict[str, Any]]:
        """Get all stage metadata.

        The returned mapping is a **shallow** copy: adding or removing a
        key does not change which stages this FSM has, but the stage
        dicts inside it are the live ones, so writing through them does.
        The guarantee is over the table's shape, not its contents -- a
        caller that intends to edit a stage it read here must copy that
        stage itself.

        A deep copy is deliberately not taken. The four call sites in
        this package only iterate -- one of them, the stages roadmap, on
        every turn -- and deepcopy measures ~2500x the shallow copy for a
        guarantee none of them asks for.

        Returns:
            Dict mapping stage name to stage configuration dict.
        """
        return dict(self._stage_metadata)

    @property
    def stage_names(self) -> list[str]:
        """Get ordered list of stage names.

        Returns:
            List of stage names in definition order.
        """
        return list(self._stage_metadata.keys())

    @property
    def stage_count(self) -> int:
        """Get total number of stages.

        Returns:
            Number of stages in the wizard.
        """
        return len(self._stage_metadata)

    def _stage_field(self, stage: str | None, key: str, default: _T) -> _T:
        """Read one field of a stage's metadata as the type it is declared.

        Stage metadata is authored config carried through untouched:
        ``_StageField.extract`` copies the value out of the YAML without
        coercing it, and nothing between the file and here checks it. The
        accessors below nonetheless declare concrete return types, and
        every caller relies on them -- a stage written ``can_skip: "no"``
        yields a *truthy string* from a method declared ``-> bool``, so the
        stage the author marked unskippable becomes skippable.

        A value of the wrong type is therefore replaced by the field's
        documented default and reported once, rather than handed to a
        caller that will fail somewhere the config is no longer in view.
        Warning once per stage and field keeps a broken config from
        filling the log a turn at a time.

        The check is on the container, not its contents: ``tools: [1, 2]``
        is a list and passes. What it is written to catch is a value of
        the wrong *shape* -- a string where a list belongs, iterated one
        character at a time -- which is the failure a YAML author actually
        produces. An element check would need each accessor to declare its
        element type, which is a different contract from the one the
        ``default`` argument carries here.

        Args:
            stage: Stage name, or ``None`` for the current stage.
            key: Metadata field to read.
            default: The field's documented default; its type is the
                contract the returned value is held to.

        Returns:
            The authored value when it matches ``default``'s type,
            otherwise ``default``.
        """
        stage_name = stage or self.current_stage
        value = self._stage_metadata.get(stage_name, {}).get(key, default)
        if isinstance(value, type(default)):
            return value

        if value is None:
            # Not a wrong-typed value: ``_StageField.extract`` writes
            # ``None`` for every field the stage did not declare, so an
            # unset field is *present* in the metadata holding it. Saying
            # a stage "declares skip_default_mode as NoneType" would
            # accuse the author of writing something they never wrote,
            # once for every stage in every config that leaves the field
            # out -- which is nearly all of them. An authored ``null``
            # reaches here identically and means the same thing: unset.
            return default

        self._report_once(
            stage_name,
            key,
            "Stage '%s' declares %s as %s; %s is required, using the default %r",
            stage_name,
            key,
            type(value).__name__,
            type(default).__name__,
            default,
        )
        return default

    def _report_once(self, stage_name: str, key: str, message: str, *args: Any) -> None:
        """Warn about *key* on *stage_name* the first time only.

        A broken config is broken on every turn, so a per-turn warning
        fills the log with one problem and buries the next one. Shared by
        every reader of authored stage metadata rather than duplicated
        per reader, so the "once" is once across all of them: a stage
        whose ``skip_default`` is unreadable says so as many times as one
        whose ``can_skip`` is, which is once.

        Args:
            stage_name: The stage the finding is about.
            key: The metadata field, dotted for anything nested inside
                one -- the pair with *stage_name* is what "once" counts.
            message: Lazy-formatted log message.
            *args: Its arguments.
        """
        if (stage_name, key) in self._reported_stage_field_types:
            return
        self._reported_stage_field_types.add((stage_name, key))
        logger.warning(message, *args)

    def get_stage_prompt(self, stage: str | None = None) -> str:
        """Get prompt for a stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            Stage prompt string
        """
        return self._stage_field(stage, "prompt", "")

    def get_stage_schema(self, stage: str | None = None) -> dict[str, Any] | None:
        """Get validation schema for a stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            JSON Schema dict or None
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("schema")

    def get_stage_tools(self, stage: str | None = None) -> list[str]:
        """Get available tool names for a stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            List of tool names available in the stage
        """
        return self._stage_field(stage, "tools", [])

    def get_stage_suggestions(self, stage: str | None = None) -> list[str]:
        """Get quick-reply suggestions for a stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            List of suggestion strings
        """
        return self._stage_field(stage, "suggestions", [])

    def _matched_transition(
        self, from_stage: str, to_stage: str, arc_name: str | None
    ) -> tuple[dict[str, Any] | None, int]:
        """Which declared transition describes the arc that fired.

        Two callers ask this -- the DEBUG step log and
        :meth:`get_transition_condition` -- and both used to scan the
        stage's transitions for the first one declaring ``to_stage``.
        That answer is right only when a single arc leads there; when two
        do, it names the first regardless of which fired, and it does so
        with no indication that it guessed.

        ``arc_name`` is the discriminator, carried on the arc by
        :func:`~dataknobs_bots.reasoning.wizard_loader.arc_identity` and
        reported back as ``StepResult.transition``.  Matching on it is
        exact.  When it identifies nothing -- a caller that has no step
        result, or metadata built before the arc was named -- the target
        scan is still correct for a target only one arc leads to, so that
        case keeps its answer; an ambiguous one returns no entry rather
        than a guess.

        A name identifies an arc only while it is unique.  The derived
        form carries the transition's index and so cannot collide, but an
        authored ``metadata: {name: ...}`` can be repeated, and two arcs
        answering to one string are exactly as unidentifiable as the
        anonymous arcs this naming replaced.  So a name matching more
        than one candidate is treated as the ambiguous case, not as a
        match -- and the scan is confined to the transitions that lead to
        ``to_stage``, so a name reused on a route to somewhere else
        cannot answer here at all.

        Args:
            from_stage: Stage the step started from.
            to_stage: Stage it ended on.
            arc_name: ``StepResult.transition``, when the caller has one.

        Returns:
            ``(entry, candidates)`` -- the matched transition metadata or
            ``None``, and how many declared transitions lead to
            ``to_stage``.  The count is what a caller reports when the
            entry is ``None``.
        """
        stage_meta = self._stage_metadata.get(from_stage, {})
        transitions = stage_meta.get("transitions", [])
        # ``arc_target`` is what the compiled arc points at; ``target`` is
        # what the author declared, and the two differ for a subflow
        # transition, whose ``_subflow`` sentinel compiles to a self-loop.
        # Metadata built by hand rather than by the loader carries only
        # the latter, hence the fallback.
        candidates = [t for t in transitions if t.get("arc_target", t.get("target")) == to_stage]

        if arc_name:
            # Matched among the candidates, not across every transition
            # the stage declares: a name is unique only by construction,
            # and an authored one repeated on a transition to some *other*
            # target would otherwise answer for a move it did not cause.
            named = [t for t in candidates if t.get("name") == arc_name]
            if len(named) == 1:
                return named[0], len(candidates)
            if named:
                # One name, two arcs -- an authored ``metadata: {name:
                # ...}`` duplicated within the stage, which
                # ``WizardConfigLoader._validate_config`` reports at load.
                # The discriminator does not discriminate, so this is the
                # ambiguous case however exact the match looked.
                return None, len(candidates)
            # A name that matches no candidate identifies nothing here --
            # metadata predating arc names, or a caller's own string. The
            # target scan below is still right for an unambiguous target.

        if len(candidates) == 1:
            return candidates[0], 1

        return None, len(candidates)

    def get_transition_condition(
        self, from_stage: str, to_stage: str, *, arc_name: str | None = None
    ) -> str | None:
        """Get the condition expression for a transition.

        Pass ``arc_name`` -- ``StepResult.transition`` from the step that
        made this move -- whenever it is in hand.  Without it, a stage
        with two transitions to ``to_stage`` cannot say which one fired,
        and this returns ``None``: the value is recorded as a transition
        record's ``condition_evaluated``, where a wrong expression is
        worse than an absent one, and the record is persisted rather than
        merely logged.  A target only one transition leads to is
        unambiguous and answers the same either way.

        Args:
            from_stage: Source stage name
            to_stage: Target stage name
            arc_name: Name of the arc that fired, when known

        Returns:
            Condition expression string, or None if no condition is
            declared or the arc cannot be identified
        """
        transition, _candidates = self._matched_transition(from_stage, to_stage, arc_name)
        if transition is None:
            return None

        condition = transition.get("condition")
        # Feeds a transition record's ``condition_evaluated`` and
        # nothing else. A non-string would be read back as the
        # expression that fired; "nothing recorded" is honest.
        return condition if isinstance(condition, str) else None

    def can_skip(self, stage: str | None = None) -> bool:
        """Check if stage can be skipped.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            True if stage can be skipped
        """
        return self._stage_field(stage, "can_skip", False)

    def get_skip_defaults(self, stage: str | None = None) -> SkipDefaults:
        """The values a stage writes when it is skipped, with their modes.

        Two authored fields make one answer -- ``skip_default`` holds the
        values and ``skip_default_mode`` says what may be overwritten --
        so they are read together here rather than separately by each
        caller. Both go through :meth:`_stage_field`, which means a
        ``skip_default`` written as a bare string (which this field has
        never honoured, and which the builder's own signature invited)
        is reported rather than dropped in silence.

        Args:
            stage: Stage name (defaults to current stage).

        Returns:
            The resolved block; empty when the stage declares none.
        """
        stage_name = stage or self.current_stage

        def _report(key: str, value: Any, requirement: str, outcome: str) -> None:
            # "" is the block-level mode, which is a *sibling* field --
            # naming it "skip_default" would point the author at the
            # block they wrote correctly, and would share a warn-once
            # slot with the block's own type check.
            field = f"skip_default.{key}" if key else "skip_default_mode"
            # The outcome is carried rather than assumed: a bad mode
            # falls back to another mode, which the author needs named,
            # while a mapping that only looks like an entry is still
            # written as the value it reads as -- no default involved.
            self._report_once(
                stage_name,
                field,
                "Stage '%s' declares %s as %r; %s. %s.",
                stage_name,
                field,
                value,
                requirement,
                outcome,
            )

        return SkipDefaults.from_stage(
            self._stage_field(stage, "skip_default", {}),
            self._stage_field(stage, "skip_default_mode", SKIP_DEFAULT_OVERWRITE),
            on_invalid=_report,
        )

    def can_go_back(self, stage: str | None = None) -> bool:
        """Check if back navigation is allowed.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            True if back navigation is allowed
        """
        return self._stage_field(stage, "can_go_back", True)

    def is_start_stage(self, stage: str | None = None) -> bool:
        """Check if stage is a start stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            True if stage is marked as start
        """
        return self._stage_field(stage, "is_start", False)

    def is_end_stage(self, stage: str | None = None) -> bool:
        """Check if stage is an end stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            True if stage is marked as end
        """
        return self._stage_field(stage, "is_end", False)

    # =========================================================================
    # Subflow Registry Methods
    # =========================================================================

    def get_subflow(self, name: str) -> "WizardFSM | None":
        """Get a subflow WizardFSM by name.

        Args:
            name: Subflow network name

        Returns:
            WizardFSM instance for the subflow, or None if not found
        """
        return self._subflow_registry.get(name)

    def has_subflow(self, name: str) -> bool:
        """Check if a subflow exists in the registry.

        Args:
            name: Subflow network name

        Returns:
            True if the subflow exists
        """
        return name in self._subflow_registry

    def register_subflow(self, name: str, subflow_fsm: "WizardFSM", *, owns: bool = True) -> None:
        """Register a subflow WizardFSM.

        Args:
            name: Subflow network name
            subflow_fsm: WizardFSM instance for the subflow
            owns: Whether this FSM owns *subflow_fsm*'s lifecycle. When
                True (the default, matching the loader-built subflows this
                registry normally holds), :meth:`close` and :meth:`aclose`
                cascade to it. Pass False to register a subflow the caller
                built and still holds — closing it out from under its owner
                would tear down an FSM that may still be stepped.

                Re-registering a name replaces both the subflow and its
                ownership, so a name is owned iff its *most recent*
                registration said so.

        Replacing an **owned** subflow with a *different* object closes the
        one displaced: reusing its name is this FSM's last chance to release
        it, and dropping it from the registry unclosed would leak whatever
        it holds — the same unreachable-daemon-thread defect this class's
        lifecycle exists to prevent, one level down. Re-registering the
        *same* object closes nothing, so the documented
        ``register_subflow(name, child, owns=False)`` hand-back does not
        destroy what it hands over; nor is an unowned subflow ever closed,
        since displacing it from this registry is not permission to tear
        down something its caller may still be stepping.
        """
        displaced = self._subflow_registry.get(name)
        if displaced is not None and displaced is not subflow_fsm and name in self._owns_subflows:
            try:
                displaced.close()
            except Exception:
                logger.exception("Error closing subflow %r displaced by re-registration", name)

        self._subflow_registry[name] = subflow_fsm
        if owns:
            self._owns_subflows.add(name)
        else:
            self._owns_subflows.discard(name)

    @property
    def subflow_names(self) -> list[str]:
        """Get list of registered subflow names.

        Returns:
            List of subflow network names
        """
        return list(self._subflow_registry.keys())

    def resolve_function(self, name: str) -> Callable[..., Any] | None:
        """Look up a registered function by name.

        Searches the FSM's function registry (functions, validators,
        and transforms) and custom functions for a callable matching
        *name*.  Used by :class:`WizardReasoning` to resolve routing
        transform names declared in stage config.

        Args:
            name: Function name as registered in the FSM.

        Returns:
            The callable, or ``None`` if not found.
        """
        registry = getattr(self._fsm.fsm, "function_registry", None)
        if registry is None or not hasattr(registry, "get_function"):
            return None
        func = registry.get_function(name)
        # The registry is typed Any at its source and holds whatever was
        # registered.  The caller calls what comes back, so a non-callable
        # would fail there, one frame away from the name that produced it.
        if callable(func):
            # mypy does not narrow Any through callable(); the annotation
            # states what the check above established at runtime.
            resolved: Callable[..., Any] = func
            return resolved
        if func is not None:
            logger.warning(
                "Function registry entry '%s' is not callable (%s); treating it as absent",
                name,
                type(func).__name__,
            )
        return None

    def _log_step_outcome(self, before_stage: str, after_stage: str, transition: str) -> None:
        """Say what the step did, at DEBUG, for both ``step`` variants.

        Standing still is not the same as matching nothing, and the case
        where they differ is the one worth naming: a **subflow**
        transition compiles to a self-loop arc carrying its condition
        (``WizardConfigLoader._translate_transition``), so a matched
        subflow transition leaves the FSM exactly where it started.
        Reporting that as "none matched" sends a reader looking for a
        broken condition when the condition was satisfied and the push is
        somebody else's job --
        ``SubflowManager.should_push``, which runs before this step.

        The two are told apart by *what the step reports*, not by what the
        stage declares. A stage holding a subflow transition whose guard
        **declined** matched nothing, and is the ordinary case rather than
        the exotic one: a guard that carries pushes the subflow, and a push
        skips this step entirely, so a step that runs at all is one where
        no push was performed. Reading the declaration instead would
        describe every declined turn as a self-loop.
        ``StepResult.transition`` is ``"none"`` exactly when the engine
        found no arc whose condition passed (``api/advanced.py``), which
        also keeps a regular arc back to its own stage off the
        "none matched" line.

        The same report also has to name *which* arc moved the wizard,
        and that is the other thing the step reports.  Scanning for the
        first transition declaring ``after_stage`` answers correctly only
        while one arc leads there; a stage offering two routes to the
        same target is the case the answer is wanted for, and the scan
        names the first of them whichever fired.  So the arc name is
        matched instead, and when it identifies nothing the line says how
        many arcs it could have been rather than picking one.

        Args:
            before_stage: Stage the step started from.
            after_stage: Stage the step ended on.
            transition: ``StepResult.transition`` from the step just taken.
        """
        stage_meta = self._stage_metadata.get(before_stage, {})
        transitions = stage_meta.get("transitions", [])

        if before_stage != after_stage:
            matched, candidates = self._matched_transition(before_stage, after_stage, transition)
            if matched is not None:
                logger.debug(
                    "WizardFSM transition: '%s' -> '%s' via condition: %s",
                    before_stage,
                    after_stage,
                    _describe_condition(matched.get("condition")),
                )
            elif candidates:
                logger.debug(
                    "WizardFSM transition: '%s' -> '%s' via one of %d arcs; "
                    "step reports transition '%s', which matches none of them "
                    "by name",
                    before_stage,
                    after_stage,
                    candidates,
                    transition,
                )
            return

        if not transitions:
            return

        if transition == "none":
            logger.debug(
                "WizardFSM no transition from '%s': %d transitions defined, none matched",
                before_stage,
                len(transitions),
            )
        else:
            subflow_transitions = [t for t in transitions if t.get("is_subflow_transition")]
            logger.debug(
                "WizardFSM stayed at '%s': step reports transition '%s'; %d "
                "transitions defined, %d of them subflow transitions, whose "
                "arc is a self-loop and whose push is decided before this step",
                before_stage,
                transition,
                len(transitions),
                len(subflow_transitions),
            )

        for trans in transitions:
            logger.debug(
                "  - target='%s', condition='%s'",
                trans.get("target") or "?",
                _describe_condition(trans.get("condition")),
            )

    def step(self, data: dict[str, Any]) -> StepResult:
        """Execute one FSM step with given data.

        Creates or updates the execution context and executes
        a single FSM transition.

        Args:
            data: Data dict for transition evaluation

        Returns:
            StepResult with transition details
        """
        before_stage = self.current_stage
        if not self._context:
            self._context = self._fsm.create_context(data)
            if self._transform_context_factory:
                self._context.transform_context_factory = self._transform_context_factory
        else:
            # Update context data
            if isinstance(self._context.data, dict):
                self._context.data.update(data)
            else:
                self._context.data = data

        result = self._fsm.execute_step_sync(self._context)

        # Sync transform mutations back to caller's data dict.
        # Transforms modify context.data in place; without this sync,
        # the caller won't see new keys set by transforms (e.g.
        # _questions, _artifact_id).
        if isinstance(self._context.data, dict):
            data.update(self._context.data)

        after_stage = self.current_stage

        self._log_step_outcome(before_stage, after_stage, result.transition)

        return result

    async def step_async(self, data: dict[str, Any]) -> StepResult:
        """Execute one FSM step asynchronously with given data.

        Mirrors step() but uses execute_step_async so that async
        pre-tests, transforms, and hooks are properly awaited.

        Args:
            data: Data dict for transition evaluation

        Returns:
            StepResult with transition details
        """
        before_stage = self.current_stage
        if not self._context:
            self._context = self._fsm.create_context(data)
            if self._transform_context_factory:
                self._context.transform_context_factory = self._transform_context_factory
        else:
            # Update context data
            if isinstance(self._context.data, dict):
                self._context.data.update(data)
            else:
                self._context.data = data

        result = await self._fsm.execute_step_async(self._context)

        # Sync transform mutations back to caller's data dict.
        # Transforms modify context.data in place; without this sync,
        # the caller won't see new keys set by transforms (e.g.
        # _questions, _artifact_id).
        if isinstance(self._context.data, dict):
            data.update(self._context.data)

        after_stage = self.current_stage

        self._log_step_outcome(before_stage, after_stage, result.transition)

        return result

    def go_back(self, history: list[str]) -> bool:
        """Navigate to previous stage.

        Args:
            history: List of visited stage names

        Returns:
            True if back navigation succeeded
        """
        if len(history) <= 1:
            return False

        if not self.can_go_back():
            return False

        # Get previous stage from history
        previous_stage = history[-2] if len(history) >= 2 else None
        if not previous_stage:
            return False

        # Restore context to previous stage
        if self._context:
            self._context.set_state(previous_stage)
            return True

        return False

    def restart(self) -> None:
        """Reset wizard to start stage.

        Clears the execution context to start fresh.
        """
        self._context = None

    def serialize(self) -> dict[str, Any]:
        """Serialize wizard state for persistence.

        Returns:
            Dict containing serializable wizard state
        """
        return {
            "current_stage": self.current_stage,
            "history": self._get_history(),
            "data": self._get_data(),
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Restore wizard from serialized state.

        Deep-copies `data` to break any shared reference with the
        caller (e.g. ``manager.metadata``).  Without this, transforms
        that mutate ``context.data`` during ``step_async`` would
        contaminate the metadata dict, causing ``json.dumps(metadata)``
        to crash on non-serializable objects.

        Args:
            state: Previously serialized state dict
        """
        current_stage = state.get("current_stage")
        data = copy.deepcopy(state.get("data", {}))

        if current_stage:
            # Create new context with restored state
            self._context = self._fsm.create_context(data)
            self._context.set_state(current_stage)
            if self._transform_context_factory:
                self._context.transform_context_factory = self._transform_context_factory

    @property
    def start_stage(self) -> str:
        """Name of the wizard's start stage (``is_start`` or first-defined).

        Returns:
            Name of the start stage
        """
        return self._find_start_stage()

    def _find_start_stage(self) -> str:
        """Find the start stage.

        Returns:
            Name of the start stage
        """
        for name, meta in self._stage_metadata.items():
            if meta.get("is_start"):
                return name
        # Fallback to first stage
        return next(iter(self._stage_metadata.keys())) if self._stage_metadata else "start"

    def _get_history(self) -> list[str]:
        """Get stage history from context.

        The execution context records visited states on its
        ``state_history`` attribute, appending as it advances. This read
        used to look for the same name in the context's ``metadata`` dict
        instead — a channel nothing writes it to — so it missed on every
        call and the fallback below reported a single-stage history for
        every wizard, however far it had run.

        Returns:
            List of visited stage names, ending at the current stage.
        """
        if self._context:
            history = list(getattr(self._context, "state_history", None) or [])
            if history:
                if history[-1] != self.current_stage:
                    history.append(self.current_stage)
                return history
        return [self.current_stage]

    def _get_data(self) -> dict[str, Any]:
        """Get current data from context.

        Returns:
            Current data dict
        """
        if self._context:
            if isinstance(self._context.data, dict):
                return self._context.data.copy()
        return {}

    # ----------------------------------------------------------------
    # Lifecycle — mirrors AdvancedFSM's six members one for one, because
    # a wrapper that hides what it wraps is how the daemon thread this
    # releases became un-releasable in the first place.
    # ----------------------------------------------------------------

    def close(self) -> None:
        """Release the wrapped FSM's lifecycle resources. Idempotent.

        Stops and joins the daemon event-loop thread that repeated
        synchronous :meth:`step` creates on the underlying
        :class:`AdvancedFSM`, and closes its resource manager. Cascades to
        every registered subflow this FSM **owns** (see
        :meth:`register_subflow`); a subflow registered with ``owns=False``
        belongs to its caller and is left alone.

        Prefer :meth:`aclose` from async code: it does everything this does
        and additionally awaits providers whose cleanup is a coroutine,
        while keeping the bridge join off the event loop.

        The FSM remains **steppable** after close: a subsequent synchronous
        :meth:`step` lazily creates a new bridge rather than raising. That
        is what makes an unconditional teardown — a test fixture, a ``with``
        block — safe to apply without tracking whether this FSM was ever
        stepped.

        That guarantee covers the bridge, not registered resources. Closing
        is terminal for the wrapped FSM's resource manager: its providers
        are closed and not re-registered, so an ``AdvancedFSM`` holding
        resources should not be stepped after close. Loader-built wizard
        FSMs register none, which is what makes the unconditional teardown
        safe in practice.
        """
        self._close_subflows()
        self._fsm.close()

    async def aclose(self) -> None:
        """Async counterpart of :meth:`close`.

        Awaits the wrapped FSM's async resource cleanup — so a provider
        exposing an ``aclose`` / ``cleanup`` coroutine is awaited rather
        than skipped — then stops and joins the bridge thread, off the event
        loop. Cascades to owned subflows via *their* :meth:`aclose`.

        Same idempotence, and the same bridge-rebuild-on-next-step /
        resources-are-terminal split described on :meth:`close`.
        """
        await self._aclose_subflows()
        await self._fsm.aclose()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _close_subflows(self) -> None:
        """Close owned subflows, isolating each child's failure.

        Teardown is a cascade: one subflow raising must not orphan the
        ones registered after it.
        """
        for name, subflow in list(self._subflow_registry.items()):
            if name not in self._owns_subflows:
                continue
            try:
                subflow.close()
            except Exception:
                logger.exception("Error closing subflow %r", name)

    async def _aclose_subflows(self) -> None:
        """Async counterpart of :meth:`_close_subflows`."""
        for name, subflow in list(self._subflow_registry.items()):
            if name not in self._owns_subflows:
                continue
            try:
                await subflow.aclose()
            except Exception:
                logger.exception("Error closing subflow %r", name)


def create_wizard_fsm(
    fsm_config: dict[str, Any],
    stage_metadata: dict[str, dict[str, Any]],
    custom_functions: dict[str, Callable[..., Any]] | None = None,
    settings: dict[str, Any] | None = None,
    subflow_registry: dict[str, WizardFSM] | None = None,
) -> WizardFSM:
    """Factory function to create a WizardFSM instance.

    Args:
        fsm_config: FSM configuration dict
        stage_metadata: Stage metadata dict. See :class:`WizardFSM` for
            the ``name`` / ``arc_target`` keys a hand-built transitions
            list should carry so that sibling arcs stay distinguishable.
        custom_functions: Optional custom functions to register
        settings: Wizard-level settings dict (optional)
        subflow_registry: Dict mapping subflow names to WizardFSM instances

    Returns:
        Configured WizardFSM instance
    """
    from dataknobs_fsm.api.advanced import create_advanced_fsm

    advanced_fsm = create_advanced_fsm(fsm_config, custom_functions=custom_functions)
    return WizardFSM(
        advanced_fsm, stage_metadata, settings=settings, subflow_registry=subflow_registry
    )
