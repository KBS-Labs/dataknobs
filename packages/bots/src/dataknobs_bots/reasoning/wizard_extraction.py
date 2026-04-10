"""Extraction pipeline for wizard reasoning.

Handles schema-driven data extraction, normalization, merge, defaults,
derivations, validation, recovery strategies, and related utilities.

Extracted from :mod:`wizard` in item 77c.  :class:`WizardReasoning`
constructs a :class:`WizardExtractor` in ``__init__`` and delegates to
it (via ``self._extraction``) for all pipeline operations.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from dataknobs_llm.extraction.schema_extractor import SimpleExtractionResult

from .wizard_derivations import DerivationRule, apply_field_derivations
from .wizard_grounding import (
    MergeFilter,
    SchemaGroundingFilter,
    detect_boolean_signal,
    field_keywords,
)
from .wizard_utils import word_in_text
from .wizard_types import (
    RECOVERY_BOOLEAN,
    RECOVERY_CLARIFICATION,
    RECOVERY_DERIVATION,
    RECOVERY_FOCUSED_RETRY,
    RECOVERY_SCOPE_ESCALATION,
    SCOPE_BREADTH,
    ExtractionPipelineResult,
    StageSchema,
    WizardState,
    _DEFAULT_AFFIRMATIVE_PHRASES,
    _DEFAULT_AFFIRMATIVE_SIGNALS,
    _DEFAULT_NEGATIVE_PHRASES,
    _DEFAULT_NEGATIVE_SIGNALS,
    _normalize_enum_value,
)

logger = logging.getLogger(__name__)


class WizardExtractor:
    """Extraction pipeline: extract -> normalize -> merge -> defaults -> derivations -> recovery."""

    # -- Boolean truthy/falsy strings for normalization --
    _BOOL_TRUE = frozenset({"yes", "true", "1", "y", "on", "enable", "enabled"})
    _BOOL_FALSE = frozenset({"no", "false", "0", "n", "off", "disable", "disabled"})
    _ALL_KEYWORDS = frozenset({"all", "everything", "all of them", "every one"})
    _NONE_KEYWORDS = frozenset({"none", "nothing", "no tools", "empty"})

    def __init__(
        self,
        # --- Core extraction ---
        extractor: Any | None,
        # --- Merge ---
        merge_filter: MergeFilter | None,
        grounding_overlap_threshold: float,
        # --- Normalization ---
        enum_normalize: bool,
        normalize_threshold: float,
        reject_unmatched: bool,
        # --- Scope ---
        extraction_scope: str,
        recent_messages_count: int,
        # --- Conflict ---
        conflict_strategy: str,
        log_conflicts: bool,
        per_turn_keys: frozenset[str],
        # --- Recovery ---
        recovery_pipeline: list[str],
        boolean_recovery: bool,
        scope_escalation_enabled: bool,
        scope_escalation_scope: str,
        focused_retry_enabled: bool,
        focused_retry_max_retries: int,
        # --- Derivations ---
        field_derivations: list[DerivationRule],
    ) -> None:
        self._extractor = extractor
        self._merge_filter = merge_filter
        self._grounding_overlap_threshold = grounding_overlap_threshold
        self._enum_normalize = enum_normalize
        self._normalize_threshold = normalize_threshold
        self._reject_unmatched = reject_unmatched
        self._extraction_scope = extraction_scope
        self._recent_messages_count = recent_messages_count
        self._conflict_strategy = conflict_strategy
        self._log_conflicts = log_conflicts
        self._per_turn_keys = per_turn_keys
        self._recovery_pipeline = recovery_pipeline
        self._boolean_recovery = boolean_recovery
        self._scope_escalation_enabled = scope_escalation_enabled
        self._scope_escalation_scope = scope_escalation_scope
        self._focused_retry_enabled = focused_retry_enabled
        self._focused_retry_max_retries = focused_retry_max_retries
        self._field_derivations = field_derivations

    # -----------------------------------------------------------------
    # Public API — called from wizard.py orchestration
    # -----------------------------------------------------------------

    async def run_extraction_pipeline(
        self,
        message: str,
        stage: dict[str, Any],
        state: WizardState,
        llm: Any,
        *,
        manager: Any | None = None,
    ) -> ExtractionPipelineResult:
        """Schema-driven extraction, normalization, merge, and recovery.

        Runs the full data-processing pipeline without any presentation
        concerns (no clarification responses, no confirmation templates).
        Used by both ``generate()`` (conversational) and ``advance()``
        (non-conversational) paths.

        The pipeline steps are:

        1. **Extract** -- LLM-driven schema extraction from raw text
        2. **Normalize** -- type coercion (bool, int, enum, array)
        3. **Merge** -- grounded merge into ``state.data``
        4. **Defaults** -- apply schema default values
        5. **Derivations** -- deterministic field relationships
        6. **Recovery** -- scope escalation, focused retry, boolean recovery
        7. **Confidence** -- assess extraction confidence with
           ``can_satisfy_required`` override

        Args:
            message: Raw user message text.
            stage: Current stage metadata (optionally includes ``schema``).
            state: Wizard state -- ``data`` is mutated in place.
            llm: LLM provider for extraction and recovery.
            manager: Optional conversation manager for message history
                context during extraction.

        Returns:
            :class:`ExtractionPipelineResult` with extraction result,
            new data keys, missing fields, and confidence assessment.
        """
        stage_name = stage.get("name", "unknown")

        # 1. Extract structured data from user input
        extraction = await self._extract_data(
            message, stage, llm, manager, state
        )

        logger.debug(
            "Extraction for stage '%s': confidence=%.2f, data_keys=%s",
            stage_name,
            extraction.confidence,
            list(extraction.data.keys()) if extraction.data else [],
        )
        if extraction.data:
            for key, value in extraction.data.items():
                if not key.startswith("_"):
                    logger.debug(
                        "  Extracted %s = %r", key, str(value)[:100]
                    )

        new_data_keys: set[str] = set()
        ss = StageSchema.from_stage(stage)

        # 2. Normalize extracted data (type coercion)
        if ss.exists and extraction.data:
            extraction.data = self._normalize_extracted_data(
                extraction.data, ss
            )

        # 3. Merge into wizard state (grounded merge)
        new_data_keys = self._merge_extraction_result(
            extraction.data, state, stage, message,
        )

        # 4. Apply schema defaults
        default_keys = self.apply_schema_defaults(state, stage)
        if default_keys:
            new_data_keys |= default_keys

        # 5. Post-extraction derivations
        derived = self.apply_field_derivations(state, stage)
        if derived:
            new_data_keys |= derived
            logger.debug(
                "Post-extraction derivation filled: %s",
                sorted(derived),
            )

        # 6. Recovery pipeline (only if required fields missing)
        missing = self.check_required_fields_missing(state, stage)
        if missing:
            new_data_keys, extraction = await self._run_recovery_pipeline(
                extraction, state, stage,
                message, llm, manager, new_data_keys,
            )
            # Re-check after recovery
            missing = self.check_required_fields_missing(state, stage)

        # Confidence assessment
        #
        # When extraction reports low confidence, check whether all
        # required fields are already satisfied.  StageSchema handles
        # all three cases uniformly via can_satisfy_required():
        #   - No schema -> no required fields -> vacuous True
        #   - Schema with required: [] -> vacuous True
        #   - Schema with required fields -> True only if all present
        is_confident = extraction.is_confident
        if not is_confident and ss.can_satisfy_required(state.data):
            is_confident = True

        return ExtractionPipelineResult(
            extraction=extraction,
            new_data_keys=new_data_keys,
            missing_fields=missing,
            is_confident=is_confident,
        )

    async def detect_intent(
        self,
        message: str,
        stage: dict[str, Any],
        state: WizardState,
        llm: Any,
    ) -> None:
        """Detect user intent and store in wizard state data.

        Examines the stage's ``intent_detection`` configuration and
        classifies the user message into one of the configured intents.
        The result is stored in ``state.data["_intent"]`` for use in
        transition conditions.

        Supports two detection methods:

        - **keyword**: Fast substring matching against configured keywords.
          First matching intent wins.
        - **llm**: Lightweight LLM classification.  Builds a prompt listing
          intents and their descriptions, asks the LLM to pick one.

        Args:
            message: Raw user message text
            stage: Current stage metadata (must contain ``intent_detection``)
            state: Current wizard state (``_intent`` is set here)
            llm: LLM provider instance (used only for ``method: llm``)
        """
        state.data.pop("_intent", None)

        intent_config = stage.get("intent_detection")
        if not intent_config:
            return

        method = intent_config.get("method", "keyword")
        intents = intent_config.get("intents", [])

        if method == "keyword":
            lower_msg = message.lower()
            for intent in intents:
                if any(kw in lower_msg for kw in intent.get("keywords", [])):
                    state.data["_intent"] = intent["id"]
                    logger.debug("Keyword intent detected: %s", intent["id"])
                    return

        elif method == "llm":
            intent_list = "\n".join(
                f"- {i['id']}: {i.get('description', '')}" for i in intents
            )
            prompt = (
                f"Classify the user's intent from this message:\n"
                f'"{message}"\n\n'
                f"Possible intents:\n{intent_list}\n\n"
                f"Return ONLY the intent ID, or 'none' if no intent matches."
            )
            try:
                from dataknobs_llm import LLMMessage

                response = await llm.complete(
                    messages=[LLMMessage(role="user", content=prompt)],
                )
                if response and response.content:
                    intent_id = response.content.strip().lower()
                    valid_ids = {i["id"] for i in intents}
                    if intent_id in valid_ids:
                        state.data["_intent"] = intent_id
                        logger.debug("LLM intent detected: %s", intent_id)
            except Exception as exc:
                logger.warning(
                    "LLM intent detection failed (%s)",
                    type(exc).__name__,
                    exc_info=True,
                )

    @staticmethod
    def is_done_signal(message: str, done_keywords: list[str]) -> bool:
        """Check whether a user message matches a collection done keyword."""
        if not done_keywords:
            return False
        normalised = message.strip().lower()
        return any(
            normalised == kw.strip().lower() for kw in done_keywords
        )

    @staticmethod
    def classify_collection_intent(
        message: str, stage: dict[str, Any],
    ) -> str:
        """Classify user intent during a collection-mode stage.

        Runs rule-based checks to distinguish help requests from data
        input **before** extraction.  Navigation and done signals are
        handled upstream (``WizardNavigator.handle_navigation`` and
        ``is_done_signal``), so this method only discriminates between:

        - ``"help"`` -- the user is asking a question about what to
          provide, not providing data.
        - ``"data_input"`` -- default; proceed to extraction.

        Custom help keywords can be supplied per-stage via
        ``collection_config.help_keywords``.

        Args:
            message: Raw user message text.
            stage: Current stage metadata dict.

        Returns:
            Intent string: ``"help"`` or ``"data_input"``.
        """
        msg = message.strip().lower()

        # Stage-configurable help keywords (exact match)
        col_config = stage.get("collection_config") or {}
        help_keywords = col_config.get("help_keywords", [])
        if help_keywords and any(msg == kw.strip().lower() for kw in help_keywords):
            return "help"

        # Built-in heuristic: question marks or common help phrasing
        if msg.endswith("?"):
            return "help"

        help_starters = (
            "what should i",
            "what do i",
            "what do you need",
            "what goes here",
            "help",
            "explain",
            "i don't understand",
            "i don't know what",
            "what kind of",
            "what format",
        )
        if any(msg.startswith(s) for s in help_starters):
            return "help"

        return "data_input"

    def check_required_fields_missing(
        self,
        wizard_state: WizardState,
        stage: dict[str, Any],
    ) -> set[str]:
        """Return required field names not yet present in wizard_state.data.

        Delegates to :meth:`StageSchema.missing_required` so that
        the "field presence" semantic is consistent with the confidence
        gate's ``can_satisfy_required()`` check.

        Args:
            wizard_state: Current wizard state with accumulated data
            stage: Stage configuration dict containing the schema

        Returns:
            Set of required field names whose values are absent or None
        """
        return StageSchema.from_stage(stage).missing_required(
            wizard_state.data,
        )

    @staticmethod
    def field_is_present(value: Any) -> bool:
        """A field has been provided if its value is not None.

        Centralises the "field presence" semantic used by the
        ``has()`` condition helper.  The confidence gate uses the
        equivalent logic via ``StageSchema.can_satisfy_required()``.

        Note: ``_can_auto_advance`` uses stricter logic -- it
        additionally rejects empty strings because auto-advance
        requires fields to be *filled*, not merely *present*.
        """
        return value is not None

    def apply_field_derivations(
        self,
        wizard_state: WizardState,
        stage: dict[str, Any],
    ) -> set[str]:
        """Apply field derivation rules to fill derivable fields.

        Called in two contexts:

        1. **Post-extraction pass** (unconditional) -- runs after merge
           and schema defaults, before the recovery pipeline check.
           Catches the common case of deriving optional fields from
           extracted required fields.
        2. **Recovery pipeline strategy** -- runs (by default first)
           when required fields are still missing after extraction.

        Derived values never overwrite user-provided or extracted data
        unless the rule specifies ``when: always``.

        Per-stage override: set ``derivation_enabled: false`` on a
        stage to suppress derivation for that stage (both contexts).

        Args:
            wizard_state: Current wizard state (data modified in-place).
            stage: Current stage metadata.

        Returns:
            Set of keys that were derived (newly added to data).
        """
        if not self._field_derivations:
            return set()

        # Per-stage override
        stage_enabled = stage.get("derivation_enabled")
        if stage_enabled is False:
            return set()

        return apply_field_derivations(
            self._field_derivations,
            wizard_state.data,
            field_is_present=self.field_is_present,
        )

    def get_extraction_scope(self, stage: dict[str, Any]) -> str:
        """Get extraction scope for a stage.

        Allows per-stage override of the global extraction_scope setting.

        Args:
            stage: Stage metadata dict

        Returns:
            Extraction scope from stage config or wizard default
        """
        return stage.get("extraction_scope") or self._extraction_scope

    def needs_llm_extraction(
        self, ss: StageSchema, stage: dict[str, Any],
    ) -> bool:
        """Determine whether LLM extraction is needed for a schema.

        Returns ``False`` when the schema describes a single required string
        field with no enum or format constraints -- the user's raw input can
        be used directly (verbatim capture).

        The decision can be overridden via ``collection_config.capture_mode``:

        - ``"auto"`` (default): use schema-based detection described above.
        - ``"verbatim"``: always skip LLM extraction.
        - ``"extract"``: always use LLM extraction.

        Args:
            ss: ``StageSchema`` for the current stage.
            stage: Current stage metadata dict.

        Returns:
            ``True`` if LLM extraction should be used, ``False`` for
            verbatim capture.
        """
        # capture_mode can be set as a top-level stage field or nested
        # under collection_config.  Top-level takes precedence.
        capture_mode = stage.get("capture_mode")
        if capture_mode is None:
            col_config = stage.get("collection_config") or {}
            capture_mode = col_config.get("capture_mode", "auto")

        if capture_mode == "verbatim":
            return False
        if capture_mode == "extract":
            return True

        # Auto-detect: single required string field with no constraints
        properties = ss.properties
        required = ss.required_fields

        if len(required) == 1 and len(properties) == 1:
            field_name = required[0]
            field_def = ss.get_property(field_name)
            if (
                field_def.get("type") == "string"
                and "enum" not in field_def
                and "pattern" not in field_def
                and "format" not in field_def
            ):
                return False

        return True

    def apply_schema_defaults(
        self, wizard_state: WizardState, stage: dict[str, Any],
    ) -> set[str]:
        """Apply schema defaults to wizard data for unset properties.

        After extraction, defaults defined in the stage schema (e.g.
        ``"default": "medium"``) are applied to any property that was
        not explicitly set by the user.  This ensures template conditions
        like ``{% if difficulty %}`` evaluate True even when the user
        didn't mention a value.

        Only top-level properties are considered -- nested object/array
        defaults are not auto-applied (they would require recursive
        merging that is unlikely to match user intent).

        Args:
            wizard_state: Current wizard state whose ``data`` may be
                updated in place.
            stage: Stage metadata dict containing ``schema``.

        Returns:
            Set of property names whose defaults were applied.
        """
        ss = StageSchema.from_stage(stage)
        if not ss.exists:
            return set()

        applied: set[str] = set()
        for prop_name, prop_def in ss.properties.items():
            if "default" not in prop_def:
                continue
            current = wizard_state.data.get(prop_name)
            if current is None:
                wizard_state.data[prop_name] = prop_def["default"]
                applied.add(prop_name)
                logger.debug(
                    "Applied schema default for '%s': %r",
                    prop_name,
                    prop_def["default"],
                )
        return applied

    # -----------------------------------------------------------------
    # Private — extraction core
    # -----------------------------------------------------------------

    async def _extract_data(
        self,
        message: str,
        stage: dict[str, Any],
        llm: Any,
        manager: Any | None = None,
        wizard_state: WizardState | None = None,
    ) -> Any:
        """Extract structured data from user message or wizard session.

        When extraction_scope is ``"wizard_session"`` or
        ``"recent_messages"``, builds context from user messages in the
        wizard session for extraction.  For ``"recent_messages"``, only
        the last ``recent_messages_count`` messages are included.  This
        allows the wizard to remember information provided in earlier
        messages.

        Schema 'default' values are stripped before extraction to prevent
        the LLM from auto-filling them. This ensures extraction only captures
        what the user actually said.

        Args:
            message: Current user message text
            stage: Current stage metadata
            llm: LLM provider (fallback if no extractor)
            manager: ConversationManager for accessing message history
            wizard_state: Current wizard state for conflict detection

        Returns:
            ExtractionResult with data and confidence
        """
        ss = StageSchema.from_stage(stage)
        stage_name = stage.get("name", "unknown")

        logger.debug(
            "Extraction start: stage='%s', has_schema=%s, "
            "has_extractor=%s, input_len=%d",
            stage_name,
            ss.exists,
            self._extractor is not None,
            len(message),
        )

        if not ss.exists:
            # No schema defined - pass through any data
            logger.debug(
                "Extraction skip: stage='%s' has no schema, returning raw input",
                stage_name,
            )
            return SimpleExtractionResult(
                data={"_raw_input": message}, confidence=1.0
            )

        # Verbatim capture: skip LLM extraction for trivial schemas
        # (single required string field, no constraints) or when
        # capture_mode is explicitly set to "verbatim".
        #
        # However, when a bot response is available in the conversation,
        # verbatim capture is unsafe: the user may be using deictic
        # references like "the first one" that require the bot's prior
        # response as context to resolve.  In that case, fall through to
        # LLM extraction so the bot-response prepending code can provide
        # the necessary context.
        has_bot_response = False
        if manager is not None:
            has_bot_response = bool(self._get_last_bot_response(manager))

        if not self.needs_llm_extraction(ss, stage) and not has_bot_response:
            field_name = next(iter(ss.properties))
            logger.debug(
                "Verbatim capture: stage='%s', field='%s'",
                stage_name,
                field_name,
            )
            return SimpleExtractionResult(
                data={field_name: message},
                confidence=1.0,
                metadata={"capture_mode": "verbatim"},
            )

        # Build extraction input based on scope (stage override or wizard default)
        extraction_scope = self.get_extraction_scope(stage)
        if (
            extraction_scope in ("wizard_session", "recent_messages")
            and manager is not None
            and wizard_state is not None
        ):
            # Build context from wizard session conversation.
            # For recent_messages scope, limit to last N user messages.
            max_msgs = (
                self._recent_messages_count
                if extraction_scope == "recent_messages"
                else None
            )
            wizard_context = self._build_wizard_context(
                manager, wizard_state, max_messages=max_msgs,
            )
            if wizard_context:
                extraction_input = (
                    f"{wizard_context}\n\nCurrent message: {message}"
                )
                logger.debug(
                    "Wizard session extraction: %d chars of context + current message",
                    len(wizard_context),
                )
            else:
                extraction_input = message
        else:
            # Current message only (original behavior)
            extraction_input = message

        # Include the bot's last response so the extraction model can
        # resolve references like "the first suggestion" or "yes to that".
        if manager is not None:
            bot_response = self._get_last_bot_response(manager)
            if bot_response:
                # Truncate very long responses to avoid overwhelming extraction
                if len(bot_response) > 1500:
                    bot_response = bot_response[:1500] + "..."
                extraction_input = (
                    f"Bot's previous message:\n{bot_response}\n\n"
                    f"User's response:\n{extraction_input}"
                )
                logger.debug(
                    "Included bot response (%d chars) in extraction context",
                    len(bot_response),
                )

        # Strip defaults to prevent extraction LLM from auto-filling them
        extraction_schema = self._strip_schema_defaults(ss.raw)

        if self._extractor:
            # Use schema extractor
            extraction_model = stage.get("extraction_model")
            context = {"stage": stage.get("name"), "prompt": stage.get("prompt")}
            result = await self._extractor.extract(
                text=extraction_input,
                schema=extraction_schema,
                context=context,
                model=extraction_model,
            )

            logger.debug(
                "Extraction result: stage='%s', keys=%s, confidence=%.2f, "
                "errors=%s",
                stage_name,
                list(result.data.keys()) if result.data else [],
                getattr(result, "confidence", -1.0),
                getattr(result, "errors", []),
            )

            # Detect conflicts with existing data
            if wizard_state is not None and result.data:
                conflicts = self._detect_conflicts(wizard_state.data, result.data)
                if conflicts:
                    if self._log_conflicts:
                        for conflict in conflicts:
                            logger.info(
                                "Data conflict detected for field '%s': "
                                "'%s' -> '%s' (using %s)",
                                conflict["field"],
                                conflict["previous"],
                                conflict["new"],
                                self._conflict_strategy,
                            )
                    # Add conflicts to result metadata for downstream use
                    if not hasattr(result, "metadata") or result.metadata is None:
                        result.metadata = {}
                    result.metadata["conflicts"] = conflicts

            return result

        # Fallback: simple heuristic extraction
        # This is very basic - the extractor should be used for real scenarios
        return SimpleExtractionResult(
            data={"_raw_input": message}, confidence=0.5
        )

    # -----------------------------------------------------------------
    # Private — context building
    # -----------------------------------------------------------------

    def _build_wizard_context(
        self,
        manager: Any,
        wizard_state: WizardState,
        *,
        max_messages: int | None = None,
    ) -> str:
        """Build extraction context from wizard session history.

        Collects user messages from the conversation to provide context
        for extraction. This allows the wizard to "remember" information
        provided in earlier messages.

        Prefers ``raw_content`` from node metadata when available, so
        that session-wide extraction context is not polluted by KB/memory
        augmentation from prior turns.

        Args:
            manager: ConversationManager instance
            wizard_state: Current wizard state
            max_messages: When set, include only the most recent *N*
                prior user messages (for ``recent_messages`` scope).
                ``None`` means include all prior messages (full session).

        Returns:
            Formatted context string from previous user messages,
            or empty string if no previous messages.
        """
        user_messages: list[str] = []

        # Use get_messages() for consistent access (includes raw_content
        # in metadata, aligned with _get_last_user_message approach).
        for msg in manager.get_messages():
            if msg.get("role") == "user":
                # Prefer raw_content from metadata (unaugmented user input)
                raw = msg.get("metadata", {}).get("raw_content")
                if raw is not None:
                    user_messages.append(raw)
                    continue
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_messages.append(content)
                elif isinstance(content, list):
                    # Handle structured content (list of content parts)
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            user_messages.append(part.get("text", ""))
                            break

        # Exclude the last message (it's the current one we're processing)
        previous_messages = user_messages[:-1] if len(user_messages) > 1 else []

        # Limit to most recent N messages for recent_messages scope
        if max_messages is not None and len(previous_messages) > max_messages:
            previous_messages = previous_messages[-max_messages:]

        if not previous_messages:
            return ""

        # Format as context
        formatted = ["Previous conversation:"]
        for i, msg in enumerate(previous_messages, 1):
            # Truncate very long messages
            truncated = msg[:500] + "..." if len(msg) > 500 else msg
            formatted.append(f"  Message {i}: {truncated}")

        return "\n".join(formatted)

    def _get_last_bot_response(self, manager: Any) -> str:
        """Extract the last assistant message from conversation.

        Used to provide the bot's previous response as context for
        extraction, so the extraction model can resolve references
        like "the first suggestion" or "yes to that".

        Args:
            manager: ConversationManager instance

        Returns:
            Last assistant message text, or empty string if none found.
        """
        messages = manager.get_messages()
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # Handle structured content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part.get("text", "")
        return ""

    # -----------------------------------------------------------------
    # Private — normalization and validation
    # -----------------------------------------------------------------

    def _normalize_extracted_data(
        self,
        data: dict[str, Any],
        ss: StageSchema,
    ) -> dict[str, Any]:
        """Normalize extracted data to match schema types.

        Applies deterministic, schema-driven corrections to LLM-extracted
        data *before* it enters wizard state.  Performs type coercion when
        the extracted type doesn't match the declared schema type, enum
        normalization for fuzzy matching, and enum rejection for values
        that are not valid entries.

        Normalizations performed:

        * **Boolean coercion** - string ``"yes"``/``"true"`` -> ``True``, etc.
        * **Array wrapping** - bare string for an ``array`` field -> ``[value]``
        * **Array shortcut expansion** - ``["all"]`` for ``array`` + ``items.enum``
          -> all enum values; ``["none"]`` -> ``[]``
        * **Number coercion** - string digits for ``integer``/``number`` -> cast
        * **Enum normalization** - string values for fields with ``enum``
          constraints -> matched to the canonical enum entry via
          case-insensitive and fuzzy matching when ``enum_normalize``
          is enabled (default ``True``)
        * **Enum rejection** - when ``reject_unmatched`` is enabled
          (default ``True``), string values that are not valid enum
          entries (after normalization, if active) are set to ``None``.
          The merge step skips ``None`` values, so the field is not
          stored in wizard state.  Works independently of normalization.

        Args:
            data: Extracted data dict (will be shallow-copied).
            ss:   ``StageSchema`` for the current stage.

        Returns:
            New dict with normalized values.  Fields set to ``None``
            indicate rejected values that should not be merged.
        """
        properties = ss.properties
        if not properties:
            return data

        normalized = dict(data)

        for field_name, value in data.items():
            if field_name.startswith("_") or field_name not in properties:
                continue

            prop = properties[field_name]
            declared_type = prop.get("type")

            # --- Boolean coercion ---
            if declared_type == "boolean" and isinstance(value, str):
                lower = value.strip().lower()
                if lower in self._BOOL_TRUE:
                    normalized[field_name] = True
                    logger.debug("Normalized %s: %r -> True", field_name, value)
                elif lower in self._BOOL_FALSE:
                    normalized[field_name] = False
                    logger.debug("Normalized %s: %r -> False", field_name, value)

            # --- Integer coercion ---
            elif declared_type == "integer" and isinstance(value, str):
                stripped = value.strip()
                if stripped.lstrip("-").isdigit():
                    normalized[field_name] = int(stripped)
                    logger.debug(
                        "Normalized %s: %r -> %d", field_name, value, int(stripped)
                    )

            # --- Number (float) coercion ---
            elif declared_type == "number" and isinstance(value, str):
                stripped = value.strip()
                try:
                    normalized[field_name] = float(stripped)
                    logger.debug(
                        "Normalized %s: %r -> %f",
                        field_name,
                        value,
                        float(stripped),
                    )
                except ValueError:
                    pass  # Leave as-is; validation will catch it

            # --- Array handling ---
            elif declared_type == "array":
                items_schema = prop.get("items", {})
                enum_values = items_schema.get("enum", [])

                # Wrap bare string -> list
                if isinstance(value, str):
                    value = [value]
                    normalized[field_name] = value
                    logger.debug(
                        "Normalized %s: wrapped string -> list", field_name
                    )

                # Expand "all"/"none" shortcuts when enum is defined
                if isinstance(value, list) and enum_values:
                    lower_items = {
                        v.strip().lower() for v in value if isinstance(v, str)
                    }
                    if lower_items & self._ALL_KEYWORDS:
                        normalized[field_name] = list(enum_values)
                        logger.debug(
                            "Normalized %s: 'all' -> %s",
                            field_name,
                            enum_values,
                        )
                    elif lower_items & self._NONE_KEYWORDS:
                        normalized[field_name] = []
                        logger.debug(
                            "Normalized %s: 'none' -> []", field_name
                        )

            # --- Enum normalization + rejection ---
            # Runs independently of type coercion above: a string field
            # with an enum constraint may have already been coerced (or
            # not), and the value still may not match the canonical enum
            # entry exactly.  Normalization tries fuzzy matching;
            # rejection drops values that don't match any enum entry.
            current_value = normalized[field_name]
            if (
                "enum" in prop
                and isinstance(current_value, str)
            ):
                x_ext = prop.get("x-extraction", {})
                should_normalize = x_ext.get(
                    "normalize", self._enum_normalize,
                )
                if should_normalize:
                    threshold = x_ext.get(
                        "normalize_threshold", self._normalize_threshold,
                    )
                    match = _normalize_enum_value(
                        current_value, prop["enum"], threshold=threshold,
                    )
                    if match is not None and match != current_value:
                        normalized[field_name] = match
                        logger.debug(
                            "Normalized %s enum: %r -> %r",
                            field_name, current_value, match,
                        )

                # Reject values that are not valid enum entries.
                # Runs after normalization (if enabled) so the check
                # sees the normalized value.  When normalization is
                # disabled, this acts as a strict enum membership check.
                final_value = normalized[field_name]
                if (
                    final_value is not None
                    and final_value not in prop["enum"]
                ):
                    should_reject = x_ext.get(
                        "reject_unmatched", self._reject_unmatched,
                    )
                    if should_reject:
                        normalized[field_name] = None
                        logger.debug(
                            "Rejected %s enum value %r: "
                            "no match in %s",
                            field_name, final_value,
                            prop["enum"],
                        )

        return normalized

    def validate_data(
        self, data: dict[str, Any], ss: StageSchema,
    ) -> list[str]:
        """Validate extracted data against stage schema.

        Args:
            data: Extracted data to validate
            ss: ``StageSchema`` to validate against

        Returns:
            List of validation error messages
        """
        errors: list[str] = []
        required = ss.required_fields
        properties = ss.properties

        # Check required fields
        for field_name in required:
            if field_name not in data or data[field_name] is None:
                errors.append(f"Missing required field: {field_name}")

        # Check enum constraints
        for name, value in data.items():
            if name.startswith("_"):
                continue  # Skip internal fields
            if name in properties:
                prop = properties[name]
                if "enum" in prop and value not in prop["enum"]:
                    errors.append(
                        f"Invalid value for {name}: must be one of {prop['enum']}"
                    )

        return errors

    # -----------------------------------------------------------------
    # Private — conflict detection
    # -----------------------------------------------------------------

    def _detect_conflicts(
        self,
        existing_data: dict[str, Any],
        new_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Detect conflicts between existing and newly extracted data.

        A conflict occurs when a field exists in both dicts with
        different non-None values.

        Args:
            existing_data: Data already in wizard state
            new_data: Newly extracted data

        Returns:
            List of conflict dicts with field, previous, and new values.
        """
        conflicts: list[dict[str, Any]] = []

        for field_name, new_value in new_data.items():
            # Skip internal fields and per-turn keys (expected to change each turn)
            if field_name.startswith("_") or field_name in self._per_turn_keys:
                continue

            # Skip if new value is None
            if new_value is None:
                continue

            # Check if field exists with a different value
            if field_name in existing_data:
                existing_value = existing_data[field_name]
                # Only count as conflict if existing is non-None and different
                if existing_value is not None and existing_value != new_value:
                    conflicts.append({
                        "field": field_name,
                        "previous": existing_value,
                        "new": new_value,
                    })

        return conflicts

    # -----------------------------------------------------------------
    # Private — schema defaults and stripping
    # -----------------------------------------------------------------

    def _strip_schema_defaults(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Deep-copy schema with 'default' removed from all properties.

        Schema defaults serve a different purpose (documenting valid defaults for
        consumers) than extraction (parsing what the user actually said). The
        extraction prompt already instructs: "If information is missing, omit
        the field."

        Args:
            schema: JSON Schema dict with potential 'default' values

        Returns:
            Copy of schema with all 'default' keys removed from properties
        """
        clean = copy.deepcopy(schema)

        # Handle properties at top level and nested
        self._strip_defaults_from_properties(clean)

        return clean

    def _strip_defaults_from_properties(self, schema_part: dict[str, Any]) -> None:
        """Recursively strip 'default' from properties in schema.

        Handles nested schemas (objects with nested properties, items in arrays).

        Args:
            schema_part: Schema or sub-schema dict to process in place
        """
        # Strip from direct properties
        for prop in schema_part.get("properties", {}).values():
            prop.pop("default", None)
            # Recurse into nested object properties
            if prop.get("type") == "object":
                self._strip_defaults_from_properties(prop)
            # Handle array items
            if prop.get("type") == "array" and isinstance(prop.get("items"), dict):
                self._strip_defaults_from_properties(prop["items"])

        # Handle allOf, anyOf, oneOf
        for key in ("allOf", "anyOf", "oneOf"):
            if key in schema_part:
                for sub_schema in schema_part[key]:
                    if isinstance(sub_schema, dict):
                        self._strip_defaults_from_properties(sub_schema)

    # -----------------------------------------------------------------
    # Private — merge
    # -----------------------------------------------------------------

    def _merge_extraction_result(
        self,
        extraction_data: dict[str, Any],
        wizard_state: WizardState,
        stage: dict[str, Any],
        user_message: str,
    ) -> set[str]:
        """Merge extracted data into wizard state, returning new/changed keys.

        Applies the grounding filter (per-stage override or wizard-level)
        to protect existing data from ungrounded overwrites.  Skips None
        values.

        Args:
            extraction_data: Dict of field->value from extraction
            wizard_state: Wizard state whose ``.data`` is updated in-place
            stage: Stage configuration dict (for schema and grounding config)
            user_message: Current user message (for grounding checks)

        Returns:
            Set of keys that were newly added or changed
        """
        ss = StageSchema.from_stage(stage)
        schema_props = ss.properties

        # Resolve merge filter: per-stage grounding override, then
        # fall back to wizard-level filter.
        #
        # When a stage explicitly sets extraction_grounding: true, it
        # overrides skip_builtin_grounding -- the stage-level opt-in
        # always creates a grounding filter as fallback.
        stage_grounding = stage.get("extraction_grounding")
        if stage_grounding is not None:
            if stage_grounding:
                # Stage explicitly enables grounding.  Use the
                # wizard-level composite filter if available, otherwise
                # create a fresh grounding filter.  This overrides
                # skip_builtin_grounding for this stage.
                active_filter: MergeFilter | None = (
                    self._merge_filter
                    or SchemaGroundingFilter(
                        overlap_threshold=self._grounding_overlap_threshold,
                    )
                )
            else:
                active_filter = None
        else:
            active_filter = self._merge_filter

        new_data_keys: set[str] = set()
        data_snapshot = dict(wizard_state.data)
        for k, v in extraction_data.items():
            if v is None:
                continue
            if active_filter is not None:
                existing = wizard_state.data.get(k)
                prop_def = schema_props.get(k, {})
                decision = active_filter.filter(
                    k, v, existing, user_message, prop_def,
                    data_snapshot,
                )
                if decision.action == "reject":
                    logger.debug(
                        "Merge filter rejected %s=%r: %s",
                        k, v, decision.reason,
                    )
                    continue
                if decision.action == "transform":
                    v = decision.value
                    logger.debug(
                        "Merge filter transformed %s -> %r: %s",
                        k, v, decision.reason,
                    )
            if k not in wizard_state.data or wizard_state.data[k] != v:
                new_data_keys.add(k)
                wizard_state.data[k] = v
        return new_data_keys

    # -----------------------------------------------------------------
    # Private — recovery pipeline
    # -----------------------------------------------------------------

    async def _run_recovery_pipeline(
        self,
        extraction: Any,
        wizard_state: WizardState,
        stage: dict[str, Any],
        user_message: str,
        llm: Any,
        manager: Any | None,
        new_data_keys: set[str],
    ) -> tuple[set[str], Any]:
        """Run recovery strategies until required fields are satisfied.

        Executes strategies in pipeline order, checking before each
        whether all required fields are still missing.  Short-circuits
        as soon as requirements are met.

        Args:
            extraction: Current extraction result (may be replaced by
                an escalated/retried result for the confidence gate).
            wizard_state: Wizard state (modified in-place during merges).
            stage: Current stage metadata.
            user_message: Raw user message for grounding.
            llm: LLM provider.
            manager: Optional conversation manager (``None`` when called
                from the ``advance()`` non-conversational path).
            new_data_keys: Set of new/changed keys (augmented in-place).

        Returns:
            Tuple of (updated new_data_keys, updated extraction result).
        """
        # Per-stage disable
        stage_recovery = stage.get("recovery_enabled")
        if stage_recovery is False:
            return new_data_keys, extraction

        for strategy in self._recovery_pipeline:
            # Check stop condition: all required fields satisfied
            missing = self.check_required_fields_missing(
                wizard_state, stage,
            )
            if not missing:
                break

            if strategy == RECOVERY_DERIVATION:
                derived = self.apply_field_derivations(
                    wizard_state, stage,
                )
                if derived:
                    new_data_keys |= derived
                    logger.debug(
                        "Recovery pipeline: derivation filled %s",
                        sorted(derived),
                    )

            elif strategy == RECOVERY_BOOLEAN:
                recovered = self._run_boolean_recovery(
                    wizard_state, stage, user_message,
                )
                if recovered:
                    new_data_keys |= recovered
                    logger.debug(
                        "Recovery pipeline: boolean_recovery filled %s",
                        sorted(recovered),
                    )

            elif strategy == RECOVERY_SCOPE_ESCALATION:
                escalated_keys, escalated_extraction = (
                    await self._run_scope_escalation(
                        extraction, wizard_state, stage,
                        user_message, llm, manager,
                    )
                )
                if escalated_keys:
                    new_data_keys |= escalated_keys
                    extraction = escalated_extraction

            elif strategy == RECOVERY_FOCUSED_RETRY:
                if self._focused_retry_enabled:
                    retry_keys, retry_extraction = (
                        await self._run_focused_retry(
                            wizard_state, stage,
                            user_message, llm, manager,
                        )
                    )
                    if retry_keys:
                        new_data_keys |= retry_keys
                        extraction = retry_extraction
                else:
                    logger.debug(
                        "Recovery pipeline: focused_retry in pipeline "
                        "but not enabled -- skipping",
                    )

            elif strategy == RECOVERY_CLARIFICATION:
                # Clarification is handled by the confidence gate
                # downstream.  Including it in the pipeline is a
                # no-op signal for documentation purposes.
                pass

        return new_data_keys, extraction

    async def _run_scope_escalation(
        self,
        extraction: Any,
        wizard_state: WizardState,
        stage: dict[str, Any],
        user_message: str,
        llm: Any,
        manager: Any | None,
    ) -> tuple[set[str], Any]:
        """Run scope escalation recovery strategy.

        When required fields are still missing and the current scope is
        narrower than the escalation target, retry extraction with a
        broader scope so that information from earlier turns can fill
        the gaps.

        Returns:
            Tuple of (new keys from escalation, updated extraction).
            Empty set and original extraction if escalation didn't fire.
        """
        if not self._scope_escalation_enabled:
            return set(), extraction

        effective_scope = self.get_extraction_scope(stage)
        target_breadth = SCOPE_BREADTH.get(
            self._scope_escalation_scope,
            SCOPE_BREADTH["wizard_session"],
        )
        current_breadth = SCOPE_BREADTH.get(effective_scope, 0)
        if current_breadth >= target_breadth:
            return set(), extraction

        missing = self.check_required_fields_missing(
            wizard_state, stage,
        )
        if not missing:
            return set(), extraction

        # Check for prior history using the same scope window
        # the escalated extraction will use.
        guard_max = (
            self._recent_messages_count
            if self._scope_escalation_scope == "recent_messages"
            else None
        )
        has_prior = bool(
            self._build_wizard_context(
                manager, wizard_state,
                max_messages=guard_max,
            )
        ) if manager is not None else False

        if not has_prior:
            return set(), extraction

        logger.debug(
            "Scope escalation: %d required fields "
            "missing after '%s' extraction: %s "
            "-- retrying with '%s' scope",
            len(missing),
            effective_scope,
            sorted(missing),
            self._scope_escalation_scope,
        )
        escalated_stage = {
            **stage,
            "extraction_scope": self._scope_escalation_scope,
        }
        escalated = await self._extract_data(
            user_message,
            escalated_stage,
            llm,
            manager,
            wizard_state,
        )
        if not escalated.data:
            return set(), extraction

        ss_esc = StageSchema.from_stage(stage)
        if ss_esc.exists:
            escalated.data = self._normalize_extracted_data(
                escalated.data, ss_esc,
            )
        escalated_keys = self._merge_extraction_result(
            escalated.data, wizard_state, stage, user_message,
        )
        if escalated_keys:
            return escalated_keys, escalated
        return set(), extraction

    async def _run_focused_retry(
        self,
        wizard_state: WizardState,
        stage: dict[str, Any],
        user_message: str,
        llm: Any,
        manager: Any | None,
    ) -> tuple[set[str], Any]:
        """Run focused retry -- extract only missing required fields.

        Builds a minimal schema containing only the missing required
        fields, then extracts using the full wizard session context.
        This is simpler for the LLM since fewer fields = easier task.

        Returns:
            Tuple of (new keys from retry, extraction result).
            Empty set and None if retry didn't produce data.
        """
        ss = StageSchema.from_stage(stage)
        if not ss.exists:
            return set(), None

        missing = ss.missing_required(wizard_state.data)
        if not missing:
            return set(), None

        # Build focused schema with only the missing fields
        properties = ss.properties
        focused_properties = {
            f: properties[f]
            for f in missing
            if f in properties
        }
        if not focused_properties:
            return set(), None

        focused_schema = {
            "type": "object",
            "properties": focused_properties,
            "required": list(missing),
        }

        # Build a focused stage with the minimal schema and broadest
        # scope for maximum context.  Force LLM extraction to prevent
        # verbatim capture when the focused schema has a single field.
        focused_stage = {
            **stage,
            "schema": focused_schema,
            "extraction_scope": "wizard_session",
            "capture_mode": "extract",
        }

        logger.debug(
            "Focused retry: extracting %d missing fields: %s",
            len(missing),
            sorted(missing),
        )

        for attempt in range(self._focused_retry_max_retries):
            retry_result = await self._extract_data(
                user_message,
                focused_stage,
                llm,
                manager,
                wizard_state,
            )
            if not retry_result.data:
                continue

            retry_result.data = self._normalize_extracted_data(
                retry_result.data, StageSchema.from_dict(focused_schema),
            )
            retry_keys = self._merge_extraction_result(
                retry_result.data, wizard_state, stage, user_message,
            )
            if retry_keys:
                logger.debug(
                    "Focused retry attempt %d filled: %s",
                    attempt + 1,
                    sorted(retry_keys),
                )
                return retry_keys, retry_result
            # Data was extracted but merge yielded no new keys
            # (already present or blocked by grounding filter).
            # Retrying with the same inputs won't help.
            break

        return set(), None

    def _run_boolean_recovery(
        self,
        wizard_state: WizardState,
        stage: dict[str, Any],
        user_message: str,
    ) -> set[str]:
        """Recover missing boolean fields via signal word detection.

        For each missing boolean field with ``boolean_recovery`` enabled,
        scans the user's message for affirmative/negative signal words
        and sets the field value deterministically.  No LLM call needed.

        When multiple boolean fields are missing, requires field-specific
        keywords in the message to avoid filling unrelated fields.  When
        only one boolean field is missing, the message is assumed to
        refer to it (no scope restriction).

        Signal word lists default to module-level constants but can be
        overridden per-field via ``x-extraction.affirmative_signals``,
        ``x-extraction.affirmative_phrases``,
        ``x-extraction.negative_signals``, and
        ``x-extraction.negative_phrases``.

        Args:
            wizard_state: Wizard state (modified in-place).
            stage: Current stage metadata with schema.
            user_message: Raw user message.

        Returns:
            Set of field names that were filled by recovery.
        """
        ss = StageSchema.from_stage(stage)
        if not ss.exists:
            return set()
        properties = ss.properties
        required_fields = set(ss.required_fields)

        msg_lower = user_message.lower()

        # Identify candidate boolean fields: required, missing, boolean
        # type, and boolean_recovery enabled.  Collect x-extraction
        # hints once per candidate for reuse in scope check and signal
        # resolution.
        candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for field_name, prop in properties.items():
            if prop.get("type") != "boolean":
                continue
            if field_name not in required_fields:
                continue
            if self.field_is_present(wizard_state.data.get(field_name)):
                continue
            x_ext = prop.get("x-extraction", {})
            enabled = x_ext.get("boolean_recovery", self._boolean_recovery)
            if enabled:
                candidates.append((field_name, prop, x_ext))

        if not candidates:
            logger.debug(
                "Boolean recovery: no eligible boolean fields "
                "(none missing, none with recovery enabled, or "
                "no boolean fields in schema)",
            )
            return set()

        # Scope restriction: when multiple boolean fields are missing,
        # require field keywords in the message to disambiguate.
        need_scope_check = len(candidates) > 1

        recovered: set[str] = set()
        for field_name, prop, x_ext in candidates:
            if need_scope_check:
                keywords = field_keywords(field_name, prop)
                if not keywords:
                    logger.warning(
                        "Boolean recovery: field %r has no extractable "
                        "keywords -- add a description to enable scope "
                        "restriction; recovery skipped",
                        field_name,
                    )
                    continue
                field_mentioned = any(
                    word_in_text(w, msg_lower) for w in keywords
                )
                if not field_mentioned:
                    logger.debug(
                        "Boolean recovery: skipping %s -- field keywords "
                        "not found in message (scope restriction)",
                        field_name,
                    )
                    continue

            # Resolve per-field signal overrides
            custom_aff = x_ext.get("affirmative_signals")
            aff_signals = (
                frozenset(custom_aff) if custom_aff is not None
                else _DEFAULT_AFFIRMATIVE_SIGNALS
            )
            custom_aff_phrases = x_ext.get("affirmative_phrases")
            aff_phrases = (
                tuple(custom_aff_phrases) if custom_aff_phrases is not None
                else _DEFAULT_AFFIRMATIVE_PHRASES
            )
            custom_neg = x_ext.get("negative_signals")
            neg_signals = (
                frozenset(custom_neg) if custom_neg is not None
                else _DEFAULT_NEGATIVE_SIGNALS
            )
            custom_neg_phrases = x_ext.get("negative_phrases")
            neg_phrases = (
                tuple(custom_neg_phrases) if custom_neg_phrases is not None
                else _DEFAULT_NEGATIVE_PHRASES
            )

            signal = detect_boolean_signal(
                msg_lower,
                affirmative_signals=aff_signals,
                affirmative_phrases=aff_phrases,
                negative_signals=neg_signals,
                negative_phrases=neg_phrases,
            )

            if signal is not None:
                wizard_state.data[field_name] = signal
                recovered.add(field_name)
                logger.debug(
                    "Boolean recovery: %s -> %s (signal detection)",
                    field_name, signal,
                )

        return recovered
