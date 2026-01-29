"""Focus guards for maintaining conversation focus in ReAct reasoning.

This module provides tools to detect and correct conversational drift,
ensuring the LLM stays focused on the current task during multi-turn
reasoning loops.

The focus guard system enables:
- Detection of off-topic responses
- Gentle redirection to the main task
- Configurable tangent tolerance
- Focus context injection into prompts

Example:
    ```python
    guard = FocusGuard(max_tangent_depth=2)

    # Build focus context from conversation
    focus_context = guard.build_context(
        primary_goal="Help user configure their bot",
        current_task="Collect the bot's name",
        collected_data={"domain": "education"},
    )

    # Get focus prompt to inject
    focus_prompt = guard.get_focus_prompt(focus_context)

    # Evaluate a response for drift
    evaluation = guard.evaluate_response(
        response_text="Let me tell you about the history of chatbots...",
        focus_context=focus_context,
    )

    if evaluation.is_drifting:
        correction_prompt = guard.get_correction_prompt(evaluation)
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..context.accumulator import ConversationContext

logger = logging.getLogger(__name__)


@dataclass
class FocusContext:
    """Context defining what the conversation should focus on.

    Captures the primary goal, current task, and relevant context
    to help evaluate whether responses are on-topic.

    Attributes:
        primary_goal: The overall objective (e.g., "Configure a bot")
        current_task: The immediate task being worked on
        collected_data: Data already gathered (to avoid re-asking)
        required_fields: Fields still needed
        stage_name: Current wizard stage (if applicable)
        tangent_count: Number of consecutive off-topic turns
        max_tangent_depth: Maximum allowed consecutive tangents
        topic_keywords: Keywords that indicate on-topic content
        off_topic_keywords: Keywords that indicate off-topic content
    """

    primary_goal: str
    current_task: str | None = None
    collected_data: dict[str, Any] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    stage_name: str | None = None
    tangent_count: int = 0
    max_tangent_depth: int = 2
    topic_keywords: list[str] = field(default_factory=list)
    off_topic_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "primary_goal": self.primary_goal,
            "current_task": self.current_task,
            "collected_data": self.collected_data,
            "required_fields": self.required_fields,
            "stage_name": self.stage_name,
            "tangent_count": self.tangent_count,
            "max_tangent_depth": self.max_tangent_depth,
            "topic_keywords": self.topic_keywords,
            "off_topic_keywords": self.off_topic_keywords,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FocusContext:
        """Create from dictionary."""
        return cls(**data)

    @property
    def is_at_tangent_limit(self) -> bool:
        """Check if tangent count has reached the limit."""
        return self.tangent_count >= self.max_tangent_depth


@dataclass
class FocusEvaluation:
    """Result of evaluating a response for focus drift.

    Attributes:
        is_drifting: Whether the response is off-topic
        drift_severity: How far off-topic (0.0 = on topic, 1.0 = completely off)
        detected_topic: What topic the response appears to be about
        reason: Explanation of why drift was detected
        suggested_redirect: Suggested topic to redirect to
        tangent_count: Updated tangent count
    """

    is_drifting: bool = False
    drift_severity: float = 0.0
    detected_topic: str | None = None
    reason: str | None = None
    suggested_redirect: str | None = None
    tangent_count: int = 0

    @property
    def needs_correction(self) -> bool:
        """Check if correction is needed based on drift and severity."""
        return self.is_drifting and self.drift_severity > 0.5


class FocusGuard:
    r"""Guards against conversational drift in ReAct reasoning.

    Provides methods to build focus context, evaluate responses for drift,
    and generate correction prompts when needed.

    Attributes:
        max_tangent_depth: Maximum consecutive off-topic turns allowed
        drift_threshold: Severity threshold for triggering correction
        use_keyword_detection: Whether to use keyword-based detection
        use_llm_evaluation: Whether to use LLM for drift evaluation

    Example:
        ```python
        guard = FocusGuard(max_tangent_depth=2)

        # Build context
        context = guard.build_context(
            primary_goal="Help configure the bot",
            current_task="Get the bot name",
        )

        # Inject focus prompt
        system_prompt = f"{original_prompt}\n\n{guard.get_focus_prompt(context)}"

        # After response, evaluate
        evaluation = guard.evaluate_response(response.text, context)

        if evaluation.needs_correction:
            # Add correction to next turn
            correction = guard.get_correction_prompt(evaluation)
        ```
    """

    def __init__(
        self,
        max_tangent_depth: int = 2,
        drift_threshold: float = 0.5,
        use_keyword_detection: bool = True,
        use_llm_evaluation: bool = False,
    ) -> None:
        """Initialize FocusGuard.

        Args:
            max_tangent_depth: Maximum consecutive off-topic turns
            drift_threshold: Severity threshold for correction (0.0-1.0)
            use_keyword_detection: Use keyword-based drift detection
            use_llm_evaluation: Use LLM for drift evaluation (not yet implemented)
        """
        self.max_tangent_depth = max_tangent_depth
        self.drift_threshold = drift_threshold
        self.use_keyword_detection = use_keyword_detection
        self.use_llm_evaluation = use_llm_evaluation

    def build_context(
        self,
        primary_goal: str,
        current_task: str | None = None,
        collected_data: dict[str, Any] | None = None,
        required_fields: list[str] | None = None,
        stage_name: str | None = None,
        topic_keywords: list[str] | None = None,
        off_topic_keywords: list[str] | None = None,
    ) -> FocusContext:
        """Build a focus context for evaluation.

        Args:
            primary_goal: The main objective
            current_task: Immediate task being worked on
            collected_data: Data already gathered
            required_fields: Fields still needed
            stage_name: Current wizard stage
            topic_keywords: Keywords indicating on-topic content
            off_topic_keywords: Keywords indicating off-topic content

        Returns:
            FocusContext ready for evaluation
        """
        return FocusContext(
            primary_goal=primary_goal,
            current_task=current_task,
            collected_data=collected_data or {},
            required_fields=required_fields or [],
            stage_name=stage_name,
            tangent_count=0,
            max_tangent_depth=self.max_tangent_depth,
            topic_keywords=topic_keywords or [],
            off_topic_keywords=off_topic_keywords or [],
        )

    def build_context_from_conversation(
        self,
        conversation_context: ConversationContext,
        current_task: str | None = None,
    ) -> FocusContext:
        """Build focus context from a ConversationContext.

        Args:
            conversation_context: ConversationContext from context builder
            current_task: Override for current task

        Returns:
            FocusContext derived from conversation context
        """
        # Extract primary goal from context
        primary_goal = "Complete the conversation objective"

        # Try to get goal from context sections
        sections = getattr(conversation_context, "sections", [])
        for section in sections:
            if hasattr(section, "title") and "goal" in section.title.lower():
                primary_goal = getattr(section, "content", primary_goal)
                break

        # Get collected data from artifacts
        collected_data = {}
        artifacts = getattr(conversation_context, "artifact_summaries", [])
        for artifact_summary in artifacts:
            content = artifact_summary.get("content")
            if content and isinstance(content, dict):
                collected_data.update(content)

        return FocusContext(
            primary_goal=primary_goal,
            current_task=current_task,
            collected_data=collected_data,
            max_tangent_depth=self.max_tangent_depth,
        )

    def get_focus_prompt(self, context: FocusContext) -> str:
        """Generate a focus prompt to inject into system message.

        This prompt reminds the LLM to stay on topic.

        Args:
            context: Current focus context

        Returns:
            Focus prompt string
        """
        lines = ["## Focus Guidance"]
        lines.append(f"**Primary Goal**: {context.primary_goal}")

        if context.current_task:
            lines.append(f"**Current Task**: {context.current_task}")

        if context.required_fields:
            lines.append(f"**Still Needed**: {', '.join(context.required_fields)}")

        if context.collected_data:
            # Summarize what's already collected
            collected = list(context.collected_data.keys())
            if len(collected) > 5:
                collected = collected[:5] + [f"...and {len(collected) - 5} more"]
            lines.append(f"**Already Have**: {', '.join(collected)}")

        lines.append("")
        lines.append("Stay focused on the current task. If the user asks about ")
        lines.append("something unrelated, acknowledge briefly and redirect to ")
        lines.append("the task at hand.")

        return "\n".join(lines)

    def evaluate_response(
        self,
        response_text: str,
        focus_context: FocusContext,
    ) -> FocusEvaluation:
        """Evaluate a response for focus drift.

        Uses keyword detection (and optionally LLM) to determine if
        the response is staying on topic.

        Args:
            response_text: The response text to evaluate
            focus_context: Current focus context

        Returns:
            FocusEvaluation with drift assessment
        """
        if not response_text:
            return FocusEvaluation()

        # Keyword-based evaluation
        if self.use_keyword_detection:
            evaluation = self._keyword_evaluation(response_text, focus_context)
        else:
            evaluation = FocusEvaluation()

        # Update tangent count
        if evaluation.is_drifting:
            evaluation.tangent_count = focus_context.tangent_count + 1
        else:
            evaluation.tangent_count = 0

        return evaluation

    def _keyword_evaluation(
        self,
        response_text: str,
        context: FocusContext,
    ) -> FocusEvaluation:
        """Evaluate response using keyword detection.

        Args:
            response_text: Response to evaluate
            context: Focus context with keywords

        Returns:
            FocusEvaluation based on keyword analysis
        """
        text_lower = response_text.lower()

        # Check for off-topic keywords
        off_topic_found = []
        for keyword in context.off_topic_keywords:
            if keyword.lower() in text_lower:
                off_topic_found.append(keyword)

        # Check for on-topic keywords
        on_topic_found = []
        for keyword in context.topic_keywords:
            if keyword.lower() in text_lower:
                on_topic_found.append(keyword)

        # Also check goal and task keywords
        goal_words = context.primary_goal.lower().split()
        goal_matches = sum(1 for word in goal_words if word in text_lower and len(word) > 3)

        task_matches = 0
        if context.current_task:
            task_words = context.current_task.lower().split()
            task_matches = sum(
                1 for word in task_words if word in text_lower and len(word) > 3
            )

        # Calculate drift severity
        total_on_topic = len(on_topic_found) + goal_matches + task_matches
        total_off_topic = len(off_topic_found)

        if total_on_topic == 0 and total_off_topic == 0:
            # Can't determine, assume on topic
            return FocusEvaluation(is_drifting=False, drift_severity=0.0)

        if total_on_topic == 0 and total_off_topic > 0:
            severity = min(1.0, total_off_topic * 0.3)
            return FocusEvaluation(
                is_drifting=True,
                drift_severity=severity,
                detected_topic="off-topic content",
                reason=f"Found off-topic keywords: {', '.join(off_topic_found[:3])}",
                suggested_redirect=context.current_task or context.primary_goal,
            )

        if total_off_topic > total_on_topic:
            severity = (total_off_topic - total_on_topic) / (
                total_off_topic + total_on_topic
            )
            return FocusEvaluation(
                is_drifting=severity > self.drift_threshold,
                drift_severity=severity,
                detected_topic="mixed content",
                reason="More off-topic content than on-topic",
                suggested_redirect=context.current_task or context.primary_goal,
            )

        return FocusEvaluation(
            is_drifting=False,
            drift_severity=0.0,
        )

    def get_correction_prompt(self, evaluation: FocusEvaluation) -> str:
        """Generate a correction prompt to redirect the conversation.

        Args:
            evaluation: FocusEvaluation indicating drift

        Returns:
            Correction prompt string
        """
        if not evaluation.is_drifting:
            return ""

        lines = ["## Focus Correction Needed"]

        if evaluation.reason:
            lines.append(f"Issue: {evaluation.reason}")

        if evaluation.suggested_redirect:
            lines.append(f"Redirect to: {evaluation.suggested_redirect}")

        if evaluation.tangent_count >= self.max_tangent_depth:
            lines.append("")
            lines.append(
                "**IMPORTANT**: The conversation has drifted off-topic for "
                f"{evaluation.tangent_count} turns. Please acknowledge the "
                "tangent briefly and firmly redirect to the main task."
            )
        else:
            lines.append("")
            lines.append(
                "Please gently steer the conversation back to the main topic "
                "while acknowledging what the user mentioned."
            )

        return "\n".join(lines)

    def update_context_after_evaluation(
        self,
        context: FocusContext,
        evaluation: FocusEvaluation,
    ) -> FocusContext:
        """Update focus context based on evaluation results.

        Args:
            context: Current focus context
            evaluation: Evaluation results

        Returns:
            Updated focus context
        """
        # Create new context with updated tangent count
        return FocusContext(
            primary_goal=context.primary_goal,
            current_task=context.current_task,
            collected_data=context.collected_data,
            required_fields=context.required_fields,
            stage_name=context.stage_name,
            tangent_count=evaluation.tangent_count,
            max_tangent_depth=context.max_tangent_depth,
            topic_keywords=context.topic_keywords,
            off_topic_keywords=context.off_topic_keywords,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FocusGuard:
        """Create FocusGuard from configuration.

        Config format:
        ```yaml
        focus_guard:
          max_tangent_depth: 2
          drift_threshold: 0.5
          use_keyword_detection: true
        ```

        Args:
            config: Configuration dict

        Returns:
            Configured FocusGuard
        """
        return cls(
            max_tangent_depth=config.get("max_tangent_depth", 2),
            drift_threshold=config.get("drift_threshold", 0.5),
            use_keyword_detection=config.get("use_keyword_detection", True),
            use_llm_evaluation=config.get("use_llm_evaluation", False),
        )
