"""Transition condition implementations.

This module provides concrete implementations of TransitionCondition for
common conversation flow patterns.
"""

import re
from typing import Dict, Any, List, Callable
from dataclasses import dataclass

from .flow import TransitionCondition


@dataclass
class AlwaysCondition(TransitionCondition):
    """Condition that always evaluates to True.

    Useful for unconditional transitions or fallback transitions.
    """

    name: str = "always"

    async def evaluate(self, response: str, context: Dict[str, Any]) -> bool:
        """Always returns True."""
        return True

    def to_fsm_function(self) -> str:
        """Return function name for FSM registration."""
        return f"always_{id(self)}"


@dataclass
class KeywordCondition(TransitionCondition):
    """Condition based on keyword matching in response.

    Evaluates to True if any of the specified keywords are found in the
    response (case-insensitive by default).

    Attributes:
        keywords: List of keywords to match
        case_sensitive: Whether matching should be case-sensitive
        match_whole_word: If True, only match complete words
    """

    keywords: List[str]
    case_sensitive: bool = False
    match_whole_word: bool = False

    async def evaluate(self, response: str, context: Dict[str, Any]) -> bool:
        """Check if any keyword is in the response."""
        text = response if self.case_sensitive else response.lower()

        for keyword in self.keywords:
            kw = keyword if self.case_sensitive else keyword.lower()

            if self.match_whole_word:
                # Use word boundaries
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, text):
                    return True
            else:
                if kw in text:
                    return True

        return False

    def to_fsm_function(self) -> str:
        """Return function name for FSM registration."""
        return f"keyword_{id(self)}"


@dataclass
class RegexCondition(TransitionCondition):
    """Condition based on regular expression matching.

    Evaluates to True if the regex pattern matches the response.

    Attributes:
        pattern: Regular expression pattern
        flags: Regex flags (re.IGNORECASE, etc.)
    """

    pattern: str
    flags: int = 0

    def __post_init__(self):
        """Compile the regex pattern."""
        self._compiled_pattern = re.compile(self.pattern, self.flags)

    async def evaluate(self, response: str, context: Dict[str, Any]) -> bool:
        """Check if pattern matches the response."""
        return bool(self._compiled_pattern.search(response))

    def to_fsm_function(self) -> str:
        """Return function name for FSM registration."""
        return f"regex_{id(self)}"


@dataclass
class LLMClassifierCondition(TransitionCondition):
    """Condition based on LLM classification of the response.

    Uses an LLM to classify the response into one of several categories.
    Evaluates to True if the classification matches the expected value.

    Attributes:
        classifier_prompt: Prompt template for classification
        expected_value: Expected classification result
        llm_config: Optional LLM configuration override
    """

    classifier_prompt: str
    expected_value: str
    llm_config: Dict[str, Any] | None = None

    async def evaluate(self, response: str, context: Dict[str, Any]) -> bool:
        """Use LLM to classify and check against expected value."""
        # Import here to avoid circular dependencies
        from dataknobs_llm.llm import create_llm_provider, LLMConfig

        # Get LLM provider from context or create new one
        llm = context.get('_llm_provider')
        if llm is None and self.llm_config:
            config = LLMConfig(**self.llm_config)
            llm = create_llm_provider(config)

        if llm is None:
            raise ValueError(
                "LLMClassifierCondition requires an LLM provider in context "
                "or llm_config parameter"
            )

        # Format the classifier prompt with the response
        prompt = self.classifier_prompt.replace("{{response}}", response)

        # Get classification from LLM
        result = await llm.complete(prompt)
        classification = result.content.strip().lower()

        # Check if it matches expected value
        return classification == self.expected_value.lower()

    def to_fsm_function(self) -> str:
        """Return function name for FSM registration."""
        return f"llm_classifier_{id(self)}"


@dataclass
class ContextCondition(TransitionCondition):
    """Condition based on context variables.

    Evaluates a condition based on values in the conversation context.

    Attributes:
        condition_func: Function that takes context and returns bool
    """

    condition_func: Callable[[Dict[str, Any]], bool]

    async def evaluate(self, response: str, context: Dict[str, Any]) -> bool:
        """Evaluate condition function with context."""
        return self.condition_func(context)

    def to_fsm_function(self) -> str:
        """Return function name for FSM registration."""
        return f"context_{id(self)}"


@dataclass
class CompositeCondition(TransitionCondition):
    """Condition that combines multiple conditions with AND/OR logic.

    Attributes:
        conditions: List of conditions to evaluate
        operator: 'and' or 'or'
    """

    conditions: List[TransitionCondition]
    operator: str = "and"  # 'and' or 'or'

    def __post_init__(self):
        """Validate operator."""
        if self.operator not in ("and", "or"):
            raise ValueError("operator must be 'and' or 'or'")

    async def evaluate(self, response: str, context: Dict[str, Any]) -> bool:
        """Evaluate all conditions with specified operator."""
        results = [
            await cond.evaluate(response, context)
            for cond in self.conditions
        ]

        if self.operator == "and":
            return all(results)
        else:  # or
            return any(results)

    def to_fsm_function(self) -> str:
        """Return function name for FSM registration."""
        return f"composite_{self.operator}_{id(self)}"


@dataclass
class SentimentCondition(TransitionCondition):
    """Condition based on sentiment analysis.

    Evaluates to True if the response sentiment matches the expected sentiment.

    Attributes:
        expected_sentiment: Expected sentiment ('positive', 'negative', 'neutral')
        threshold: Confidence threshold (0.0 to 1.0)
    """

    expected_sentiment: str
    threshold: float = 0.5

    def __post_init__(self):
        """Validate sentiment value."""
        valid_sentiments = ('positive', 'negative', 'neutral')
        if self.expected_sentiment not in valid_sentiments:
            raise ValueError(
                f"expected_sentiment must be one of {valid_sentiments}"
            )

    async def evaluate(self, response: str, context: Dict[str, Any]) -> bool:
        """Analyze sentiment and check against expected value."""
        # Simple keyword-based sentiment analysis (can be replaced with ML model)
        response_lower = response.lower()

        positive_words = {'happy', 'good', 'great', 'excellent', 'yes', 'sure', 'love', 'like'}
        negative_words = {'sad', 'bad', 'terrible', 'no', 'hate', 'dislike', 'poor'}

        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)

        total = positive_count + negative_count
        if total == 0:
            sentiment = 'neutral'
            confidence = 1.0
        else:
            if positive_count > negative_count:
                sentiment = 'positive'
                confidence = positive_count / total
            elif negative_count > positive_count:
                sentiment = 'negative'
                confidence = negative_count / total
            else:
                sentiment = 'neutral'
                confidence = 0.5

        return sentiment == self.expected_sentiment and confidence >= self.threshold

    def to_fsm_function(self) -> str:
        """Return function name for FSM registration."""
        return f"sentiment_{self.expected_sentiment}_{id(self)}"


# Factory functions for common conditions

def keyword_condition(keywords: List[str], **kwargs) -> KeywordCondition:
    """Create a keyword condition.

    Args:
        keywords: List of keywords to match
        **kwargs: Additional KeywordCondition parameters

    Returns:
        KeywordCondition instance
    """
    return KeywordCondition(keywords=keywords, **kwargs)


def regex_condition(pattern: str, **kwargs) -> RegexCondition:
    """Create a regex condition.

    Args:
        pattern: Regular expression pattern
        **kwargs: Additional RegexCondition parameters

    Returns:
        RegexCondition instance
    """
    return RegexCondition(pattern=pattern, **kwargs)


def always() -> AlwaysCondition:
    """Create an always-true condition.

    Returns:
        AlwaysCondition instance
    """
    return AlwaysCondition()


def context_condition(func: Callable[[Dict[str, Any]], bool]) -> ContextCondition:
    """Create a context condition.

    Args:
        func: Function that takes context dict and returns bool

    Returns:
        ContextCondition instance
    """
    return ContextCondition(condition_func=func)
