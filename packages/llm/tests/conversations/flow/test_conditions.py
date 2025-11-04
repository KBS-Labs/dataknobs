"""Tests for transition conditions."""

import pytest
import re
from dataknobs_llm.conversations.flow.conditions import (
    AlwaysCondition,
    KeywordCondition,
    RegexCondition,
    ContextCondition,
    CompositeCondition,
    SentimentCondition,
    keyword_condition,
    regex_condition,
    always,
    context_condition,
)


@pytest.mark.asyncio
async def test_always_condition():
    """Test AlwaysCondition."""
    cond = AlwaysCondition()

    assert await cond.evaluate("any text", {})
    assert await cond.evaluate("", {})


@pytest.mark.asyncio
async def test_keyword_condition():
    """Test KeywordCondition."""
    cond = KeywordCondition(keywords=["help", "support"])

    assert await cond.evaluate("I need help", {})
    assert await cond.evaluate("contact support", {})
    assert not await cond.evaluate("just browsing", {})


@pytest.mark.asyncio
async def test_keyword_condition_case_sensitive():
    """Test case-sensitive keyword matching."""
    cond = KeywordCondition(keywords=["Help"], case_sensitive=True)

    assert await cond.evaluate("I need Help", {})
    assert not await cond.evaluate("I need help", {})


@pytest.mark.asyncio
async def test_keyword_condition_whole_word():
    """Test whole-word keyword matching."""
    cond = KeywordCondition(keywords=["cat"], match_whole_word=True)

    assert await cond.evaluate("I have a cat", {})
    assert not await cond.evaluate("I have a catalog", {})


@pytest.mark.asyncio
async def test_regex_condition():
    """Test RegexCondition."""
    cond = RegexCondition(pattern=r"\d{3}-\d{4}")

    assert await cond.evaluate("My number is 555-1234", {})
    assert not await cond.evaluate("My number is 5551234", {})


@pytest.mark.asyncio
async def test_regex_condition_with_flags():
    """Test RegexCondition with flags."""
    cond = RegexCondition(pattern=r"hello", flags=re.IGNORECASE)

    assert await cond.evaluate("Hello there", {})
    assert await cond.evaluate("HELLO WORLD", {})


@pytest.mark.asyncio
async def test_context_condition():
    """Test ContextCondition."""
    cond = ContextCondition(
        condition_func=lambda ctx: ctx.get("ready", False)
    )

    assert await cond.evaluate("any text", {"ready": True})
    assert not await cond.evaluate("any text", {"ready": False})
    assert not await cond.evaluate("any text", {})


@pytest.mark.asyncio
async def test_composite_condition_and():
    """Test CompositeCondition with AND operator."""
    cond1 = KeywordCondition(keywords=["hello"])
    cond2 = KeywordCondition(keywords=["world"])

    composite = CompositeCondition(
        conditions=[cond1, cond2],
        operator="and"
    )

    assert await composite.evaluate("hello world", {})
    assert not await composite.evaluate("hello", {})
    assert not await composite.evaluate("world", {})


@pytest.mark.asyncio
async def test_composite_condition_or():
    """Test CompositeCondition with OR operator."""
    cond1 = KeywordCondition(keywords=["hello"])
    cond2 = KeywordCondition(keywords=["world"])

    composite = CompositeCondition(
        conditions=[cond1, cond2],
        operator="or"
    )

    assert await composite.evaluate("hello world", {})
    assert await composite.evaluate("hello", {})
    assert await composite.evaluate("world", {})
    assert not await composite.evaluate("goodbye", {})


@pytest.mark.asyncio
async def test_sentiment_condition_positive():
    """Test SentimentCondition for positive sentiment."""
    cond = SentimentCondition(expected_sentiment="positive", threshold=0.5)

    assert await cond.evaluate("I love this! It's great!", {})
    assert not await cond.evaluate("I hate this. It's terrible.", {})


@pytest.mark.asyncio
async def test_sentiment_condition_negative():
    """Test SentimentCondition for negative sentiment."""
    cond = SentimentCondition(expected_sentiment="negative", threshold=0.5)

    assert await cond.evaluate("I hate this. It's terrible.", {})
    assert not await cond.evaluate("I love this! It's great!", {})


@pytest.mark.asyncio
async def test_sentiment_condition_neutral():
    """Test SentimentCondition for neutral sentiment."""
    cond = SentimentCondition(expected_sentiment="neutral")

    assert await cond.evaluate("This is a statement.", {})
    assert await cond.evaluate("The sky is blue.", {})


def test_keyword_condition_factory():
    """Test keyword_condition factory function."""
    cond = keyword_condition(["test", "example"])

    assert isinstance(cond, KeywordCondition)
    assert cond.keywords == ["test", "example"]


def test_regex_condition_factory():
    """Test regex_condition factory function."""
    cond = regex_condition(r"\d+")

    assert isinstance(cond, RegexCondition)
    assert cond.pattern == r"\d+"


def test_always_factory():
    """Test always factory function."""
    cond = always()

    assert isinstance(cond, AlwaysCondition)


def test_context_condition_factory():
    """Test context_condition factory function."""
    func = lambda ctx: True
    cond = context_condition(func)

    assert isinstance(cond, ContextCondition)
    assert cond.condition_func is func
