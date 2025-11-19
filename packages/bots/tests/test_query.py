"""Tests for query transformation utilities."""

import pytest

from dataknobs_bots.knowledge.query import (
    ContextualExpander,
    QueryTransformer,
    TransformerConfig,
)
from dataknobs_bots.knowledge.query.expander import Message, is_ambiguous_query


class TestTransformerConfig:
    """Tests for TransformerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TransformerConfig()
        assert config.enabled is False
        assert config.llm_provider == "ollama"
        assert config.llm_model == "llama3.2"
        assert config.num_queries == 3
        assert config.domain_context == ""

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TransformerConfig(
            enabled=True,
            llm_provider="openai",
            llm_model="gpt-4",
            num_queries=5,
            domain_context="prompt engineering",
        )
        assert config.enabled is True
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4"
        assert config.num_queries == 5
        assert config.domain_context == "prompt engineering"


class TestQueryTransformer:
    """Tests for QueryTransformer."""

    def test_disabled_returns_original(self):
        """Test that disabled transformer returns original query."""
        transformer = QueryTransformer(TransformerConfig(enabled=False))

        import asyncio
        result = asyncio.run(transformer.transform("test query"))

        assert result == ["test query"]

    def test_transform_without_init_raises(self):
        """Test that transform without initialize raises error."""
        transformer = QueryTransformer(TransformerConfig(enabled=True))

        import asyncio
        with pytest.raises(RuntimeError, match="not initialized"):
            asyncio.run(transformer.transform("test"))

    def test_default_config(self):
        """Test transformer with default config."""
        transformer = QueryTransformer()
        assert transformer.config.enabled is False

    def test_build_prompt(self):
        """Test building transformation prompt."""
        config = TransformerConfig(
            enabled=True,
            domain_context="prompt engineering",
            num_queries=3,
        )
        transformer = QueryTransformer(config)

        prompt = transformer._build_prompt("How do I use few-shot learning?", 3)

        assert "3" in prompt
        assert "prompt engineering" in prompt
        assert "few-shot learning" in prompt

    def test_parse_response_basic(self):
        """Test parsing LLM response."""
        transformer = QueryTransformer()

        response = """few-shot learning techniques
prompt examples
learning from demonstrations"""

        queries = transformer._parse_response(response, "fallback")

        assert len(queries) == 3
        assert "few-shot learning techniques" in queries

    def test_parse_response_with_numbering(self):
        """Test parsing response with numbered list."""
        transformer = QueryTransformer()

        response = """1. few-shot learning
2. prompt examples
3. demonstrations"""

        queries = transformer._parse_response(response, "fallback")

        assert len(queries) == 3
        assert "few-shot learning" in queries

    def test_parse_response_empty_fallback(self):
        """Test that empty response returns fallback."""
        transformer = QueryTransformer()

        queries = transformer._parse_response("", "fallback query")

        assert queries == ["fallback query"]

    def test_parse_response_strips_quotes(self):
        """Test that quotes are stripped from queries."""
        transformer = QueryTransformer()

        response = '"query one"\n\'query two\''

        queries = transformer._parse_response(response, "fallback")

        assert "query one" in queries
        assert "query two" in queries


class TestContextualExpander:
    """Tests for ContextualExpander."""

    def test_empty_history(self):
        """Test expansion with empty history."""
        expander = ContextualExpander()
        result = expander.expand("Show me an example", [])

        assert result == "Show me an example"

    def test_basic_expansion(self):
        """Test basic query expansion from history."""
        expander = ContextualExpander()
        history = [
            {"role": "user", "content": "Tell me about chain-of-thought prompting"},
            {"role": "assistant", "content": "Chain-of-thought is a technique..."},
            {"role": "user", "content": "Show me an example"},
        ]

        result = expander.expand("Show me an example", history[:-1])

        # Should include keywords from history
        assert "chain-of-thought" in result.lower()
        assert "Show me an example" in result

    def test_message_object_history(self):
        """Test expansion with Message objects."""
        expander = ContextualExpander()
        history = [
            Message(role="user", content="Explain prompt engineering basics"),
        ]

        result = expander.expand("More details", history)

        assert "prompt" in result.lower()
        assert "engineering" in result.lower()

    def test_max_context_turns(self):
        """Test that max context turns is respected."""
        expander = ContextualExpander(max_context_turns=1)
        history = [
            {"role": "user", "content": "First message about topic A"},
            {"role": "user", "content": "Second message about topic B"},
            {"role": "user", "content": "Third message about topic C"},
        ]

        result = expander.expand("More", history)

        # Should only consider last turn (topic C)
        assert "topic" in result.lower()

    def test_include_assistant(self):
        """Test including assistant messages in context."""
        expander = ContextualExpander(include_assistant=True)
        history = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "The answer involves neural networks"},
        ]

        result = expander.expand("Tell me more", history)

        # Should include keywords from assistant message
        assert "neural" in result.lower() or "networks" in result.lower()

    def test_exclude_assistant_default(self):
        """Test that assistant messages are excluded by default."""
        expander = ContextualExpander(include_assistant=False)
        history = [
            {"role": "assistant", "content": "unique_keyword_xyz"},
        ]

        result = expander.expand("More", history)

        assert "unique_keyword_xyz" not in result

    def test_stop_words_filtered(self):
        """Test that stop words are filtered from expansion."""
        expander = ContextualExpander()
        history = [
            {"role": "user", "content": "The and but or is are was neural network"},
        ]

        result = expander.expand("More", history)

        # Common stop words should not be in keywords
        keywords = result.replace("More", "").strip().split()
        assert "the" not in [k.lower() for k in keywords]
        assert "and" not in [k.lower() for k in keywords]

    def test_expand_with_topics(self):
        """Test expansion with custom topic extractor."""

        def extract_topics(text):
            # Simple mock extractor
            if "machine learning" in text.lower():
                return ["ML", "algorithms"]
            return []

        expander = ContextualExpander()
        history = [
            {"role": "user", "content": "Tell me about machine learning"},
        ]

        result = expander.expand_with_topics(
            "Show examples",
            history,
            topic_extractor=extract_topics,
        )

        assert "ML" in result
        assert "algorithms" in result

    def test_keyword_extraction(self):
        """Test keyword extraction from context."""
        expander = ContextualExpander()
        context = ["Deep reinforcement learning for robotics applications"]

        keywords = expander._extract_keywords(context)

        assert "deep" in keywords
        assert "reinforcement" in keywords
        assert "learning" in keywords


class TestIsAmbiguousQuery:
    """Tests for is_ambiguous_query function."""

    def test_short_queries(self):
        """Test that short queries are marked ambiguous."""
        assert is_ambiguous_query("Show me") is True
        assert is_ambiguous_query("Example") is True
        assert is_ambiguous_query("More") is True

    def test_demonstrative_words(self):
        """Test that queries with demonstratives are ambiguous."""
        assert is_ambiguous_query("Tell me more about this") is True
        assert is_ambiguous_query("Can you explain that technique") is True
        assert is_ambiguous_query("Show me another example") is True

    def test_clear_queries(self):
        """Test that clear queries are not ambiguous."""
        assert is_ambiguous_query("How do I configure OAuth authentication") is False
        assert is_ambiguous_query("Explain the difference between REST and GraphQL") is False

    def test_example_keyword(self):
        """Test that 'example' makes query ambiguous."""
        assert is_ambiguous_query("Show me an example of this") is True

    def test_case_insensitive(self):
        """Test that detection is case insensitive."""
        assert is_ambiguous_query("Tell me about THIS") is True
        assert is_ambiguous_query("ANOTHER example please") is True


class TestMessage:
    """Tests for Message dataclass."""

    def test_creation(self):
        """Test creating a Message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_different_roles(self):
        """Test different message roles."""
        user_msg = Message(role="user", content="Question")
        assistant_msg = Message(role="assistant", content="Answer")
        system_msg = Message(role="system", content="Instructions")

        assert user_msg.role == "user"
        assert assistant_msg.role == "assistant"
        assert system_msg.role == "system"
