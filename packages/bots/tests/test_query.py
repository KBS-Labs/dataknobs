"""Tests for query transformation utilities."""

import pytest

from dataknobs_bots.knowledge.query import (
    ContextualExpander,
    QueryTransformer,
    TransformerConfig,
    parse_query_response,
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


class TestParseQueryResponse:
    """Tests for parse_query_response module-level function."""

    def test_basic_lines(self) -> None:
        queries = parse_query_response("query one\nquery two\nquery three", "fallback")
        assert queries == ["query one", "query two", "query three"]

    def test_numbered_list(self) -> None:
        queries = parse_query_response("1. first query\n2. second query", "fallback")
        assert queries == ["first query", "second query"]

    def test_empty_returns_fallback(self) -> None:
        assert parse_query_response("", "fallback") == ["fallback"]

    def test_strips_quotes(self) -> None:
        queries = parse_query_response('"quoted query"\n\'single quoted\'', "fb")
        assert "quoted query" in queries
        assert "single quoted" in queries

    def test_filters_short_lines(self) -> None:
        queries = parse_query_response("ok\nvalid query\nno", "fallback")
        assert queries == ["valid query"]


class TestQueryTransformerExternalProvider:
    """Tests for QueryTransformer with externally-injected providers."""

    @pytest.mark.asyncio
    async def test_constructor_provider(self) -> None:
        """External provider makes transformer immediately ready."""
        from dataknobs_llm import EchoProvider
        from dataknobs_llm.testing import text_response

        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([text_response("OAuth grant types\nauthorization code")])

        config = TransformerConfig(enabled=True, num_queries=2)
        transformer = QueryTransformer(config, provider=provider)

        queries = await transformer.transform("What are OAuth grants?")
        assert len(queries) == 2
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_set_provider(self) -> None:
        """set_provider() makes transformer ready without initialize()."""
        from dataknobs_llm import EchoProvider
        from dataknobs_llm.testing import text_response

        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([text_response("search query one\nsearch query two")])

        config = TransformerConfig(enabled=True, num_queries=2)
        transformer = QueryTransformer(config)

        # Not initialized yet
        with pytest.raises(RuntimeError, match="not initialized"):
            await transformer.transform("test")

        # Now inject provider
        transformer.set_provider(provider)
        queries = await transformer.transform("test question")
        assert len(queries) == 2

    @pytest.mark.asyncio
    async def test_close_does_not_close_external_provider(self) -> None:
        """close() does not close externally-provided providers."""
        from dataknobs_llm import EchoProvider

        provider = EchoProvider({"provider": "echo", "model": "test"})
        transformer = QueryTransformer(
            TransformerConfig(enabled=True),
            provider=provider,
        )

        await transformer.close()
        # Transformer is no longer initialized, but provider is still alive
        assert transformer._initialized is False
        # Provider should still work (not closed)
        response = await provider.complete("test")
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_transform_with_context(self) -> None:
        """transform_with_context uses contextual prompt."""
        from dataknobs_llm import EchoProvider
        from dataknobs_llm.testing import text_response

        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([text_response("context aware query")])

        transformer = QueryTransformer(
            TransformerConfig(enabled=True, domain_context="OAuth 2.0"),
            provider=provider,
        )

        queries = await transformer.transform_with_context(
            "What about refresh tokens?",
            "user: How do OAuth grants work?\nassistant: There are several types...",
        )
        assert len(queries) >= 1
        # Verify the prompt included conversation context
        last_call = provider.get_last_call()
        assert last_call is not None
        # The prompt is the first message content
        prompt = last_call["messages"][0].content if hasattr(last_call["messages"][0], "content") else str(last_call["messages"][0])
        assert "conversation context" in prompt.lower()
        assert "OAuth 2.0" in prompt


class TestSuppressThinking:
    """Tests for suppress_thinking config option."""

    @pytest.mark.asyncio
    async def test_suppress_thinking_passes_config_override(self) -> None:
        """When suppress_thinking=True, think: False is in config_overrides."""
        from dataknobs_llm import EchoProvider
        from dataknobs_llm.testing import text_response

        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([text_response("query one\nquery two")])

        config = TransformerConfig(enabled=True, num_queries=2, suppress_thinking=True)
        transformer = QueryTransformer(config, provider=provider)

        await transformer.transform("test question")
        last_call = provider.get_last_call()
        assert last_call is not None
        overrides = last_call.get("config_overrides", {})
        assert overrides.get("options", {}).get("think") is False

    @pytest.mark.asyncio
    async def test_no_suppress_thinking_no_override(self) -> None:
        """When suppress_thinking=False (default), no config_overrides."""
        from dataknobs_llm import EchoProvider
        from dataknobs_llm.testing import text_response

        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([text_response("query one")])

        config = TransformerConfig(enabled=True, num_queries=1, suppress_thinking=False)
        transformer = QueryTransformer(config, provider=provider)

        await transformer.transform("test question")
        last_call = provider.get_last_call()
        assert last_call is not None
        # Should have no config_overrides or None
        overrides = last_call.get("config_overrides")
        assert overrides is None

    def test_suppress_thinking_default_false(self) -> None:
        """suppress_thinking defaults to False."""
        config = TransformerConfig()
        assert config.suppress_thinking is False


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

    def test_prefers_raw_content_over_augmented(self):
        """Messages with metadata.raw_content use that instead of content."""
        expander = ContextualExpander()
        history = [
            {
                "role": "user",
                "content": "<knowledge_base>kb chunks here</knowledge_base>\nTell me about neural networks",
                "metadata": {"raw_content": "Tell me about neural networks"},
            },
        ]

        result = expander.expand("More details", history)

        # Should extract keywords from raw_content, not augmented content
        assert "neural" in result.lower() or "networks" in result.lower()
        assert "knowledge_base" not in result.lower()

    def test_falls_back_to_content_without_raw_content(self):
        """Without metadata.raw_content, uses content as before."""
        expander = ContextualExpander()
        history = [
            {"role": "user", "content": "Tell me about quantum computing"},
        ]

        result = expander.expand("More details", history)

        assert "quantum" in result.lower()


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
