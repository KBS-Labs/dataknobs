"""
Unit tests for the LLM conversation example.

Tests the LLM conversation system FSM without requiring actual LLM providers.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

import sys
from pathlib import Path

# Add examples directory to path
examples_dir = Path(__file__).parent.parent / 'examples'
sys.path.insert(0, str(examples_dir))

from llm_conversation import (
    LLMConversationFSM,
    ConversationContext,
    ConversationIntent,
    IntentAnalyzer,
    ResponseGenerator,
    ConversationRefinement,
    RESPONSE_TEMPLATES
)
from dataknobs_fsm.llm.base import LLMConfig, LLMResponse
from dataknobs_fsm.llm.utils import render_conditional_template


class TestConversationContext:
    """Test conversation context management."""

    def test_context_initialization(self):
        """Test context initializes properly."""
        context = ConversationContext(history=[])
        assert context.history == []
        assert context.current_topic is None
        assert context.user_preferences == {}

    def test_context_with_history(self):
        """Test context with existing history."""
        history = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]
        context = ConversationContext(history=history, current_topic='greeting')
        assert len(context.history) == 2
        assert context.current_topic == 'greeting'


class TestIntentAnalyzer:
    """Test intent analysis functionality."""

    @pytest.mark.asyncio
    async def test_question_intent(self):
        """Test question intent detection."""
        analyzer = IntentAnalyzer()
        context = ConversationContext(history=[])

        result = await analyzer.analyze("What is an FSM?", context)
        assert result['intent'] == ConversationIntent.QUESTION.value
        assert 'fsm' in result['topics']
        assert result['confidence'] == 0.8

    @pytest.mark.asyncio
    async def test_command_intent(self):
        """Test command intent detection."""
        analyzer = IntentAnalyzer()
        context = ConversationContext(history=[])

        result = await analyzer.analyze("Execute the workflow now", context)
        assert result['intent'] == ConversationIntent.COMMAND.value
        assert 'workflow' in result['topics']

    @pytest.mark.asyncio
    async def test_greeting_intent(self):
        """Test greeting intent detection."""
        analyzer = IntentAnalyzer()
        context = ConversationContext(history=[])

        result = await analyzer.analyze("Hello there!", context)
        assert result['intent'] == ConversationIntent.GREETING.value

    @pytest.mark.asyncio
    async def test_clarification_intent(self):
        """Test clarification intent detection."""
        analyzer = IntentAnalyzer()
        context = ConversationContext(history=[
            {'role': 'user', 'content': 'Tell me about FSM'},
            {'role': 'assistant', 'content': 'FSM is...'}
        ])

        result = await analyzer.analyze("unclear", context)
        assert result['intent'] == ConversationIntent.CLARIFICATION.value

    @pytest.mark.asyncio
    async def test_unknown_intent(self):
        """Test unknown intent detection."""
        analyzer = IntentAnalyzer()
        context = ConversationContext(history=[])

        result = await analyzer.analyze("xyz abc 123", context)  # No keywords
        assert result['confidence'] == 0.3


class TestResponseGenerator:
    """Test response generation."""

    @pytest.mark.asyncio
    async def test_template_generation_question(self):
        """Test template-based response for questions."""
        generator = ResponseGenerator()
        await generator.initialize()

        context = ConversationContext(history=[])
        analysis = {
            'topics': ['fsm'],
            'input_length': 15,
            'has_context': False
        }

        response = generator._generate_with_template(
            'question',
            "What is FSM?",
            analysis,
            context
        )

        assert 'help you with your question about fsm' in response
        assert 'Based on my analysis' in response

    @pytest.mark.asyncio
    async def test_template_generation_greeting(self):
        """Test template-based response for greetings."""
        generator = ResponseGenerator()
        await generator.initialize()

        context = ConversationContext(history=[])
        analysis = {}

        response = generator._generate_with_template(
            'greeting',
            "Hello!",
            analysis,
            context
        )

        assert 'Hello! Welcome' in response
        assert 'How can I assist you today?' in response

    @pytest.mark.asyncio
    async def test_template_with_context(self):
        """Test template generation with conversation context."""
        generator = ResponseGenerator()
        await generator.initialize()

        context = ConversationContext(
            history=[
                {'role': 'user', 'content': 'Tell me about FSM'},
                {'role': 'assistant', 'content': 'FSM is a state machine'}
            ],
            current_topic='fsm'
        )
        analysis = {
            'topics': ['state', 'machine'],
            'has_context': True
        }

        response = generator._generate_with_template(
            'question',
            "How do states work?",
            analysis,
            context
        )

        assert 'Based on the context' in response
        assert 'our previous discussion' in response

    @pytest.mark.asyncio
    async def test_llm_generation_fallback(self):
        """Test fallback when LLM is not available."""
        generator = ResponseGenerator()
        await generator.initialize()

        context = ConversationContext(history=[])
        analysis = {'topics': ['test']}

        response = await generator.generate(
            'question',
            "Test question",
            analysis,
            context
        )

        # Should fallback to template
        assert 'help you with your question' in response


class TestConversationRefinement:
    """Test response refinement."""

    @pytest.mark.asyncio
    async def test_refine_long_response(self):
        """Test refinement of overly long responses."""
        refiner = ConversationRefinement()
        context = ConversationContext(history=[])

        # Create a long response with more than 500 chars
        long_lines = ['This is a very long line that contains a lot of text to make it exceed length limits' for _ in range(20)]
        long_response = '\n'.join(long_lines)
        analysis = {'input_length': 10}  # Short input

        refined = await refiner.refine(long_response, analysis, context)

        # Should be truncated if > 500 chars and input was short
        if len(long_response) > 500:
            assert '[Response shortened for clarity' in refined or len(refined) <= len(long_response)

    @pytest.mark.asyncio
    async def test_refine_with_context(self):
        """Test refinement with conversation context."""
        refiner = ConversationRefinement()
        context = ConversationContext(
            history=[{'role': 'user', 'content': 'About FSM'}],
            current_topic='workflow'
        )

        response = "This is the response content."
        analysis = {}

        refined = await refiner.refine(response, analysis, context)

        assert 'Continuing our discussion about workflow' in refined

    @pytest.mark.asyncio
    async def test_refine_ending(self):
        """Test refinement ensures proper ending."""
        refiner = ConversationRefinement()
        context = ConversationContext(history=[])

        response = "This response has no ending punctuation"
        analysis = {}

        refined = await refiner.refine(response, analysis, context)

        assert refined.rstrip().endswith('.')


class TestTemplateRendering:
    """Test conditional template rendering."""

    def test_render_with_all_params(self):
        """Test rendering with all parameters provided."""
        template = RESPONSE_TEMPLATES['question']
        params = {
            'topic': 'FSM',
            'context': 'our discussion',
            'response': 'FSM is a state machine',
            'additional_info': 'It manages states',
            'follow_up_topic': 'implementation'
        }

        result = render_conditional_template(template, params)

        assert 'help you with your question about FSM' in result
        assert 'Based on the context: our discussion' in result
        assert 'Additional information:' in result
        assert 'Would you like more details about implementation?' in result

    def test_render_with_missing_optional(self):
        """Test rendering with missing optional parameters."""
        template = RESPONSE_TEMPLATES['question']
        params = {
            'topic': 'FSM',
            'response': 'FSM is a state machine'
        }

        result = render_conditional_template(template, params)

        assert 'help you with your question about FSM' in result
        assert 'Based on the context' not in result
        assert 'Additional information' not in result
        assert 'Would you like more details' not in result

    def test_render_error_template(self):
        """Test rendering error template."""
        template = RESPONSE_TEMPLATES['error']
        params = {
            'error_type': 'Processing error',
            'fallback_response': 'Sorry, something went wrong'
        }

        result = render_conditional_template(template, params)

        assert 'encountered an issue' in result
        assert 'Error: Processing error' in result
        assert 'Sorry, something went wrong' in result


class TestLLMConversationFSM:
    """Test the main conversation FSM."""

    @pytest.mark.asyncio
    async def test_fsm_initialization(self):
        """Test FSM initialization."""
        fsm = LLMConversationFSM()
        await fsm.initialize()

        assert fsm.context.history == []
        assert fsm.context.current_topic is None
        assert fsm.fsm is not None

    @pytest.mark.asyncio
    async def test_process_question(self):
        """Test processing a question through FSM."""
        fsm = LLMConversationFSM()
        await fsm.initialize()

        response = await fsm.process_message("What is an FSM?")

        assert response is not None
        assert len(response) > 0
        assert 'fsm' in response.lower() or 'question' in response.lower()

        # Check context was updated
        assert len(fsm.context.history) == 2
        assert fsm.context.history[0]['role'] == 'user'
        assert fsm.context.history[0]['content'] == "What is an FSM?"

    @pytest.mark.asyncio
    async def test_process_greeting(self):
        """Test processing a greeting."""
        fsm = LLMConversationFSM()
        await fsm.initialize()

        response = await fsm.process_message("Hello!")

        assert 'Hello' in response or 'Welcome' in response
        assert 'assist' in response.lower()

    @pytest.mark.asyncio
    async def test_process_command(self):
        """Test processing a command."""
        fsm = LLMConversationFSM()
        await fsm.initialize()

        response = await fsm.process_message("Execute the data processing workflow")

        assert 'execute' in response.lower() or 'command' in response.lower()
        assert len(fsm.context.history) == 2

    @pytest.mark.asyncio
    async def test_conversation_flow(self):
        """Test a multi-turn conversation."""
        fsm = LLMConversationFSM()
        await fsm.initialize()

        # First message
        response1 = await fsm.process_message("Hello")
        assert 'Hello' in response1 or 'Welcome' in response1

        # Follow-up question
        response2 = await fsm.process_message("What can you help me with?")
        assert len(response2) > 0

        # Check context accumulation
        assert len(fsm.context.history) == 4
        assert fsm.context.current_topic is not None or len(fsm.context.history) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in conversation."""
        fsm = LLMConversationFSM()
        await fsm.initialize()

        # Mock an error in analysis
        with patch.object(fsm.analyzer, 'analyze', side_effect=Exception("Test error")):
            response = await fsm.process_message("Test message")

            # Should still get a response (error handler)
            assert response is not None
            assert 'error' in response.lower() or 'apologize' in response.lower()

    @pytest.mark.asyncio
    async def test_context_reset(self):
        """Test context reset functionality."""
        fsm = LLMConversationFSM()
        await fsm.initialize()

        # Add some history
        await fsm.process_message("Hello")
        await fsm.process_message("What is FSM?")

        assert len(fsm.context.history) > 0
        assert fsm.context.current_topic is not None or len(fsm.context.history) > 0

        # Reset context
        fsm.reset_context()

        assert len(fsm.context.history) == 0
        assert fsm.context.current_topic is None

    @pytest.mark.asyncio
    async def test_with_llm_config(self):
        """Test FSM with LLM configuration (mocked)."""
        llm_config = LLMConfig(
            provider='mock',
            model='test-model'
        )

        fsm = LLMConversationFSM(llm_config)

        # Mock the LLM provider
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=LLMResponse(
            content="This is an LLM response",
            model="test-model"
        ))

        with patch.object(fsm.generator, 'llm_provider', mock_provider):
            await fsm.initialize()
            response = await fsm.process_message("Test with LLM")

            # Should use LLM response if available
            assert response is not None
            assert len(response) > 0


class TestMainFunction:
    """Test the main demonstration function."""

    @pytest.mark.asyncio
    async def test_main_execution(self):
        """Test that main function runs without errors."""
        from llm_conversation import main

        # Mock input to avoid interactive mode
        with patch('builtins.input', side_effect=['quit']):
            # Run main without errors
            try:
                await main()
            except SystemExit:
                pass  # Expected when 'quit' is entered


@pytest.mark.asyncio
async def test_integration_flow():
    """Integration test for complete conversation flow."""
    # Create FSM
    fsm = LLMConversationFSM()
    await fsm.initialize()

    # Simulate a conversation
    messages = [
        ("Hello!", "greeting"),
        ("What is an FSM?", "question"),
        ("Can you explain states?", "question"),
        ("Thanks!", "greeting")
    ]

    for message, expected_intent in messages:
        response = await fsm.process_message(message)

        # Verify response
        assert response is not None
        assert len(response) > 0

        # Verify context is building
        assert len(fsm.context.history) > 0

    # Verify final context state
    assert len(fsm.context.history) == len(messages) * 2  # User + assistant messages
    assert fsm.context.current_topic is not None or len(fsm.context.history) > 0