#!/usr/bin/env python3
"""
LLM Conversation System Example.

This example demonstrates an FSM-based LLM conversation system with:
- Multi-stage conversation flow (analyze, respond, refine)
- Template selection based on input characteristics
- Context management and conversation history
- Error recovery and fallback mechanisms
- Integration with multiple LLM providers
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.llm.utils import render_conditional_template, MessageBuilder
from dataknobs_fsm.llm.base import LLMConfig
from dataknobs_fsm.llm.providers import create_llm_provider


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConversationIntent(Enum):
    """Types of conversation intents."""
    QUESTION = "question"
    COMMAND = "command"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"
    GREETING = "greeting"
    UNKNOWN = "unknown"


@dataclass
class ConversationContext:
    """Maintains conversation context."""
    history: List[Dict[str, str]]
    current_topic: Optional[str] = None
    user_preferences: Dict[str, Any] = None

    def __post_init__(self):
        if self.user_preferences is None:
            self.user_preferences = {}


# Template library with conditional sections
RESPONSE_TEMPLATES = {
    "question": """
I'll help you with your question about {{topic}}.

((Based on the context: {{context}}

))Here's what I found:
{{response}}

((Additional information:
{{additional_info}}

))((Would you like more details about {{follow_up_topic}}?))
""",

    "command": """
I'll execute that {{action}} for you.

((Processing: {{details}}

))Result:
{{result}}

((Status: {{status}}
))((Next steps: {{next_steps}}))
""",

    "clarification": """
I need some clarification to better help you.

((Regarding: {{topic}}

)){{question}}

((Options:
{{options}}))
""",

    "greeting": """
{{greeting_text}}

(({{personal_note}}

))How can I assist you today?((

Some things I can help with:
{{suggestions}}))
""",

    "error": """
I encountered an issue while processing your request.

((Error: {{error_type}}

)){{fallback_response}}

((Please try: {{suggestions}}))
"""
}


class IntentAnalyzer:
    """Analyzes user input to determine intent and extract entities."""

    @staticmethod
    async def analyze(input_text: str, context: ConversationContext) -> Dict[str, Any]:
        """
        Analyze user input to determine intent and extract key information.

        Args:
            input_text: User's input text
            context: Current conversation context

        Returns:
            Analysis results with intent and entities
        """
        # Simple rule-based intent detection (in production, use NLP/LLM)
        input_lower = input_text.lower()

        # Detect intent
        if any(word in input_lower for word in ['?', 'what', 'how', 'why', 'when', 'where', 'who']):
            intent = ConversationIntent.QUESTION
        elif any(word in input_lower for word in ['do', 'execute', 'run', 'create', 'delete', 'update']):
            intent = ConversationIntent.COMMAND
        elif any(word in input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            intent = ConversationIntent.GREETING
        elif any(word in input_lower for word in ['unclear', 'confused', 'don\'t understand']):
            intent = ConversationIntent.CLARIFICATION
        elif context.history and len(input_text.split()) < 5:
            intent = ConversationIntent.CLARIFICATION
        else:
            intent = ConversationIntent.UNKNOWN

        # Extract topic (simple keyword extraction)
        topics = []
        keywords = ['fsm', 'state', 'machine', 'workflow', 'llm', 'model', 'data', 'process']
        for keyword in keywords:
            if keyword in input_lower:
                topics.append(keyword)

        # Build analysis result
        return {
            'intent': intent.value,
            'topics': topics,
            'input_length': len(input_text),
            'has_context': len(context.history) > 0,
            'confidence': 0.8 if intent != ConversationIntent.UNKNOWN else 0.3
        }


class ResponseGenerator:
    """Generates responses using LLM or templates."""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config
        self.llm_provider = None

    async def initialize(self):
        """Initialize LLM provider if configured."""
        if self.llm_config:
            try:
                self.llm_provider = create_llm_provider(self.llm_config, is_async=True)
                await self.llm_provider.initialize()
                logger.info(f"Initialized LLM provider: {self.llm_config.provider}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM provider: {e}. Using fallback mode.")
                self.llm_provider = None

    async def generate(
        self,
        intent: str,
        input_text: str,
        analysis: Dict[str, Any],
        context: ConversationContext
    ) -> str:
        """
        Generate response based on intent and context.

        Args:
            intent: Detected intent
            input_text: Original user input
            analysis: Analysis results
            context: Conversation context

        Returns:
            Generated response
        """
        # Try LLM generation first if available
        if self.llm_provider:
            try:
                response = await self._generate_with_llm(intent, input_text, analysis, context)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}. Using template fallback.")

        # Fallback to template-based generation
        return self._generate_with_template(intent, input_text, analysis, context)

    async def _generate_with_llm(
        self,
        intent: str,
        input_text: str,
        analysis: Dict[str, Any],
        context: ConversationContext
    ) -> Optional[str]:
        """Generate response using LLM."""
        if not self.llm_provider:
            return None

        # Build conversation history
        builder = MessageBuilder()

        # System prompt
        builder.system("""You are a helpful assistant in an FSM-based conversation system.
Respond appropriately based on the user's intent and maintain context from previous messages.
Be concise and helpful.""")

        # Add conversation history
        for msg in context.history[-5:]:  # Last 5 messages
            if msg['role'] == 'user':
                builder.user(msg['content'])
            else:
                builder.assistant(msg['content'])

        # Add current message with context
        prompt = f"Intent: {intent}\nTopics: {', '.join(analysis.get('topics', []))}\n\nUser: {input_text}"
        builder.user(prompt)

        # Generate response
        messages = builder.build()
        response = await self.llm_provider.complete(messages)

        return response.content

    def _generate_with_template(
        self,
        intent: str,
        input_text: str,
        analysis: Dict[str, Any],
        context: ConversationContext
    ) -> str:
        """Generate response using templates."""
        # Get appropriate template
        template = RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES['error'])

        # Prepare template parameters based on intent
        params = self._prepare_template_params(intent, input_text, analysis, context)

        # Render template with conditional sections
        return render_conditional_template(template, params)

    def _prepare_template_params(
        self,
        intent: str,
        input_text: str,
        analysis: Dict[str, Any],
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Prepare parameters for template rendering."""
        params = {}

        if intent == "question":
            topics = analysis.get('topics', [])
            params['topic'] = topics[0] if topics else 'your question'
            params['response'] = f"Based on my analysis, here's information about {params['topic']}..."

            # Add context if available
            if context.history:
                params['context'] = f"our previous discussion about {context.current_topic or 'this topic'}"

            # Add follow-up suggestion for longer conversations
            if len(context.history) > 2:
                params['follow_up_topic'] = 'implementation details'

        elif intent == "command":
            params['action'] = 'command'
            params['result'] = 'Command executed successfully'
            params['status'] = 'Complete'

            # Add details for complex commands
            if len(input_text) > 50:
                params['details'] = 'Analyzing command structure...'
                params['next_steps'] = 'You can now proceed with the next operation'

        elif intent == "clarification":
            topics = analysis.get('topics', [])
            params['topic'] = topics[0] if topics else 'your request'
            params['question'] = 'Could you provide more details about what you need?'

            # Add options if we have context
            if topics:
                params['options'] = '\n'.join([f"- More about {t}" for t in topics])

        elif intent == "greeting":
            params['greeting_text'] = 'Hello! Welcome to the FSM conversation system.'

            # Add personalization if we have history
            if context.history:
                params['personal_note'] = "It's good to continue our conversation."
                params['suggestions'] = "- Continue our previous discussion\n- Start a new topic\n- Ask about FSM capabilities"
            else:
                params['suggestions'] = "- Ask questions about FSM\n- Request workflow execution\n- Learn about features"

        else:  # error/unknown
            params['error_type'] = 'Intent unclear'
            params['fallback_response'] = "I'm not sure I understood your request correctly."
            params['suggestions'] = "rephrasing your question or providing more context"

        return params


class ConversationRefinement:
    """Refines responses based on feedback and context."""

    @staticmethod
    async def refine(
        response: str,
        analysis: Dict[str, Any],
        context: ConversationContext
    ) -> str:
        """
        Refine response based on context and quality checks.

        Args:
            response: Initial response
            analysis: Analysis results
            context: Conversation context

        Returns:
            Refined response
        """
        refined = response

        # Check response length and adjust
        if len(response) > 500 and analysis.get('input_length', 0) < 50:
            # User asked a short question, response too long
            lines = response.split('\n')
            if len(lines) > 5:
                # Keep first 3 lines and add truncation notice
                refined = '\n'.join(lines[:3]) + '\n\n[Response shortened for clarity. Ask for more details if needed.]'

        # Add context reference if continuing conversation
        if context.history and context.current_topic:
            if context.current_topic not in response.lower():
                refined = f"Continuing our discussion about {context.current_topic}:\n\n{refined}"

        # Ensure response ends properly
        if not refined.rstrip().endswith(('.', '?', '!', ']')):
            refined = refined.rstrip() + '.'

        return refined


class LLMConversationFSM:
    """FSM-based LLM conversation system."""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """
        Initialize conversation FSM.

        Args:
            llm_config: Optional LLM configuration
        """
        self.context = ConversationContext(history=[])
        self.analyzer = IntentAnalyzer()
        self.generator = ResponseGenerator(llm_config)
        self.refiner = ConversationRefinement()
        self.fsm = self._create_fsm()  # Create FSM after other components are initialized

    def _sync_wrapper(self, async_func):
        """Wrap an async function to be callable in sync context."""
        def wrapper(data, context=None):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, create a task
                task = asyncio.create_task(async_func(data))
                # For now, we'll need to block - this is not ideal
                # In production, the FSM should support async handlers natively
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_func(data))
                    return future.result()
            else:
                # No loop running, we can run directly
                return asyncio.run(async_func(data))
        return wrapper

    def _create_fsm(self) -> SimpleFSM:
        """Create the conversation FSM."""
        config = {
            'name': 'LLM_Conversation',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'analyze_input', 'transform': 'analyze_input'},
                {'name': 'select_template', 'transform': 'select_template'},
                {'name': 'generate_response', 'transform': 'generate_response'},
                {'name': 'refine_response', 'transform': 'refine_response'},
                {'name': 'update_context', 'transform': 'update_context'},
                {'name': 'handle_error', 'transform': 'handle_error'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {'from': 'start', 'to': 'analyze_input', 'name': 'begin'},
                {'from': 'analyze_input', 'to': 'select_template', 'name': 'analysis_complete',
                 'pre_test': 'check_analysis_success'},
                {'from': 'analyze_input', 'to': 'handle_error', 'name': 'analysis_failed',
                 'pre_test': 'check_analysis_failure'},
                {'from': 'select_template', 'to': 'generate_response', 'name': 'template_selected'},
                {'from': 'generate_response', 'to': 'refine_response', 'name': 'response_generated',
                 'pre_test': 'check_generation_success'},
                {'from': 'generate_response', 'to': 'handle_error', 'name': 'generation_failed',
                 'pre_test': 'check_generation_failure'},
                {'from': 'refine_response', 'to': 'update_context', 'name': 'response_refined'},
                {'from': 'update_context', 'to': 'end', 'name': 'context_updated'},
                {'from': 'handle_error', 'to': 'end', 'name': 'error_handled'}
            ]
        }

        # Define custom functions to register
        custom_functions = {
            'analyze_input': self._analyze_input,
            'select_template': self._select_template,
            'generate_response': self._generate_response,
            'refine_response': self._refine_response,
            'update_context': self._update_context,
            'handle_error': self._handle_error,
            'check_analysis_success': lambda data, context=None: data.get('analysis', {}).get('confidence', 0) >= 0.3,
            'check_analysis_failure': lambda data, context=None: data.get('analysis', {}).get('confidence', 1) < 0.3,
            'check_generation_success': lambda data, context=None: 'response' in data and not data.get('error'),
            'check_generation_failure': lambda data, context=None: 'error' in data
        }

        # Create FSM with custom functions
        fsm = SimpleFSM(
            config,
            data_mode=DataHandlingMode.REFERENCE,
            custom_functions=custom_functions
        )
        return fsm

    async def _analyze_input(self, data: Dict[str, Any], context=None) -> Dict[str, Any]:
        """Analyze user input."""
        logger.info(f"_analyze_input called with data type: {type(data)}")

        # Handle different data types
        if hasattr(data, 'data'):
            # It's a namespace/wrapper, extract the actual data
            actual_data = data.data if isinstance(data.data, dict) else data
        else:
            actual_data = data

        input_text = actual_data.get('input', '')
        analysis = await self.analyzer.analyze(input_text, self.context)

        actual_data['analysis'] = analysis
        actual_data['intent'] = analysis['intent']

        logger.info(f"Analyzed input - Intent: {analysis['intent']}, Confidence: {analysis['confidence']}")
        return actual_data

    async def _select_template(self, data: Dict[str, Any], context=None) -> Dict[str, Any]:
        """Select response template based on intent."""
        # Handle different data types
        if hasattr(data, 'data'):
            actual_data = data.data if isinstance(data.data, dict) else data
        else:
            actual_data = data

        intent = actual_data.get('intent', 'unknown')

        # Select template
        template_name = intent if intent in RESPONSE_TEMPLATES else 'error'
        actual_data['template'] = template_name

        logger.info(f"Selected template: {template_name}")
        return actual_data

    async def _generate_response(self, data: Dict[str, Any], context=None) -> Dict[str, Any]:
        """Generate response using selected approach."""
        # Handle different data types
        if hasattr(data, 'data'):
            actual_data = data.data if isinstance(data.data, dict) else data
        else:
            actual_data = data

        try:
            response = await self.generator.generate(
                actual_data['intent'],
                actual_data['input'],
                actual_data['analysis'],
                self.context
            )

            actual_data['response'] = response
            logger.info("Response generated successfully")

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            actual_data['error'] = str(e)

        return actual_data

    async def _refine_response(self, data: Dict[str, Any], context=None) -> Dict[str, Any]:
        """Refine the generated response."""
        # Handle different data types
        if hasattr(data, 'data'):
            actual_data = data.data if isinstance(data.data, dict) else data
        else:
            actual_data = data

        response = actual_data.get('response', '')
        refined = await self.refiner.refine(response, actual_data['analysis'], self.context)

        actual_data['response'] = refined

        logger.info("Response refined")
        return actual_data

    async def _update_context(self, data: Dict[str, Any], context=None) -> Dict[str, Any]:
        """Update conversation context."""
        # Handle different data types
        if hasattr(data, 'data'):
            actual_data = data.data if isinstance(data.data, dict) else data
        else:
            actual_data = data

        # Add to history
        self.context.history.append({
            'role': 'user',
            'content': actual_data['input']
        })
        self.context.history.append({
            'role': 'assistant',
            'content': actual_data['response']
        })

        # Update current topic
        topics = actual_data['analysis'].get('topics', [])
        if topics:
            self.context.current_topic = topics[0]

        # Keep history size manageable
        if len(self.context.history) > 20:
            self.context.history = self.context.history[-20:]

        logger.info("Context updated")
        return actual_data

    async def _handle_error(self, data: Dict[str, Any], context=None) -> Dict[str, Any]:
        """Handle errors with fallback response."""
        # Handle different data types
        if hasattr(data, 'data'):
            actual_data = data.data if isinstance(data.data, dict) else data
        else:
            actual_data = data

        error = actual_data.get('error', 'Unknown error')

        # Generate error response using template
        params = {
            'error_type': 'Processing error',
            'fallback_response': "I apologize, but I couldn't process your request properly.",
            'suggestions': 'rephrasing your message or trying again'
        }

        actual_data['response'] = render_conditional_template(RESPONSE_TEMPLATES['error'], params)

        logger.warning(f"Handled error: {error}")
        return actual_data

    async def initialize(self):
        """Initialize the conversation system."""
        await self.generator.initialize()
        logger.info("Conversation system initialized")

    async def process_message(self, message: str) -> str:
        """
        Process a user message and return response.

        Args:
            message: User input message

        Returns:
            System response
        """
        # Process through FSM using async method
        try:
            result = await self.fsm.process_async({'input': message})
            logger.debug(f"FSM result: {result}")

            # Handle different result formats
            if isinstance(result, dict):
                # Check if data is nested
                data = result.get('data', result)
                response = data.get('response')
                if response:
                    return response

            return 'I apologize, but I encountered an error processing your message.'
        except Exception as e:
            logger.error(f"FSM processing error: {e}")
            return 'I apologize, but I encountered an error processing your message.'

    def reset_context(self):
        """Reset conversation context."""
        self.context = ConversationContext(history=[])
        logger.info("Context reset")


async def main():
    """Main function demonstrating the conversation system."""
    print("=" * 60)
    print("LLM Conversation System Example")
    print("=" * 60)

    # Create LLM configuration (optional - will use templates if not available)
    llm_config = None
    try:
        # Try to use OpenAI if available
        llm_config = LLMConfig(
            provider='openai',
            model='gpt-3.5-turbo',
            temperature=0.7,
            max_tokens=150
        )
    except Exception:
        print("Note: LLM provider not configured. Using template-based responses.\n")

    # Create conversation system
    conversation = LLMConversationFSM(llm_config)
    await conversation.initialize()

    # Example conversations
    test_messages = [
        "Hello! What is an FSM?",
        "Can you explain how states work?",
        "Execute a workflow for data processing",
        "That's unclear, can you clarify?",
        "Thanks for the help!"
    ]

    print("\nStarting example conversation...\n")

    for message in test_messages:
        print(f"User: {message}")
        response = await conversation.process_message(message)
        print(f"Assistant: {response}")
        print("-" * 40)

        # Small delay for readability
        await asyncio.sleep(0.5)

    # Show context summary
    print("\nConversation Summary:")
    print(f"- Messages exchanged: {len(conversation.context.history)}")
    print(f"- Current topic: {conversation.context.current_topic or 'General'}")
    print(f"- Intent distribution: {len(test_messages)} messages processed")

    # Interactive mode (optional)
    print("\n" + "=" * 60)
    print("Interactive Mode (type 'quit' to exit, 'reset' to clear context)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'reset':
                conversation.reset_context()
                print("Context reset. Starting fresh conversation.")
                continue
            elif not user_input:
                continue

            response = await conversation.process_message(user_input)
            print(f"Assistant: {response}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())