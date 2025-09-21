# LLM Conversation System Example

This example demonstrates how to build an FSM-based LLM conversation system with sophisticated conversation flow management.

## Overview

The example showcases:

- **Multi-stage conversation flow** (analyze, respond, refine)
- **Template selection** based on input characteristics
- **Context management** and conversation history
- **Error recovery** and fallback mechanisms
- **Integration** with multiple LLM providers

## Source Code

The complete example is available at: [`packages/fsm/examples/llm_conversation.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/fsm/examples/llm_conversation.py)

## Implementation Details

### Conversation Flow

The system implements a sophisticated multi-stage conversation flow:

```python
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

def create_conversation_fsm() -> SimpleFSM:
    """Create the conversation FSM."""
    config = {
        "name": "llm_conversation_system",
        "states": [
            {"name": "start", "initial": True},
            {"name": "analyze_input"},
            {"name": "select_template"},
            {"name": "generate_response"},
            {"name": "refine_response"},
            {"name": "update_context"},
            {"name": "complete", "terminal": True},
            {"name": "error", "terminal": True}
        ],
        "arcs": [
            {"from": "start", "to": "analyze_input"},
            {"from": "analyze_input", "to": "select_template"},
            {"from": "select_template", "to": "generate_response"},
            {"from": "generate_response", "to": "refine_response"},
            {"from": "refine_response", "to": "update_context"},
            {"from": "update_context", "to": "complete"}
        ]
    }

    return SimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={
            "analyze_user_input": analyze_user_input,
            "select_response_template": select_response_template,
            "generate_llm_response": generate_llm_response,
            "refine_and_validate": refine_and_validate,
            "update_conversation_context": update_conversation_context
        }
    )
```

### Key Components

#### Intent Recognition

```python
class ConversationIntent(Enum):
    """Types of conversation intents."""
    QUESTION = "question"
    COMMAND = "command"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"
    GREETING = "greeting"
    UNKNOWN = "unknown"

def analyze_user_input(state) -> Dict[str, Any]:
    """Analyze user input to determine intent and extract entities."""
    data = state.data.copy()
    user_input = data['user_input']

    # Intent detection logic
    if any(word in user_input.lower() for word in ['what', 'how', 'why', 'when']):
        intent = ConversationIntent.QUESTION
    elif any(word in user_input.lower() for word in ['do', 'create', 'make']):
        intent = ConversationIntent.COMMAND
    elif any(word in user_input.lower() for word in ['hello', 'hi', 'hey']):
        intent = ConversationIntent.GREETING
    else:
        intent = ConversationIntent.UNKNOWN

    data['intent'] = intent.value
    return data
```

#### Context Management

```python
@dataclass
class ConversationContext:
    """Maintains conversation context."""
    history: List[Dict[str, str]]
    current_topic: Optional[str] = None
    user_preferences: Dict[str, Any] = None

    def add_exchange(self, user_input: str, system_response: str):
        """Add a conversation exchange to history."""
        self.history.append({
            "user": user_input,
            "assistant": system_response
        })

    def get_recent_context(self, n: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation exchanges."""
        return self.history[-n:] if len(self.history) >= n else self.history
```

#### LLM Integration

```python
from dataknobs_fsm.llm.providers import create_llm_provider
from dataknobs_fsm.llm.base import LLMConfig

def generate_llm_response(state) -> Dict[str, Any]:
    """Generate response using LLM provider."""
    data = state.data.copy()

    # Configure LLM
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=500
    )

    provider = create_llm_provider(config)

    # Build prompt with context
    prompt = build_prompt_with_context(
        template=data['selected_template'],
        user_input=data['user_input'],
        context=data['conversation_context']
    )

    # Generate response
    response = provider.generate(prompt)
    data['llm_response'] = response.content

    return data
```

## Usage Example

```python
import asyncio

# Initialize conversation
context = ConversationContext(history=[])

# Create FSM
fsm = create_conversation_fsm()

# Process user input
result = fsm.process({
    "user_input": "How does the FSM pattern help with conversation management?",
    "conversation_context": context
})

# Get response
response = result['data']['final_response']
print(f"Assistant: {response}")

# Update context
context.add_exchange(
    user_input="How does the FSM pattern help?",
    system_response=response
)
```

## Advanced Features

### Template System

The example includes a sophisticated template selection system:

```python
REPONSE_TEMPLATES = {
    ConversationIntent.QUESTION: [
        "Based on my understanding, {answer}. {elaboration}",
        "To answer your question: {answer}. {context_reference}"
    ],
    ConversationIntent.COMMAND: [
        "I'll help you with that. {action_description}. {next_steps}",
        "Processing your request: {action_status}. {result}"
    ],
    ConversationIntent.GREETING: [
        "Hello! {greeting_response}. How can I assist you today?",
        "Hi there! {greeting_response}. What would you like to know?"
    ]
}
```

### Error Recovery

Robust error handling with fallback responses:

```python
def handle_llm_error(state) -> Dict[str, Any]:
    """Handle LLM generation errors with fallback."""
    data = state.data.copy()

    # Use fallback response
    fallback_responses = [
        "I apologize, but I'm having trouble understanding. Could you rephrase?",
        "I need a moment to process that. Could you provide more context?",
        "I'm not quite sure how to respond to that. Can you be more specific?"
    ]

    data['final_response'] = random.choice(fallback_responses)
    data['error_handled'] = True

    return data
```

## Benefits

1. **Structured Conversation Flow**: FSM ensures consistent conversation patterns
2. **Context Preservation**: Maintains conversation history and user preferences
3. **Error Resilience**: Graceful handling of LLM failures
4. **Provider Flexibility**: Easy switching between LLM providers
5. **Template-based Responses**: Consistent response formatting