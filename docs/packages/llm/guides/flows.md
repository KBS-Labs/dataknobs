# FSM-Based Conversation Flows

Build complex conversation workflows using finite state machines.

## Overview

The conversation flow system provides:

- **State Machine Integration**: Built on dataknobs-fsm
- **Transition Conditions**: Keyword, regex, LLM classifier, sentiment
- **Loop Detection**: Automatic detection and prevention
- **Resource Integration**: Seamless FSM resource integration
- **Pre-built Patterns**: Customer support, sales qualification, etc.

## Quick Start

```python
from dataknobs_llm.conversations.flow import (
    ConversationFlow,
    FlowState,
    KeywordCondition,
    RegexCondition
)

# Define flow
flow = ConversationFlow(
    name="customer_support",
    initial_state="greeting",
    states={
        "greeting": FlowState(
            prompt="support_greeting",
            transitions={
                "help": KeywordCondition(["help", "issue"]),
                "info": KeywordCondition(["info", "question"])
            },
            next_states={"help": "collect_issue", "info": "provide_info"}
        ),
        "collect_issue": FlowState(
            prompt="ask_issue_details",
            transitions={
                "technical": KeywordCondition(["bug", "error"]),
                "billing": KeywordCondition(["payment", "charge"])
            },
            next_states={"technical": "tech_support", "billing": "billing_support"}
        ),
        # ... more states
    }
)

# Execute flow
manager = await ConversationManager.create(llm=llm, flow=flow)
await manager.execute_flow()
```

## Transition Conditions

### 1. Keyword Conditions

```python
from dataknobs_llm.conversations.flow import KeywordCondition

condition = KeywordCondition(
    keywords=["yes", "sure", "ok"],
    match_any=True  # Match any keyword
)
```

### 2. Regex Conditions

```python
from dataknobs_llm.conversations.flow import RegexCondition

condition = RegexCondition(
    pattern=r"\b\d{3}-\d{3}-\d{4}\b"  # Phone number
)
```

### 3. LLM Classifier Conditions

```python
from dataknobs_llm.conversations.flow import LLMClassifierCondition

condition = LLMClassifierCondition(
    llm=llm,
    classification_prompt="Classify sentiment as positive/negative/neutral: {{text}}",
    target_class="positive"
)
```

### 4. Sentiment Conditions

```python
from dataknobs_llm.conversations.flow import SentimentCondition

condition = SentimentCondition(
    target_sentiment="positive",
    threshold=0.7
)
```

## Loop Detection

```python
flow = ConversationFlow(
    name="support",
    initial_state="greeting",
    states=states,
    max_loops=3  # Prevent infinite loops
)
```

## Detailed Documentation

Complete flow documentation is available in the local package:

**Location**: `packages/llm/docs/` (FSM-related sections in USER_GUIDE.md)

## Common Patterns

### Customer Support Flow

```python
support_flow = ConversationFlow(
    name="customer_support",
    initial_state="greeting",
    states={
        "greeting": FlowState(
            prompt="support_greeting",
            transitions={
                "need_help": KeywordCondition(["help", "problem"]),
                "just_browsing": KeywordCondition(["browse", "look"])
            },
            next_states={"need_help": "collect_issue", "just_browsing": "end"}
        ),
        "collect_issue": FlowState(
            prompt="ask_issue_details",
            transitions={
                "technical": KeywordCondition(["bug", "error", "crash"]),
                "billing": KeywordCondition(["payment", "bill", "charge"]),
                "account": KeywordCondition(["account", "login", "password"])
            },
            next_states={
                "technical": "technical_support",
                "billing": "billing_support",
                "account": "account_support"
            }
        ),
        # ... more states
    }
)
```

### Sales Qualification Flow

```python
sales_flow = ConversationFlow(
    name="sales_qualification",
    initial_state="introduction",
    states={
        "introduction": FlowState(
            prompt="sales_intro",
            transitions={
                "interested": SentimentCondition("positive", threshold=0.6),
                "not_interested": SentimentCondition("negative", threshold=0.6)
            },
            next_states={"interested": "qualify_budget", "not_interested": "polite_close"}
        ),
        "qualify_budget": FlowState(
            prompt="ask_budget",
            transitions={
                "has_budget": KeywordCondition(["budget", "approved", "allocated"]),
                "no_budget": KeywordCondition(["no budget", "not approved"])
            },
            next_states={"has_budget": "schedule_demo", "no_budget": "nurture"}
        ),
        # ... more states
    }
)
```

## FSM Integration

The conversation flows are built on dataknobs-fsm, providing access to all FSM features:

```python
from dataknobs_fsm import StateMachine, State, Transition

# Convert flow to FSM
fsm = flow.to_fsm()

# Use FSM features
fsm.visualize()  # Generate state diagram
fsm.validate()   # Validate flow definition
```

## See Also

- [Conversation Management](conversations.md) - Core conversation features
- [FSM Package](../../fsm/index.md) - Underlying FSM framework
- [Examples](../examples/conversation-flows.md) - Flow examples
- [API Reference](../api/conversations.md) - Conversations API (includes flows)
