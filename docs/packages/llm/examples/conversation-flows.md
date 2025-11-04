# Conversation Flow Examples

FSM-based conversation workflows and patterns.

## Basic Flow

### Simple Two-State Flow

```python
from dataknobs_llm.conversations.flow import (
    ConversationFlow,
    FlowState,
    KeywordCondition
)
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage
)
from dataknobs_llm import create_llm_provider, LLMConfig
from dataknobs_data.backends import AsyncMemoryDatabase

# Define flow
flow = ConversationFlow(
    name="simple_greeting",
    initial_state="greeting",
    states={
        "greeting": FlowState(
            prompt="greeting",
            transitions={
                "continue": KeywordCondition(["yes", "sure", "ok"])
            },
            next_states={"continue": "help"}
        ),
        "help": FlowState(
            prompt="provide_help",
            transitions={},
            next_states={}
        )
    }
)

# Execute flow
config = LLMConfig(provider="openai", api_key="your-key")
llm = create_llm_provider(config)

db = AsyncMemoryDatabase()
storage = DataknobsConversationStorage(db)

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    flow=flow
)

await manager.execute_flow()
```

### Branching Flow

```python
flow = ConversationFlow(
    name="support_routing",
    initial_state="greeting",
    states={
        "greeting": FlowState(
            prompt="support_greeting",
            transitions={
                "technical": KeywordCondition(["bug", "error", "crash"]),
                "billing": KeywordCondition(["payment", "bill", "charge"]),
                "general": KeywordCondition(["question", "help", "info"])
            },
            next_states={
                "technical": "tech_support",
                "billing": "billing_support",
                "general": "general_help"
            }
        ),
        "tech_support": FlowState(
            prompt="tech_support_prompt",
            transitions={},
            next_states={}
        ),
        "billing_support": FlowState(
            prompt="billing_support_prompt",
            transitions={},
            next_states={}
        ),
        "general_help": FlowState(
            prompt="general_help_prompt",
            transitions={},
            next_states={}
        )
    }
)
```

## Customer Support Flow

### Complete Support Workflow

```python
from dataknobs_llm.conversations.flow import (
    ConversationFlow,
    FlowState,
    KeywordCondition,
    RegexCondition,
    SentimentCondition
)

support_flow = ConversationFlow(
    name="customer_support",
    initial_state="greeting",
    states={
        # Initial greeting
        "greeting": FlowState(
            prompt="support_greeting",
            transitions={
                "need_help": KeywordCondition([
                    "help", "problem", "issue", "question"
                ]),
                "complaint": SentimentCondition("negative", threshold=0.7),
                "just_browsing": KeywordCondition(["browse", "look", "info"])
            },
            next_states={
                "need_help": "classify_issue",
                "complaint": "escalate",
                "just_browsing": "provide_info"
            }
        ),

        # Classify the issue
        "classify_issue": FlowState(
            prompt="ask_issue_type",
            transitions={
                "technical": KeywordCondition([
                    "bug", "error", "crash", "not working", "broken"
                ]),
                "account": KeywordCondition([
                    "account", "login", "password", "access"
                ]),
                "billing": KeywordCondition([
                    "payment", "bill", "charge", "refund", "pricing"
                ]),
                "feature": KeywordCondition([
                    "feature", "how to", "tutorial", "guide"
                ])
            },
            next_states={
                "technical": "technical_support",
                "account": "account_support",
                "billing": "billing_support",
                "feature": "feature_help"
            }
        ),

        # Technical support path
        "technical_support": FlowState(
            prompt="tech_troubleshooting",
            transitions={
                "resolved": KeywordCondition(["fixed", "solved", "working"]),
                "escalate": KeywordCondition(["still", "not working", "persist"]),
            },
            next_states={
                "resolved": "satisfaction_check",
                "escalate": "escalate"
            }
        ),

        # Account support
        "account_support": FlowState(
            prompt="account_assistance",
            transitions={
                "resolved": KeywordCondition(["fixed", "solved", "access"]),
                "security": KeywordCondition(["hacked", "security", "suspicious"])
            },
            next_states={
                "resolved": "satisfaction_check",
                "security": "security_escalation"
            }
        ),

        # Billing support
        "billing_support": FlowState(
            prompt="billing_assistance",
            transitions={
                "refund": KeywordCondition(["refund", "money back"]),
                "clarified": KeywordCondition(["understand", "clear", "thanks"])
            },
            next_states={
                "refund": "process_refund",
                "clarified": "satisfaction_check"
            }
        ),

        # Feature help
        "feature_help": FlowState(
            prompt="feature_explanation",
            transitions={
                "understood": KeywordCondition(["got it", "understand", "clear"]),
                "more_help": KeywordCondition(["more", "else", "another"])
            },
            next_states={
                "understood": "satisfaction_check",
                "more_help": "classify_issue"
            }
        ),

        # Escalation
        "escalate": FlowState(
            prompt="escalate_to_human",
            transitions={},
            next_states={}
        ),

        # Security escalation
        "security_escalation": FlowState(
            prompt="security_protocol",
            transitions={},
            next_states={}
        ),

        # Process refund
        "process_refund": FlowState(
            prompt="refund_process",
            transitions={
                "accepted": KeywordCondition(["yes", "ok", "proceed"])
            },
            next_states={"accepted": "satisfaction_check"}
        ),

        # General info
        "provide_info": FlowState(
            prompt="general_information",
            transitions={
                "need_help": KeywordCondition(["help", "question"])
            },
            next_states={"need_help": "classify_issue"}
        ),

        # Final satisfaction check
        "satisfaction_check": FlowState(
            prompt="satisfaction_survey",
            transitions={},
            next_states={}
        )
    },
    max_loops=3  # Prevent infinite loops
)
```

## Sales Qualification Flow

### Lead Qualification

```python
from dataknobs_llm.conversations.flow import LLMClassifierCondition

sales_flow = ConversationFlow(
    name="sales_qualification",
    initial_state="introduction",
    states={
        # Introduction
        "introduction": FlowState(
            prompt="sales_intro",
            transitions={
                "interested": SentimentCondition("positive", threshold=0.6),
                "not_interested": SentimentCondition("negative", threshold=0.6),
                "neutral": KeywordCondition(["maybe", "not sure", "thinking"])
            },
            next_states={
                "interested": "qualify_budget",
                "not_interested": "polite_close",
                "neutral": "build_interest"
            }
        ),

        # Build interest
        "build_interest": FlowState(
            prompt="value_proposition",
            transitions={
                "interested": SentimentCondition("positive", threshold=0.5),
                "still_unsure": KeywordCondition(["unsure", "thinking"])
            },
            next_states={
                "interested": "qualify_budget",
                "still_unsure": "nurture"
            }
        ),

        # Qualify budget
        "qualify_budget": FlowState(
            prompt="ask_budget",
            transitions={
                "has_budget": KeywordCondition([
                    "budget", "allocated", "approved", "ready"
                ]),
                "no_budget": KeywordCondition([
                    "no budget", "not approved", "need approval"
                ]),
                "price_concern": KeywordCondition(["expensive", "cost", "price"])
            },
            next_states={
                "has_budget": "qualify_authority",
                "no_budget": "nurture",
                "price_concern": "address_pricing"
            }
        ),

        # Address pricing concerns
        "address_pricing": FlowState(
            prompt="pricing_justification",
            transitions={
                "convinced": SentimentCondition("positive", threshold=0.6),
                "still_expensive": KeywordCondition(["still", "too much"])
            },
            next_states={
                "convinced": "qualify_authority",
                "still_expensive": "nurture"
            }
        ),

        # Qualify authority
        "qualify_authority": FlowState(
            prompt="ask_decision_maker",
            transitions={
                "is_decision_maker": KeywordCondition([
                    "yes", "I am", "me", "I decide"
                ]),
                "not_decision_maker": KeywordCondition([
                    "not me", "manager", "team", "need approval"
                ])
            },
            next_states={
                "is_decision_maker": "qualify_need",
                "not_decision_maker": "involve_stakeholders"
            }
        ),

        # Qualify need
        "qualify_need": FlowState(
            prompt="assess_need",
            transitions={
                "urgent": KeywordCondition(["urgent", "asap", "soon", "now"]),
                "planned": KeywordCondition(["planning", "next quarter", "future"])
            },
            next_states={
                "urgent": "schedule_demo",
                "planned": "nurture_timeline"
            }
        ),

        # Schedule demo
        "schedule_demo": FlowState(
            prompt="demo_scheduling",
            transitions={
                "scheduled": RegexCondition(
                    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"  # Date pattern
                )
            },
            next_states={"scheduled": "confirm_demo"}
        ),

        # Confirm demo
        "confirm_demo": FlowState(
            prompt="demo_confirmation",
            transitions={},
            next_states={}
        ),

        # Involve stakeholders
        "involve_stakeholders": FlowState(
            prompt="stakeholder_engagement",
            transitions={},
            next_states={}
        ),

        # Nurture (not ready)
        "nurture": FlowState(
            prompt="nurture_campaign",
            transitions={},
            next_states={}
        ),

        # Nurture with timeline
        "nurture_timeline": FlowState(
            prompt="follow_up_timeline",
            transitions={},
            next_states={}
        ),

        # Polite close
        "polite_close": FlowState(
            prompt="thank_you_close",
            transitions={},
            next_states={}
        )
    }
)
```

## Onboarding Flow

### User Onboarding

```python
onboarding_flow = ConversationFlow(
    name="user_onboarding",
    initial_state="welcome",
    states={
        "welcome": FlowState(
            prompt="onboarding_welcome",
            transitions={
                "ready": KeywordCondition(["ready", "start", "begin", "yes"])
            },
            next_states={"ready": "collect_info"}
        ),

        "collect_info": FlowState(
            prompt="collect_user_info",
            transitions={
                "info_provided": LLMClassifierCondition(
                    llm=llm,
                    classification_prompt="Does this message contain user information? {{text}}",
                    target_class="yes"
                )
            },
            next_states={"info_provided": "set_preferences"}
        ),

        "set_preferences": FlowState(
            prompt="preference_questions",
            transitions={
                "preferences_set": KeywordCondition(["done", "finished", "set"])
            },
            next_states={"preferences_set": "feature_tour"}
        ),

        "feature_tour": FlowState(
            prompt="feature_overview",
            transitions={
                "continue_tour": KeywordCondition(["next", "continue", "more"]),
                "skip_tour": KeywordCondition(["skip", "later", "no"])
            },
            next_states={
                "continue_tour": "feature_demo",
                "skip_tour": "completion"
            }
        ),

        "feature_demo": FlowState(
            prompt="interactive_demo",
            transitions={
                "understood": KeywordCondition(["got it", "understand", "clear"]),
                "more_help": KeywordCondition(["help", "explain", "confused"])
            },
            next_states={
                "understood": "completion",
                "more_help": "additional_help"
            }
        ),

        "additional_help": FlowState(
            prompt="detailed_help",
            transitions={
                "ready_now": KeywordCondition(["ready", "understand", "thanks"])
            },
            next_states={"ready_now": "completion"}
        ),

        "completion": FlowState(
            prompt="onboarding_complete",
            transitions={},
            next_states={}
        )
    }
)
```

## Advanced Patterns

### Loop Detection

```python
# Flow with loop prevention
interview_flow = ConversationFlow(
    name="interview",
    initial_state="start",
    states={
        "start": FlowState(
            prompt="interview_start",
            transitions={"continue": KeywordCondition(["yes"])},
            next_states={"continue": "question_1"}
        ),
        "question_1": FlowState(
            prompt="ask_question_1",
            transitions={
                "answered": KeywordCondition([".*"]),  # Any answer
                "skip": KeywordCondition(["skip", "pass"])
            },
            next_states={
                "answered": "question_2",
                "skip": "question_2"
            }
        ),
        "question_2": FlowState(
            prompt="ask_question_2",
            transitions={
                "answered": KeywordCondition([".*"]),
                "back": KeywordCondition(["back", "previous"])  # Can loop
            },
            next_states={
                "answered": "question_3",
                "back": "question_1"  # Potential loop
            }
        ),
        "question_3": FlowState(
            prompt="ask_question_3",
            transitions={
                "done": KeywordCondition(["done", "finished"])
            },
            next_states={"done": "end"}
        ),
        "end": FlowState(
            prompt="interview_end",
            transitions={},
            next_states={}
        )
    },
    max_loops=3  # Prevent infinite loops
)
```

### Dynamic State Selection

```python
class DynamicFlowState(FlowState):
    """State that selects next state dynamically."""

    async def get_next_state(self, user_input, context):
        # Use LLM to determine next state
        classification = await llm.acomplete(f"""
        Based on this user input: "{user_input}"
        And this context: {context}

        What should the next state be? Choose from:
        - technical_support
        - billing_support
        - general_help

        Answer with just the state name.
        """)

        return classification.content.strip()

dynamic_flow = ConversationFlow(
    name="dynamic_routing",
    initial_state="greeting",
    states={
        "greeting": DynamicFlowState(
            prompt="greeting",
            transitions={"auto": KeywordCondition([".*"])},
            next_states={"auto": "dynamic"}  # Determined at runtime
        ),
        # ... other states
    }
)
```

### State with Context Accumulation

```python
class ContextAccumulatingFlow:
    def __init__(self, flow, manager):
        self.flow = flow
        self.manager = manager
        self.context = {}

    async def execute_with_context(self):
        current_state = self.flow.initial_state

        while current_state:
            state = self.flow.states[current_state]

            # Render prompt with accumulated context
            await self.manager.add_message(
                role="user",
                prompt_name=state.prompt,
                params={"context": self.context}
            )

            response = await self.manager.complete()

            # Extract information from response
            self.context.update(
                self._extract_context(response.content)
            )

            # Determine next state
            current_state = await self._next_state(
                state,
                response.content
            )

    def _extract_context(self, response):
        # Extract key information from response
        # (e.g., user name, preferences, issues mentioned)
        return {}

    async def _next_state(self, state, user_input):
        # Determine next state based on transitions
        for transition_name, condition in state.transitions.items():
            if await condition.evaluate(user_input):
                return state.next_states[transition_name]
        return None
```

## Conversation Persistence with Flows

### Save and Resume Flow

```python
from dataknobs_llm.conversations import DataknobsConversationStorage
from dataknobs_data.backends import AsyncMemoryDatabase

# Create with persistence
db = AsyncMemoryDatabase()
storage = DataknobsConversationStorage(db)

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    flow=support_flow,
    storage=storage,
    conversation_id="user123-support-session1"
)

# Start flow
await manager.execute_flow()

# Later, resume from same point
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    flow=support_flow,
    storage=storage,
    conversation_id="user123-support-session1"  # Loads existing state
)

# Continue from where we left off
await manager.execute_flow()
```

## See Also

- [FSM-Based Flows Guide](../guides/flows.md) - Detailed guide
- [Conversation Management](../guides/conversations.md) - Core concepts
- [FSM Package](../../fsm/index.md) - Underlying FSM framework
- [Basic Usage Examples](basic-usage.md) - Simple conversations
