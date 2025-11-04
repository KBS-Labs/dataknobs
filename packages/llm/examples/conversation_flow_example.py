#!/usr/bin/env python3
"""
Conversation Flow Example.

This example demonstrates how to use ConversationFlow to create
structured conversation flows using the FSM engine.
"""

import asyncio
import logging
from dataknobs_llm.conversations.flow import (
    ConversationFlow,
    FlowState,
    keyword_condition,
    regex_condition,
    always,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_support_flow() -> ConversationFlow:
    """Create a customer support conversation flow.

    This flow demonstrates:
    - Intent detection via keyword conditions
    - State transitions based on user input
    - Loop detection to prevent infinite loops
    - Multiple conversation paths
    """
    flow = ConversationFlow(
        name="customer_support",
        description="Customer support conversation flow with intent routing",
        initial_state="greeting",
        max_total_loops=15,  # Prevent infinite loops
        states={
            # Initial greeting state
            "greeting": FlowState(
                prompt_name="support_greeting",
                transitions={
                    "need_help": "collect_issue",
                    "just_browsing": "end_browsing"
                },
                transition_conditions={
                    "need_help": keyword_condition(
                        ["help", "issue", "problem", "support"],
                        case_sensitive=False
                    ),
                    "just_browsing": keyword_condition(
                        ["browse", "look", "explore", "no thanks"],
                        case_sensitive=False
                    )
                }
            ),

            # Collect issue details
            "collect_issue": FlowState(
                prompt_name="issue_details",
                transitions={
                    "technical": "tech_support",
                    "billing": "billing_support",
                    "account": "account_support",
                    "unclear": "clarify_issue"
                },
                transition_conditions={
                    "technical": keyword_condition(
                        ["technical", "bug", "error", "broken", "crash"],
                        case_sensitive=False
                    ),
                    "billing": keyword_condition(
                        ["billing", "payment", "charge", "invoice", "cost"],
                        case_sensitive=False
                    ),
                    "account": keyword_condition(
                        ["account", "login", "password", "access"],
                        case_sensitive=False
                    ),
                    "unclear": always()  # Fallback
                },
                max_loops=2  # Allow 2 attempts to clarify
            ),

            # Technical support path
            "tech_support": FlowState(
                prompt_name="tech_support_response",
                transitions={
                    "resolved": "end_resolved",
                    "escalate": "end_escalate"
                },
                transition_conditions={
                    "resolved": keyword_condition(
                        ["yes", "solved", "fixed", "worked", "thanks"],
                        case_sensitive=False
                    ),
                    "escalate": keyword_condition(
                        ["no", "still", "didn't work", "escalate", "manager"],
                        case_sensitive=False
                    )
                }
            ),

            # Billing support path
            "billing_support": FlowState(
                prompt_name="billing_support_response",
                transitions={
                    "resolved": "end_resolved",
                    "escalate": "end_escalate"
                },
                transition_conditions={
                    "resolved": keyword_condition(
                        ["yes", "solved", "understood", "thanks"],
                        case_sensitive=False
                    ),
                    "escalate": keyword_condition(
                        ["no", "still", "dispute", "escalate", "manager"],
                        case_sensitive=False
                    )
                }
            ),

            # Account support path
            "account_support": FlowState(
                prompt_name="account_support_response",
                transitions={
                    "resolved": "end_resolved",
                    "escalate": "end_escalate"
                },
                transition_conditions={
                    "resolved": keyword_condition(
                        ["yes", "solved", "logged in", "thanks"],
                        case_sensitive=False
                    ),
                    "escalate": keyword_condition(
                        ["no", "still", "can't access", "escalate"],
                        case_sensitive=False
                    )
                }
            ),

            # Clarify unclear issues
            "clarify_issue": FlowState(
                prompt_name="clarify_request",
                transitions={
                    "retry": "collect_issue",
                    "give_up": "end_unclear"
                },
                transition_conditions={
                    "retry": keyword_condition(
                        ["let me explain", "try again", "here's my issue"],
                        case_sensitive=False
                    ),
                    "give_up": always()  # Fallback
                },
                max_loops=1  # Only clarify once
            ),

            # End states (terminal states have no transitions)
            "end_browsing": FlowState(
                prompt_name="goodbye_browsing",
                transitions={},
                transition_conditions={}
            ),

            "end_resolved": FlowState(
                prompt_name="goodbye_resolved",
                transitions={},
                transition_conditions={}
            ),

            "end_escalate": FlowState(
                prompt_name="escalation_message",
                transitions={},
                transition_conditions={}
            ),

            "end_unclear": FlowState(
                prompt_name="goodbye_unclear",
                transitions={},
                transition_conditions={}
            )
        }
    )

    # Validate flow
    warnings = flow.validate_flow()
    if warnings:
        logger.warning(f"Flow validation warnings: {warnings}")

    return flow


def create_sales_flow() -> ConversationFlow:
    """Create a sales conversation flow.

    This flow demonstrates:
    - Product interest qualification
    - Price discussion
    - Closing techniques
    """
    flow = ConversationFlow(
        name="sales_qualification",
        description="Sales qualification and closing flow",
        initial_state="introduce",
        max_total_loops=20,
        states={
            "introduce": FlowState(
                prompt_name="sales_intro",
                transitions={
                    "interested": "qualify_needs",
                    "not_interested": "end_no_interest"
                },
                transition_conditions={
                    "interested": keyword_condition(
                        ["interested", "tell me more", "yes", "sounds good"],
                        case_sensitive=False
                    ),
                    "not_interested": keyword_condition(
                        ["not interested", "no thanks", "busy", "not now"],
                        case_sensitive=False
                    )
                }
            ),

            "qualify_needs": FlowState(
                prompt_name="qualify_questions",
                transitions={
                    "good_fit": "present_solution",
                    "poor_fit": "end_poor_fit"
                },
                transition_conditions={
                    "good_fit": keyword_condition(
                        ["yes", "need", "looking for", "help with"],
                        case_sensitive=False
                    ),
                    "poor_fit": keyword_condition(
                        ["no", "don't need", "already have"],
                        case_sensitive=False
                    )
                }
            ),

            "present_solution": FlowState(
                prompt_name="solution_presentation",
                transitions={
                    "discuss_price": "pricing_discussion",
                    "not_convinced": "handle_objection"
                },
                transition_conditions={
                    "discuss_price": keyword_condition(
                        ["price", "cost", "how much", "pricing"],
                        case_sensitive=False
                    ),
                    "not_convinced": keyword_condition(
                        ["but", "however", "concern", "not sure"],
                        case_sensitive=False
                    )
                }
            ),

            "handle_objection": FlowState(
                prompt_name="objection_handling",
                transitions={
                    "convinced": "pricing_discussion",
                    "still_not_convinced": "end_follow_up"
                },
                transition_conditions={
                    "convinced": keyword_condition(
                        ["makes sense", "okay", "alright", "fair"],
                        case_sensitive=False
                    ),
                    "still_not_convinced": always()
                },
                max_loops=2
            ),

            "pricing_discussion": FlowState(
                prompt_name="pricing_info",
                transitions={
                    "close": "end_close",
                    "think_about_it": "end_think"
                },
                transition_conditions={
                    "close": keyword_condition(
                        ["sounds good", "let's do it", "sign me up", "yes"],
                        case_sensitive=False
                    ),
                    "think_about_it": keyword_condition(
                        ["think about it", "get back", "consider", "maybe"],
                        case_sensitive=False
                    )
                }
            ),

            # End states
            "end_no_interest": FlowState(
                prompt_name="goodbye_no_interest",
                transitions={},
                transition_conditions={}
            ),

            "end_poor_fit": FlowState(
                prompt_name="goodbye_poor_fit",
                transitions={},
                transition_conditions={}
            ),

            "end_follow_up": FlowState(
                prompt_name="schedule_follow_up",
                transitions={},
                transition_conditions={}
            ),

            "end_think": FlowState(
                prompt_name="goodbye_think",
                transitions={},
                transition_conditions={}
            ),

            "end_close": FlowState(
                prompt_name="congratulations_close",
                transitions={},
                transition_conditions={}
            )
        }
    )

    return flow


async def example_usage():
    """Demonstrate flow usage.

    Note: This is a simplified example. In production, you would:
    1. Set up actual LLM providers
    2. Configure prompt library with real prompts
    3. Use ConversationManager to execute flows
    """
    logger.info("Creating support flow...")
    support_flow = create_support_flow()

    logger.info(f"Flow: {support_flow.name}")
    logger.info(f"Initial state: {support_flow.initial_state}")
    logger.info(f"Total states: {len(support_flow.states)}")

    # Show flow structure
    logger.info("\nFlow structure:")
    for state_name, state in support_flow.states.items():
        transitions_str = ", ".join(
            f"{cond} -> {target}"
            for cond, target in state.transitions.items()
        )
        logger.info(f"  {state_name}:")
        logger.info(f"    Prompt: {state.prompt_name}")
        if transitions_str:
            logger.info(f"    Transitions: {transitions_str}")
        if state.max_loops:
            logger.info(f"    Max loops: {state.max_loops}")

    # Validate flows
    logger.info("\nValidating support flow...")
    warnings = support_flow.validate_flow()
    if warnings:
        logger.warning(f"Warnings: {warnings}")
    else:
        logger.info("No validation issues!")

    logger.info("\nCreating sales flow...")
    sales_flow = create_sales_flow()
    logger.info(f"Sales flow has {len(sales_flow.states)} states")

    logger.info("\n" + "="*60)
    logger.info("Example complete!")
    logger.info("\nTo use these flows in production:")
    logger.info("1. Create a ConversationManager with LLM and prompt library")
    logger.info("2. Call: async for node in manager.execute_flow(flow):")
    logger.info("3. Process each node as the conversation progresses")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(example_usage())
