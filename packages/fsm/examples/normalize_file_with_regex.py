#!/usr/bin/env python3
"""Example of using SimpleFSM with regular expressions in inline transforms."""

import yaml
from dataknobs_fsm.api.simple import SimpleFSM

# FSM workflow using regex for text normalization
REGEX_NORMALIZE_WORKFLOW_YAML = '''
name: regex_normalization_workflow
description: Process text using regular expressions in inline transforms

states:
  - name: start
    is_start: true
    metadata:
      description: Initial state to receive text line

  - name: clean_whitespace
    metadata:
      description: Normalize whitespace using regex

  - name: remove_punctuation
    metadata:
      description: Remove or normalize punctuation

  - name: normalize_emails
    metadata:
      description: Normalize email addresses to lowercase

  - name: complete
    is_end: true
    metadata:
      description: Final state with fully normalized text

arcs:
  - from: start
    to: clean_whitespace
    # Use regex to replace multiple spaces with single space
    transform:
      type: inline
      code: |
        lambda data, ctx: {
            **data,
            'original_text': data.get('original_text', data.get('text', '')),
            'clean_whitespace': __import__('re').sub(r'\\s+', ' ', data.get('text', '')).strip()
        }
    metadata:
      description: Clean up whitespace

  - from: clean_whitespace
    to: remove_punctuation
    # Remove excessive punctuation but keep single instances
    transform:
      type: inline
      code: |
        lambda data, ctx: {
            **data,
            'clean_punctuation': __import__('re').sub(r'([.!?,;])\\1+', r'\\1', data.get('clean_whitespace', ''))
        }
    metadata:
      description: Normalize repeated punctuation

  - from: remove_punctuation
    to: normalize_emails
    # Convert email addresses to lowercase using regex
    transform:
      type: inline
      code: |
        lambda data, ctx: {
            **data,
            'normalized_emails': __import__('re').sub(
                r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b',
                lambda m: m.group(0).lower(),
                data.get('clean_punctuation', '')
            )
        }
    metadata:
      description: Normalize email addresses

  - from: normalize_emails
    to: complete
    # Final normalization: capitalize sentences
    transform:
      type: inline
      code: |
        lambda data, ctx: {
            **data,
            'capitalized': __import__('re').sub(
                r'(^|\\. )([a-z])',
                lambda m: m.group(1) + m.group(2).upper(),
                data.get('normalized_emails', '')
            ),
            'final_text': __import__('re').sub(
                r'(^|\\. )([a-z])',
                lambda m: m.group(1) + m.group(2).upper(),
                data.get('normalized_emails', '')
            )
        }
    metadata:
      description: Capitalize first letter of sentences
'''

# Alternative approach: More complex regex operations
ADVANCED_REGEX_WORKFLOW_YAML = '''
name: advanced_regex_workflow
description: Advanced text processing with complex regex patterns

states:
  - name: start
    is_start: true
  - name: process
  - name: complete
    is_end: true

arcs:
  - from: start
    to: process
    # Complex multi-step regex processing in a single transform
    transform:
      type: inline
      code: |
        lambda data, ctx: (lambda re: {
            **data,
            'text': re.sub(
                r'\\b(\\w+)\\b(?:\\s+\\1\\b)+',  # Remove duplicate words
                r'\\1',
                re.sub(
                    r'https?://[^\\s]+',  # Replace URLs with [URL]
                    '[URL]',
                    re.sub(
                        r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b',  # Mask phone numbers
                        '[PHONE]',
                        re.sub(
                            r'\\s+', ' ',  # Normalize whitespace
                            data.get('text', '')
                        )
                    )
                )
            ).strip()
        })(__import__('re'))
    metadata:
      description: Apply all regex transformations

  - from: process
    to: complete
'''

def demo_regex_normalization():
    """Demonstrate regex-based text normalization."""

    # Load the workflow
    config = yaml.safe_load(REGEX_NORMALIZE_WORKFLOW_YAML)
    fsm = SimpleFSM(config)

    # Test data with various normalization needs
    test_texts = [
        "  This   has    too    much   whitespace  ",
        "Multiple punctuation marks!!!!! Need fixing.....",
        "Email addresses like JOHN@EXAMPLE.COM should be lowercase",
        "this sentence. needs capitalization. at the start.",
        "Repeated repeated words words need fixing",
    ]

    print("Regex-based Normalization Results:")
    print("=" * 60)

    try:
        for text in test_texts:
            result = fsm.process({'text': text})

            if result['success']:
                data = result['data']
                print(f"Original:            '{data.get('original_text', text)}'")
                print(f"clean_whitespace:    '{data.get('clean_whitespace', '')}'")
                print(f"clean_punctuation:   '{data.get('clean_punctuation', '')}'")
                print(f"normalized_emails:   '{data.get('normalized_emails', '')}'")
                print(f"capitalized:         '{data.get('capitalized', '')}'")
                print(f"Final text:          '{data.get('final_text', '')}'")
                print()
            else:
                print(f"Error processing: {text}")
                print(f"Error: {result.get('error')}")
    finally:
        fsm.close()

def demo_advanced_regex():
    """Demonstrate advanced regex processing."""

    config = yaml.safe_load(ADVANCED_REGEX_WORKFLOW_YAML)
    fsm = SimpleFSM(config)

    test_texts = [
        "Visit https://example.com for more more information",
        "Call me at 555-123-4567 or 555.123.4567",
        "This this has duplicate duplicate words words",
        "Check out http://website.org and call 800-555-1234",
    ]

    print("\nAdvanced Regex Processing:")
    print("=" * 60)

    try:
        for text in test_texts:
            result = fsm.process({'text': text})

            if result['success']:
                print(f"Original:   '{text}'")
                print(f"Processed:  '{result['data']['text']}'")
                print()
    finally:
        fsm.close()

def create_custom_regex_workflow(patterns: dict) -> dict:
    """Create a custom workflow with user-defined regex patterns."""

    # Build the transform code dynamically
    transform_code = "lambda data, ctx: (lambda re: {**data, 'text': "

    # Chain regex substitutions
    text_expr = "data.get('text', '')"
    for pattern, replacement in patterns.items():
        # Don't escape - patterns are already raw strings
        text_expr = f"re.sub(r'{pattern}', r'{replacement}', {text_expr})"

    transform_code += text_expr + "})(__import__('re'))"

    return {
        'name': 'custom_regex_workflow',
        'states': [
            {'name': 'start', 'is_start': True},
            {'name': 'process'},
            {'name': 'complete', 'is_end': True}
        ],
        'arcs': [
            {
                'from': 'start',
                'to': 'process',
                'transform': {
                    'type': 'inline',
                    'code': transform_code
                }
            },
            {'from': 'process', 'to': 'complete'}
        ]
    }

def demo_custom_regex():
    """Demonstrate dynamically created regex workflows."""

    # Define custom patterns
    patterns = {
        r'\b([A-Z]{2,})\b': lambda m: m.group(1).capitalize(),  # ACRONYMS -> Acronyms
        r'(\d+)\s*%': r'\1 percent',  # 50% -> 50 percent
        r'\$(\d+)': r'\1 dollars',  # $100 -> 100 dollars
    }

    # For the config, we need string replacements
    string_patterns = {
        r'\b([A-Z]{2,})\b': r'\1',  # Keep as-is for now
        r'(\d+)\s*%': r'\1 percent',
        r'\$(\d+)': r'\1 dollars',
    }

    config = create_custom_regex_workflow(string_patterns)
    fsm = SimpleFSM(config)

    test_texts = [
        "The API costs $50 with a 10% discount",
        "NASA and FBI are ACRONYMS",
        "Save 25% on items over $100",
    ]

    print("\nCustom Regex Patterns:")
    print("=" * 60)

    try:
        for text in test_texts:
            result = fsm.process({'text': text})

            if result['success']:
                print(f"Original:   '{text}'")
                print(f"Processed:  '{result['data']['text']}'")
                print()
    finally:
        fsm.close()

if __name__ == "__main__":
    # Run all demonstrations
    demo_regex_normalization()
    demo_advanced_regex()
    demo_custom_regex()
