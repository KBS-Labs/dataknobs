#!/usr/bin/env python3
"""Test script for regex transformations defined in YAML."""

import yaml
from pathlib import Path
from dataknobs_fsm.api.simple import SimpleFSM

def test_regex_yaml():
    """Test the regex transformations from YAML config."""

    # Load the first configuration from the YAML file
    yaml_path = Path(__file__).parent / "regex_transforms.yaml"

    with open(yaml_path) as f:
        configs = list(yaml.safe_load_all(f))

    # Test the field transforms workflow
    print("Testing Field Transforms Workflow")
    print("=" * 60)

    fsm1 = SimpleFSM(configs[0])

    test_data = {
        'text': "Contact John at 555-123-4567 or email john@example.com. Visit https://example.com #urgent @support"
    }

    result = fsm1.process(test_data)

    if result['success']:
        data = result['data']
        print(f"Original:              '{data.get('original', '')}'")
        print(f"Whitespace normalized: '{data.get('whitespace_normalized', '')}'")
        print(f"Phone masked:          '{data.get('phone_masked', '')}'")
        print(f"SSN masked:            '{data.get('ssn_masked', '')}'")
        print(f"Emails found:          {data.get('emails_found', [])}")
        print(f"URLs found:            {data.get('urls_found', [])}")
        print(f"Hashtags found:        {data.get('hashtags_found', [])}")
        print(f"Mentions found:        {data.get('mentions_found', [])}")

    fsm1.close()

    print("\n" + "=" * 60)
    print("Testing All-in-One Transforms Workflow")
    print("=" * 60)

    fsm2 = SimpleFSM(configs[1])

    test_data2 = {
        'text': "Hello World 123! Test message with email@example.com"
    }

    result2 = fsm2.process(test_data2)

    if result2['success']:
        data = result2['data']
        print(f"Original:         '{data.get('original_text', '')}'")
        print(f"Lowercase:        '{data.get('lowercase', '')}'")
        print(f"Uppercase:        '{data.get('uppercase', '')}'")
        print(f"Title case:       '{data.get('title_case', '')}'")
        print(f"No punctuation:   '{data.get('no_punctuation', '')}'")
        print(f"No digits:        '{data.get('no_digits', '')}'")
        print(f"Alphanumeric:     '{data.get('alphanumeric_only', '')}'")
        print(f"Words only:       '{data.get('words_only', '')}'")
        print(f"Snake case:       '{data.get('snake_case', '')}'")
        print(f"Kebab case:       '{data.get('kebab_case', '')}'")
        print(f"Camel case:       '{data.get('camel_case', '')}'")
        print(f"Word count:       {data.get('word_count', 0)}")
        print(f"Char count:       {data.get('char_count', 0)}")
        print(f"Has email:        {data.get('has_email', False)}")
        print(f"Has URL:          {data.get('has_url', False)}")
        print(f"Has phone:        {data.get('has_phone', False)}")

    fsm2.close()

if __name__ == "__main__":
    test_regex_yaml()