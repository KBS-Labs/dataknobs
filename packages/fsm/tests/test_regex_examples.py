"""Unit tests for regex transformation examples."""

import pytest
import yaml
from pathlib import Path
from dataknobs_fsm.api.simple import SimpleFSM


class TestRegexNormalizationWorkflow:
    """Test the regex normalization workflow examples."""

    def test_whitespace_normalization(self):
        """Test that multiple spaces are normalized to single space."""
        config = {
            'name': 'test_whitespace',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'normalize'},
                {'name': 'complete', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'normalize',
                    'transform': {
                        'type': 'inline',
                        'code': "lambda data, ctx: {**data, 'normalized': __import__('re').sub(r'\\s+', ' ', data.get('text', '')).strip()}"
                    }
                },
                {'from': 'normalize', 'to': 'complete'}
            ]
        }

        fsm = SimpleFSM(config)
        try:
            result = fsm.process({'text': '  This   has    too    much   whitespace  '})
            assert result['success']
            assert result['data']['normalized'] == 'This has too much whitespace'
        finally:
            fsm.close()

    def test_punctuation_normalization(self):
        """Test that repeated punctuation is reduced to single instances."""
        config = {
            'name': 'test_punctuation',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'normalize'},
                {'name': 'complete', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'normalize',
                    'transform': {
                        'type': 'inline',
                        'code': "lambda data, ctx: {**data, 'normalized': __import__('re').sub(r'([.!?,;])\\1+', r'\\1', data.get('text', ''))}"
                    }
                },
                {'from': 'normalize', 'to': 'complete'}
            ]
        }

        fsm = SimpleFSM(config)
        try:
            result = fsm.process({'text': 'Multiple punctuation marks!!!!! Need fixing.....'})
            assert result['success']
            assert result['data']['normalized'] == 'Multiple punctuation marks! Need fixing.'
        finally:
            fsm.close()

    def test_email_lowercase_normalization(self):
        """Test that email addresses are converted to lowercase."""
        config = {
            'name': 'test_email',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'normalize'},
                {'name': 'complete', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'normalize',
                    'transform': {
                        'type': 'inline',
                        'code': """lambda data, ctx: {
                            **data,
                            'normalized': __import__('re').sub(
                                r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b',
                                lambda m: m.group(0).lower(),
                                data.get('text', '')
                            )
                        }"""
                    }
                },
                {'from': 'normalize', 'to': 'complete'}
            ]
        }

        fsm = SimpleFSM(config)
        try:
            result = fsm.process({'text': 'Contact JOHN@EXAMPLE.COM for info'})
            assert result['success']
            assert result['data']['normalized'] == 'Contact john@example.com for info'
        finally:
            fsm.close()

    def test_sentence_capitalization(self):
        """Test that first letter of sentences is capitalized."""
        config = {
            'name': 'test_capitalize',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'normalize'},
                {'name': 'complete', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'normalize',
                    'transform': {
                        'type': 'inline',
                        'code': """lambda data, ctx: {
                            **data,
                            'normalized': __import__('re').sub(
                                r'(^|\\. )([a-z])',
                                lambda m: m.group(1) + m.group(2).upper(),
                                data.get('text', '')
                            )
                        }"""
                    }
                },
                {'from': 'normalize', 'to': 'complete'}
            ]
        }

        fsm = SimpleFSM(config)
        try:
            result = fsm.process({'text': 'this sentence. needs capitalization. at the start.'})
            assert result['success']
            assert result['data']['normalized'] == 'This sentence. Needs capitalization. At the start.'
        finally:
            fsm.close()

    def test_preserve_original_text(self):
        """Test that original text is preserved while adding transformation fields."""
        config = {
            'name': 'test_preserve',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'transform'},
                {'name': 'complete', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'transform',
                    'transform': {
                        'type': 'inline',
                        'code': """lambda data, ctx: {
                            **data,
                            'original_text': data.get('text', ''),
                            'clean_whitespace': __import__('re').sub(r'\\s+', ' ', data.get('text', '')).strip(),
                            'lowercase': data.get('text', '').lower()
                        }"""
                    }
                },
                {'from': 'transform', 'to': 'complete'}
            ]
        }

        fsm = SimpleFSM(config)
        try:
            original = '  Test   TEXT  '
            result = fsm.process({'text': original})
            assert result['success']
            data = result['data']
            assert data['original_text'] == original
            assert data['clean_whitespace'] == 'Test TEXT'
            assert data['lowercase'] == '  test   text  '
            assert data['text'] == original  # Original text field unchanged
        finally:
            fsm.close()


class TestAdvancedRegexProcessing:
    """Test advanced regex processing patterns."""

    def test_url_masking(self):
        """Test that URLs are replaced with [URL] placeholder."""
        config = {
            'name': 'test_url',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'mask'},
                {'name': 'complete', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'mask',
                    'transform': {
                        'type': 'inline',
                        'code': "lambda data, ctx: {**data, 'masked': __import__('re').sub(r'https?://[^\\s]+', '[URL]', data.get('text', ''))}"
                    }
                },
                {'from': 'mask', 'to': 'complete'}
            ]
        }

        fsm = SimpleFSM(config)
        try:
            result = fsm.process({'text': 'Visit https://example.com and http://test.org for info'})
            assert result['success']
            assert result['data']['masked'] == 'Visit [URL] and [URL] for info'
        finally:
            fsm.close()

    def test_phone_number_masking(self):
        """Test that phone numbers are replaced with [PHONE] placeholder."""
        config = {
            'name': 'test_phone',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'mask'},
                {'name': 'complete', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'mask',
                    'transform': {
                        'type': 'inline',
                        'code': "lambda data, ctx: {**data, 'masked': __import__('re').sub(r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b', '[PHONE]', data.get('text', ''))}"
                    }
                },
                {'from': 'mask', 'to': 'complete'}
            ]
        }

        fsm = SimpleFSM(config)
        try:
            test_cases = [
                ('Call 555-123-4567', 'Call [PHONE]'),
                ('Phone: 555.123.4567', 'Phone: [PHONE]'),
                ('Number is 5551234567', 'Number is [PHONE]'),
            ]
            for input_text, expected in test_cases:
                result = fsm.process({'text': input_text})
                assert result['success']
                assert result['data']['masked'] == expected
        finally:
            fsm.close()

    def test_duplicate_word_removal(self):
        """Test that duplicate consecutive words are removed."""
        config = {
            'name': 'test_duplicates',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'deduplicate'},
                {'name': 'complete', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'deduplicate',
                    'transform': {
                        'type': 'inline',
                        'code': "lambda data, ctx: {**data, 'deduped': __import__('re').sub(r'\\b(\\w+)\\b(?:\\s+\\1\\b)+', r'\\1', data.get('text', ''))}"
                    }
                },
                {'from': 'deduplicate', 'to': 'complete'}
            ]
        }

        fsm = SimpleFSM(config)
        try:
            test_cases = [
                ('This this has duplicate duplicate words', 'This this has duplicate words'),  # Case-sensitive: This != this
                ('The the the quick quick brown fox', 'The the quick brown fox'),  # Consecutive duplicates
                ('test test has duplicate duplicate words', 'test has duplicate words'),  # All lowercase
                ('No duplicates here', 'No duplicates here'),
            ]
            for input_text, expected in test_cases:
                result = fsm.process({'text': input_text})
                assert result['success']
                assert result['data']['deduped'] == expected
        finally:
            fsm.close()


class TestRegexTransformsYAML:
    """Test regex transformations loaded from YAML files."""

    def get_yaml_configs(self):
        """Load configurations from regex_transforms.yaml."""
        yaml_path = Path(__file__).parent.parent / "examples" / "regex_transforms.yaml"
        if not yaml_path.exists():
            pytest.skip(f"YAML file not found: {yaml_path}")

        with open(yaml_path) as f:
            return list(yaml.safe_load_all(f))

    def test_field_transforms_workflow(self):
        """Test the field transforms workflow from YAML."""
        configs = self.get_yaml_configs()
        fsm = SimpleFSM(configs[0])  # First config: regex_field_transforms

        try:
            test_data = {
                'text': "Contact John at 555-123-4567 or email john@example.com. Visit https://example.com #urgent @support"
            }

            result = fsm.process(test_data)
            assert result['success']

            data = result['data']
            # Check original is preserved
            assert data['original'] == test_data['text']

            # Check whitespace normalization (no extra spaces in this case)
            assert data['whitespace_normalized'] == test_data['text']

            # Check phone masking
            assert '[PHONE]' in data['phone_masked']
            assert '555-123-4567' not in data['phone_masked']

            # Check pattern extraction
            assert data['emails_found'] == ['john@example.com']
            assert data['urls_found'] == ['https://example.com']
            assert data['hashtags_found'] == ['#urgent']
            assert '@support' in data['mentions_found']

            # Check completion flag
            assert data['processing_complete'] is True
            assert 'whitespace_normalized' in data['transformations_applied']
        finally:
            fsm.close()

    def test_all_in_one_transforms(self):
        """Test the all-in-one transformation workflow from YAML."""
        configs = self.get_yaml_configs()
        fsm = SimpleFSM(configs[1])  # Second config: all_in_one_regex

        try:
            test_data = {
                'text': "Hello World 123! Test message with email@example.com"
            }

            result = fsm.process(test_data)
            assert result['success']

            data = result['data']

            # Check original is preserved
            assert data['original_text'] == test_data['text']

            # Check case transformations
            assert data['lowercase'] == 'hello world 123! test message with email@example.com'
            assert data['uppercase'] == 'HELLO WORLD 123! TEST MESSAGE WITH EMAIL@EXAMPLE.COM'
            assert data['title_case'] == 'Hello World 123! Test Message With Email@Example.Com'

            # Check punctuation removal
            assert '!' not in data['no_punctuation']
            assert '@' not in data['no_punctuation']

            # Check digit removal
            assert '123' not in data['no_digits']

            # Check alphanumeric only
            assert '!' not in data['alphanumeric_only']
            assert '@' not in data['alphanumeric_only']

            # Check case conversions
            assert data['snake_case'] == 'hello_world_123!_test_message_with_email@example.com'
            assert data['kebab_case'] == 'hello-world-123!-test-message-with-email@example.com'
            assert ' ' not in data['camel_case']

            # Check counts
            assert data['word_count'] == 7
            assert data['char_count'] == 52

            # Check pattern detection
            assert data['has_email'] is True
            assert data['has_url'] is False
            assert data['has_phone'] is False
        finally:
            fsm.close()

    def test_sensitive_data_masking(self):
        """Test masking of sensitive information."""
        configs = self.get_yaml_configs()
        fsm = SimpleFSM(configs[0])  # First config has masking

        try:
            test_cases = [
                {
                    'input': {'text': 'My SSN is 123-45-6789 and phone is 555-123-4567'},
                    'check': lambda d: '[SSN]' in d['ssn_masked'] and '[PHONE]' in d['phone_masked']
                },
                {
                    'input': {'text': 'Card number: 1234-5678-9012-3456'},
                    'check': lambda d: '[CARD]' in d.get('credit_card_masked', '')
                }
            ]

            for test_case in test_cases:
                result = fsm.process(test_case['input'])
                assert result['success']
                assert test_case['check'](result['data'])
        finally:
            fsm.close()


class TestCustomRegexWorkflow:
    """Test dynamically created custom regex workflows."""

    def test_custom_pattern_replacement(self):
        """Test custom regex pattern replacements."""
        from examples.normalize_file_with_regex import create_custom_regex_workflow

        patterns = {
            r'\b([A-Z]{2,})\b': r'\1',  # Keep acronyms
            r'(\d+)\s*%': r'\1 percent',  # Replace % with 'percent'
            r'\$(\d+)': r'\1 dollars',  # Replace $ with 'dollars'
        }

        config = create_custom_regex_workflow(patterns)
        fsm = SimpleFSM(config)

        try:
            test_cases = [
                ('The API costs $50 with a 10% discount', 'The API costs 50 dollars with a 10 percent discount'),
                ('Save 25% on items over $100', 'Save 25 percent on items over 100 dollars'),
                ('NASA and FBI are ACRONYMS', 'NASA and FBI are ACRONYMS'),
            ]

            for input_text, expected in test_cases:
                result = fsm.process({'text': input_text})
                assert result['success']
                assert result['data']['text'] == expected
        finally:
            fsm.close()

    def test_empty_text_handling(self):
        """Test handling of empty or missing text."""
        config = {
            'name': 'test_empty',
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
                        'code': "lambda data, ctx: {**data, 'normalized': __import__('re').sub(r'\\s+', ' ', data.get('text') or '').strip()}"
                    }
                },
                {'from': 'process', 'to': 'complete'}
            ]
        }

        fsm = SimpleFSM(config)
        try:
            # Test with empty string
            result = fsm.process({'text': ''})
            assert result['success']
            assert result['data']['normalized'] == ''

            # Test with missing text field
            result = fsm.process({})
            assert result['success']
            assert result['data']['normalized'] == ''

            # Test with None value
            result = fsm.process({'text': None})
            assert result['success']
            assert result['data']['normalized'] == ''
        finally:
            fsm.close()


class TestRegexPerformance:
    """Test performance characteristics of regex transformations."""

    def test_multiple_field_preservation(self):
        """Test that multiple transformation fields are preserved through the pipeline."""
        config = {
            'name': 'test_multi_field',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'step1'},
                {'name': 'step2'},
                {'name': 'step3'},
                {'name': 'complete', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'step1',
                    'transform': {
                        'type': 'inline',
                        'code': "lambda data, ctx: {**data, 'original': data.get('text', ''), 'step1_result': data.get('text', '').lower()}"
                    }
                },
                {
                    'from': 'step1',
                    'to': 'step2',
                    'transform': {
                        'type': 'inline',
                        'code': "lambda data, ctx: {**data, 'step2_result': __import__('re').sub(r'\\s+', '_', data.get('step1_result', ''))}"
                    }
                },
                {
                    'from': 'step2',
                    'to': 'step3',
                    'transform': {
                        'type': 'inline',
                        'code': "lambda data, ctx: {**data, 'step3_result': data.get('step2_result', '').replace('_', '-')}"
                    }
                },
                {
                    'from': 'step3',
                    'to': 'complete',
                    'transform': {
                        'type': 'inline',
                        'code': "lambda data, ctx: {**data, 'final': data.get('step3_result', ''), 'all_steps_complete': True}"
                    }
                }
            ]
        }

        fsm = SimpleFSM(config)
        try:
            result = fsm.process({'text': 'Test Multiple Steps'})
            assert result['success']

            data = result['data']
            # All intermediate results should be preserved
            assert data['original'] == 'Test Multiple Steps'
            assert data['step1_result'] == 'test multiple steps'
            assert data['step2_result'] == 'test_multiple_steps'
            assert data['step3_result'] == 'test-multiple-steps'
            assert data['final'] == 'test-multiple-steps'
            assert data['all_steps_complete'] is True
        finally:
            fsm.close()

    def test_complex_regex_chaining(self):
        """Test chaining of complex regex operations."""
        config = {
            'name': 'test_complex',
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
                        'code': """lambda data, ctx: (lambda re, text: {
                            **data,
                            'original': text,
                            'processed': re.sub(
                                r'[^\\w\\s]', '',  # Remove punctuation
                                re.sub(
                                    r'\\b(\\w+)\\b(?:\\s+\\1\\b)+', r'\\1',  # Remove duplicates
                                    re.sub(
                                        r'\\s+', ' ',  # Normalize spaces
                                        text.lower()  # Lowercase
                                    )
                                )
                            ).strip()
                        })(__import__('re'), data.get('text', ''))"""
                    }
                },
                {'from': 'process', 'to': 'complete'}
            ]
        }

        fsm = SimpleFSM(config)
        try:
            result = fsm.process({'text': 'This  THIS  has   DUPLICATE duplicate  words!!! And punctuation...'})
            assert result['success']

            data = result['data']
            # Check that all transformations were applied
            assert 'this' in data['processed'].lower()  # Lowercased
            assert '  ' not in data['processed']  # No multiple spaces
            assert '!' not in data['processed']  # No punctuation
            assert data['processed'].count('this') == 1  # No duplicates
            assert data['processed'].count('duplicate') == 1  # No duplicates
        finally:
            fsm.close()