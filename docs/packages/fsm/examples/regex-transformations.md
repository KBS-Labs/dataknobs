# Regex Transformations with FSM

This guide demonstrates how to use regular expressions directly in FSM YAML configurations for powerful text transformations.

## Overview

The FSM framework supports using Python's `re` module directly in inline transform blocks within YAML configurations. This enables complex text processing without writing separate Python functions.

## Key Examples

### 1. Text Normalization Pipeline (`normalize_file_example.py`)

A simple example showing how to normalize text files using FSM with streaming support.

**Features:**
- Streaming file processing for memory efficiency
- Text normalization (whitespace, case conversion)
- Multiple processing methods (streaming, batch, individual lines)
- Integration with `SimpleFSM.process_stream()`

**Basic Usage:**
```python
from dataknobs_fsm.api.simple import SimpleFSM
import yaml

WORKFLOW_YAML = '''
name: text_normalization
states:
  - name: start
    is_start: true
  - name: normalize
  - name: complete
    is_end: true

arcs:
  - from: start
    to: normalize
  - from: normalize
    to: complete
    transform:
      type: inline
      code: "lambda data, ctx: {**data, 'text': data.get('text', '').lower().strip()}"
'''

config = yaml.safe_load(WORKFLOW_YAML)
fsm = SimpleFSM(config)

# Process single line
result = fsm.process({'text': '  HELLO WORLD  '})
print(result['data']['text'])  # Output: 'hello world'

fsm.close()
```

### 2. Advanced Regex Transformations (`normalize_file_with_regex.py`)

Comprehensive examples of regex-based text processing with field preservation.

**Features:**
- Multiple regex patterns in a pipeline
- Field preservation (original text kept while adding transformation fields)
- Pattern extraction (emails, URLs, phone numbers)
- Sensitive data masking
- Custom regex workflow generation

**Key Patterns Demonstrated:**

#### Whitespace Normalization
```yaml
transform:
  type: inline
  code: |
    lambda data, ctx: {
        **data,
        'clean_whitespace': __import__('re').sub(r'\\s+', ' ', data.get('text', '')).strip()
    }
```

#### Email Normalization
```yaml
transform:
  type: inline
  code: |
    lambda data, ctx: {
        **data,
        'normalized_emails': __import__('re').sub(
            r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b',
            lambda m: m.group(0).lower(),
            data.get('text', '')
        )
    }
```

#### Phone Number Masking
```yaml
transform:
  type: inline
  code: |
    lambda data, ctx: {
        **data,
        'phone_masked': __import__('re').sub(
            r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b',
            '[PHONE]',
            data.get('text', '')
        )
    }
```

#### Duplicate Word Removal
```yaml
transform:
  type: inline
  code: |
    lambda data, ctx: {
        **data,
        'deduped': __import__('re').sub(
            r'\\b(\\w+)\\b(?:\\s+\\1\\b)+',
            r'\\1',
            data.get('text', '')
        )
    }
```

### 3. YAML-Based Regex Configurations (`regex_transforms.yaml`)

Two complete YAML workflows demonstrating different approaches:

1. **Field Transforms Workflow** - Sequential transformations with field tracking
2. **All-in-One Transforms** - Multiple transformations in a single step

**Example Output:**
```python
# Input
{'text': 'Contact John at 555-123-4567 or email john@example.com'}

# Output (field transforms)
{
    'original': 'Contact John at 555-123-4567 or email john@example.com',
    'whitespace_normalized': 'Contact John at 555-123-4567 or email john@example.com',
    'phone_masked': 'Contact John at [PHONE] or email john@example.com',
    'emails_found': ['john@example.com'],
    'urls_found': [],
    'hashtags_found': [],
    'processing_complete': True
}
```

### 4. Pattern Extraction Workflow (`regex_workflow.yaml`)

YAML configuration for extracting and transforming specific patterns.

**Features:**
- Email, URL, hashtag, and mention extraction
- SSN and credit card masking
- Multiple format conversions (snake_case, kebab-case, CamelCase)
- Pattern detection flags

## Using Regular Expressions in YAML

### Basic Pattern

The key to using regex in YAML configurations is the `__import__('re')` pattern:

```yaml
transform:
  type: inline
  code: |
    lambda data, ctx: {
        **data,
        'result': __import__('re').sub(
            r'pattern',
            r'replacement',
            data.get('field', '')
        )
    }
```

### Important Escaping Rules

When using regex in YAML:
1. Backslashes in regex patterns must be escaped: `\\s` instead of `\s`
2. Use raw strings (r'...') in the Python code
3. For backreferences, use `\\1` in the pattern

### Complex Patterns Example

```yaml
# Chaining multiple regex operations
transform:
  type: inline
  code: |
    lambda data, ctx: (lambda re, text: {
        **data,
        'processed': re.sub(
            r'[^\\w\\s]', '',  # Remove punctuation
            re.sub(
                r'\\b(\\w+)\\b(?:\\s+\\1\\b)+', r'\\1',  # Remove duplicates
                re.sub(
                    r'\\s+', ' ',  # Normalize spaces
                    text.lower()
                )
            )
        ).strip()
    })(__import__('re'), data.get('text', ''))
```

## Field Preservation Pattern

Best practice is to preserve the original text while adding transformation fields:

```yaml
arcs:
  - from: start
    to: step1
    transform:
      type: inline
      code: |
        lambda data, ctx: {
            **data,
            'original': data.get('text', ''),  # Preserve original
            'step1_result': transform_function(data.get('text', ''))
        }

  - from: step1
    to: step2
    transform:
      type: inline
      code: |
        lambda data, ctx: {
            **data,  # Keep all previous fields
            'step2_result': transform_function(data.get('step1_result', ''))
        }
```

## Common Use Cases

### 1. Data Cleaning Pipeline
- Remove extra whitespace
- Normalize punctuation
- Fix capitalization
- Remove duplicate words

### 2. Data Masking for Privacy
- Mask phone numbers, SSNs, credit cards
- Anonymize email addresses
- Redact sensitive patterns

### 3. Format Standardization
- Convert between naming conventions (snake_case, CamelCase)
- Normalize dates and times
- Standardize phone number formats

### 4. Content Extraction
- Extract emails, URLs, mentions
- Find and collect hashtags
- Identify specific patterns

## Running the Examples

```bash
# Navigate to FSM package
cd packages/fsm

# Run text normalization example
python examples/normalize_file_example.py

# Run regex transformation examples
python examples/normalize_file_with_regex.py

# Test YAML-based transformations
python examples/test_regex_yaml.py

# Process a file with regex transformations
python examples/normalize_file_example.py
```

### Processing Files

The examples support file processing:

```python
from dataknobs_fsm.api.simple import SimpleFSM

# Load configuration
fsm = SimpleFSM("regex_transforms.yaml")

# Process file with streaming
results = fsm.process_stream(
    source="input.txt",
    sink="output.jsonl",
    input_format='text',
    text_field_name='text',
    chunk_size=1000
)

print(f"Processed: {results['total_processed']} lines")
fsm.close()
```

## Testing

Comprehensive unit tests are provided in `tests/test_regex_examples.py`:

```bash
# Run regex example tests
pytest tests/test_regex_examples.py -v

# Run specific test class
pytest tests/test_regex_examples.py::TestRegexNormalizationWorkflow -v
```

## Best Practices

1. **Always preserve original data** - Add new fields rather than overwriting
2. **Use descriptive field names** - Make it clear what each transformation does
3. **Handle None/empty values** - Use `data.get('field') or ''` pattern
4. **Test regex patterns** - Verify patterns work as expected before deployment
5. **Document complex patterns** - Add comments explaining what patterns do
6. **Consider performance** - Chain operations efficiently

## Limitations

- Complex regex patterns can impact performance on large datasets
- YAML escaping rules can make patterns harder to read
- Debug output may be needed for complex transformations
- Some regex features may require custom Python functions

## See Also

- [SimpleFSM API Reference](../api/simple.md)
- [Data Modes Guide](../guides/data-modes.md)
- [Streaming Guide](../guides/streaming.md)
- [Python re module documentation](https://docs.python.org/3/library/re.html)