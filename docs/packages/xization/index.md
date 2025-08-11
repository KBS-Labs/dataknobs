# Dataknobs Xization

The `dataknobs-xization` package provides text processing, normalization, and tokenization tools for AI knowledge bases.

## Installation

```bash
pip install dataknobs-xization
```

## Overview

The Xization package specializes in text preprocessing and includes:

- **Tokenization**: Advanced character-based and feature-driven tokenization
- **Normalization**: Text normalization with camelCase expansion, symbol handling
- **Masking**: Character-level masking and feature extraction
- **Authorities**: Text authority and lexicon management
- **Annotations**: Text annotation utilities

## Package Structure

```
dataknobs-xization/
├── src/
│   └── dataknobs_xization/
│       ├── __init__.py
│       ├── annotations.py
│       ├── authorities.py
│       ├── lexicon.py
│       ├── masking_tokenizer.py
│       └── normalize.py
└── tests/
```

## Quick Start

### Text Normalization

```python
from dataknobs_xization import normalize

# Basic normalization
text = "CamelCaseText with SYMBOLS!"
normalized = normalize.basic_normalization_fn(
    text,
    lowercase=True,
    expand_camelcase=True,
    drop_non_embedded_symbols=True
)
print(normalized)  # "camel case text with symbols"

# Get lexical variations
variations = normalize.get_lexical_variations(
    "XML-HTTP-Request",
    expand_camelcase=True,
    do_hyphen_expansion=True,
    do_hyphen_split=True
)
print(variations)  # {"XML-HTTP-Request", "XML HTTP Request", "XML", "HTTP", "Request", ...}
```

### Character-Based Tokenization

```python
from dataknobs_xization import TextFeatures
from dataknobs_structures import Text, TextMetaData

# Create document
metadata = TextMetaData(text_id=1, text_label="sample")
doc = Text("Hello CamelCase123 world!", metadata)

# Extract features
features = TextFeatures(
    doc,
    split_camelcase=True,
    mark_alpha=True,
    mark_digit=True
)

# Get tokens
tokens = features.get_tokens()
for token in tokens:
    print(f"Token: '{token.token_text}' at position {token.token_pos}")
```

### Masking and Features

```python
from dataknobs_xization import CharacterFeatures, TextFeatures

# Create text features with masking
text = "Hello123World"
features = TextFeatures(
    text,
    split_camelcase=True,
    mark_upper=True,
    mark_lower=True,
    mark_digit=True
)

# Access character dataframe
cdf = features.cdf
print(cdf.head())  # Shows character-level features

# Tokenize with normalization
def normalize_token(token_text):
    return token_text.lower()

tokens = features.get_tokens(normalize_fn=normalize_token)
for token in tokens:
    print(f"Original: '{token.token_text}', Normalized: '{token.norm_text}'")
```

## Core Classes

### TextFeatures

Character-level feature extraction with tokenization support.

```python
from dataknobs_xization import TextFeatures

features = TextFeatures(
    doctext="ProcessHTMLData",
    split_camelcase=True,    # Split on camelCase boundaries
    mark_alpha=True,         # Mark alphabetic characters
    mark_digit=True,         # Mark numeric characters
    mark_upper=True,         # Mark uppercase characters
    mark_lower=True          # Mark lowercase characters
)

# Get character features as DataFrame
df = features.cdf
print(df.columns)  # Shows feature columns

# Extract tokens
tokens = features.get_tokens()
print([token.token_text for token in tokens])  # ["Process", "HTML", "Data"]
```

### CharacterFeatures (Abstract Base)

Base class for character-level text analysis.

```python
from dataknobs_xization import CharacterFeatures

# TextFeatures inherits from CharacterFeatures
# Provides common interface for character-based processing
```

### Token Classes

Individual token representation with position and feature information.

```python
# Tokens are returned by TextFeatures.get_tokens()
for token in tokens:
    print(f"Text: {token.token_text}")
    print(f"Position: {token.start_pos}-{token.end_pos}")
    print(f"Length: {token.len}")
    print(f"Normalized: {token.norm_text}")
```

## Normalization Functions

### Basic Normalization

```python
from dataknobs_xization.normalize import basic_normalization_fn

text = "XMLHttpRequest"
normalized = basic_normalization_fn(
    text,
    lowercase=True,
    expand_camelcase=True,
    squash_whitespace=True
)
print(normalized)  # "xml http request"
```

### Camel Case Expansion

```python
from dataknobs_xization.normalize import expand_camelcase_fn

text = "parseXMLData"
expanded = expand_camelcase_fn(text)
print(expanded)  # "parse XML Data"
```

### Symbol Handling

```python
from dataknobs_xization.normalize import (
    drop_non_embedded_symbols_fn,
    drop_embedded_symbols_fn
)

# Remove symbols at word boundaries
text = "!hello@world#"
clean = drop_non_embedded_symbols_fn(text)
print(clean)  # "hello@world"

# Remove symbols within words
text = "hello@world"
clean = drop_embedded_symbols_fn(text)
print(clean)  # "helloworld"
```

### Lexical Variations

```python
from dataknobs_xization.normalize import get_lexical_variations

variations = get_lexical_variations(
    "co-worker",
    include_self=True,
    expand_camelcase=False,
    do_hyphen_expansion=True,
    do_hyphen_split=True
)
print(variations)  # {"co-worker", "co worker", "coworker", "co", "worker"}
```

## Advanced Features

### Hyphen and Slash Expansion

```python
from dataknobs_xization.normalize import get_hyphen_slash_expansions_fn

# Generate variations for hyphenated/slashed terms
variations = get_hyphen_slash_expansions_fn(
    "client-server",
    subs=["-", " ", ""],
    add_self=True,
    do_split=True
)
print(variations)  # {"client-server", "client server", "clientserver", "client", "server"}
```

### Number to English Conversion

```python
from dataknobs_xization.normalize import int_to_en, year_variations_fn

# Convert integers to English
print(int_to_en(42))    # "forty two"
print(int_to_en(1999))  # "one thousand nine hundred and ninety nine"

# Generate year variations
years = year_variations_fn(1999)
print(years)  # {"1999", "one thousand nine hundred and ninety nine", "nineteen ninety nine", ...}
```

### Smart Quote Handling

```python
from dataknobs_xization.normalize import replace_smart_quotes_fn

text = ""Hello World" with 'smart quotes'"
cleaned = replace_smart_quotes_fn(text)
print(cleaned)  # "\"Hello World\" with 'smart quotes'"
```

## Integration Examples

### With Document Structures

```python
from dataknobs_structures import Text, TextMetaData
from dataknobs_xization import TextFeatures, normalize

# Create document
metadata = TextMetaData(text_id="doc_001", text_label="article")
doc = Text("parseXMLHttpRequest", metadata)

# Normalize content
normalized_text = normalize.basic_normalization_fn(
    doc.text,
    expand_camelcase=True,
    lowercase=True
)

# Update document
doc._text = normalized_text  # Note: direct access for example

# Extract features
features = TextFeatures(doc, split_camelcase=True)
tokens = features.get_tokens()
```

### With JSON Processing

```python
from dataknobs_utils import json_utils
from dataknobs_xization import normalize

# Process JSON text fields
def normalize_json_text(item, path):
    if isinstance(item, str) and len(item) > 0:
        normalized = normalize.basic_normalization_fn(item)
        print(f"Path: {path}, Normalized: {normalized}")

# Apply to JSON stream
json_utils.stream_json_data("data.json", normalize_json_text)
```

## Tokenization Patterns

### CamelCase Tokenization

```python
from dataknobs_xization import TextFeatures

# Split on camelCase boundaries
text = "XMLHttpRequest"
features = TextFeatures(text, split_camelcase=True)
tokens = features.get_tokens()

print([t.token_text for t in tokens])  # ["XML", "Http", "Request"]
```

### Symbol-Aware Tokenization

```python
# Handle embedded symbols differently
text = "client-server@domain.com"
features = TextFeatures(text, split_camelcase=False)
tokens = features.get_tokens()

# Tokens respect symbol boundaries
print([t.token_text for t in tokens])  # ["client", "server", "domain", "com"]
```

### Custom Normalization

```python
def custom_normalize(text):
    """Custom normalization function"""
    # Apply multiple normalization steps
    text = normalize.expand_camelcase_fn(text)
    text = normalize.basic_normalization_fn(
        text,
        lowercase=True,
        drop_embedded_symbols=True,
        squash_whitespace=True
    )
    return text

# Use with tokenizer
features = TextFeatures("parseXML-Data")
tokens = features.get_tokens(normalize_fn=custom_normalize)
```

## Error Handling

```python
from dataknobs_xization import TextFeatures, normalize

try:
    # Handle empty or None text
    text = ""
    if not text:
        text = "default text"
    
    features = TextFeatures(text)
    tokens = features.get_tokens()
    
except Exception as e:
    print(f"Tokenization error: {e}")

try:
    # Handle normalization errors
    result = normalize.basic_normalization_fn(None)
except (TypeError, AttributeError) as e:
    print(f"Normalization error: {e}")
    result = ""
```

## Performance Tips

1. **Reuse TextFeatures**: Create once, tokenize multiple times with different normalization
2. **Batch Processing**: Process similar texts together
3. **Selective Features**: Only enable needed character features
4. **Normalization Caching**: Cache normalization results for repeated text

## Configuration Options

### Character Features

```python
features = TextFeatures(
    text,
    split_camelcase=True,     # Enable camelCase splitting
    mark_alpha=True,          # Mark alphabetic characters
    mark_digit=True,          # Mark numeric characters
    mark_upper=True,          # Mark uppercase characters
    mark_lower=True,          # Mark lowercase characters
    emoji_data=None           # Optional emoji processing
)
```

### Normalization Options

```python
normalized = normalize.basic_normalization_fn(
    text,
    lowercase=True,                    # Convert to lowercase
    expand_camelcase=True,            # Expand camelCase
    simplify_quote_chars=True,        # Replace smart quotes
    drop_non_embedded_symbols=False,  # Remove boundary symbols
    spacify_embedded_symbols=False,   # Replace embedded symbols with spaces
    drop_embedded_symbols=False,      # Remove embedded symbols
    squash_whitespace=False,          # Collapse whitespace
    do_all=False                      # Apply all options
)
```

## API Reference

For complete API documentation, see the [Xization API Reference](api.md).

## Module Documentation

- [Tokenization](tokenization.md) - Character-based tokenization
- [Normalization](normalization.md) - Text normalization functions  
- [Masking](masking.md) - Character masking and features

## See Also

- [Text Normalization Examples](../../examples/text-normalization.md)
- [Integration with Structures](../structures/index.md)
- [Utils Package](../utils/index.md)