# Normalization API Documentation

The `normalize` module provides text normalization functions for cleaning, standardizing, and preprocessing text data.

## Overview

Text normalization includes:

- Whitespace handling and cleanup
- Case conversion and standardization
- Symbol and punctuation processing
- CamelCase expansion
- Hyphen and slash expansion
- Lexical variation generation
- Parenthetical expression handling
- Ampersand expansion

## Regular Expressions

The module provides pre-compiled regular expressions for common text patterns:

### SQUASH_WS_RE
```python
SQUASH_WS_RE = re.compile(r"\s+")
```
Collapses consecutive whitespace to a single space.

### ALL_SYMBOLS_RE
```python
ALL_SYMBOLS_RE = re.compile(r"[^\w\s]+")
```
Identifies strings with any symbols (non-word, non-space characters).

### CAMELCASE_LU_RE
```python
CAMELCASE_LU_RE = re.compile(r"([a-z]+)([A-Z])")
```
Splits between consecutive lower and upper case characters.

### CAMELCASE_UL_RE
```python
CAMELCASE_UL_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
```
Splits between consecutive upper case and upper-lower case characters.

### Symbol Handling Patterns
```python
# Non-embedded symbols (without word char on both sides)
NON_EMBEDDED_WORD_SYMS_RE = re.compile(r"((?<!\w)[^\w\s]+)|([^\w\s]+(?!\w))")

# Embedded symbols (with word chars on both sides)
EMBEDDED_SYMS_RE = re.compile(r"(?<=\w)[^\w\s]+(?=\w)")
```

### Delimiter Patterns
```python
# Hyphen and/or slash between word characters
HYPHEN_SLASH_RE = re.compile(r"(?<=\w)[\-\/ ](?=\w)")

# Hyphen only between word characters
HYPHEN_ONLY_RE = re.compile(r"(?<=\w)[\- ](?=\w)")

# Slash only between word characters
SLASH_ONLY_RE = re.compile(r"(?<=\w)\/(?=\w)")
```

### Other Patterns
```python
# Parenthetical expressions
PARENTHETICAL_RE = re.compile(r"\(.*\)")

# Ampersand with optional whitespace
AMPERSAND_RE = re.compile(r"\s*\&\s*")
```

## Core Functions

### expand_camelcase_fn()
```python
def expand_camelcase_fn(text: str) -> str
```

Expands both "lU" and "UUl" camelcasing patterns to add spaces.

**Parameters:**
- `text` (str): Input text with camelCase patterns

**Returns:** Text with expanded camelCase (spaces added)

**Example:**
```python
from dataknobs_xization import normalize

# Expand camelCase
text1 = "firstName"
result1 = normalize.expand_camelcase_fn(text1)
print(result1)  # "first Name"

text2 = "XMLParser"
result2 = normalize.expand_camelcase_fn(text2)
print(result2)  # "XML Parser"

text3 = "iPhone"
result3 = normalize.expand_camelcase_fn(text3)
print(result3)  # "i Phone"

text4 = "getUserID"
result4 = normalize.expand_camelcase_fn(text4)
print(result4)  # "get User ID"
```

### drop_non_embedded_symbols_fn()
```python
def drop_non_embedded_symbols_fn(text: str, repl: str = "") -> str
```

Removes symbols that are not embedded within word characters.

**Parameters:**
- `text` (str): Input text
- `repl` (str, default=""): Replacement string for dropped symbols

**Returns:** Text with non-embedded symbols removed

**Example:**
```python
# Remove leading/trailing punctuation
text = "!Hello world?"
result = normalize.drop_non_embedded_symbols_fn(text)
print(result)  # "Hello world"

# Keep embedded symbols
text2 = "user@domain.com"
result2 = normalize.drop_non_embedded_symbols_fn(text2)
print(result2)  # "user@domain.com" (@ and . are embedded)

# Custom replacement
text3 = "*important*"
result3 = normalize.drop_non_embedded_symbols_fn(text3, " ")
print(result3)  # " important "
```

### drop_embedded_symbols_fn()
```python
def drop_embedded_symbols_fn(text: str, repl: str = "") -> str
```

Removes symbols that are embedded within word characters.

**Parameters:**
- `text` (str): Input text
- `repl` (str, default=""): Replacement string for dropped symbols

**Returns:** Text with embedded symbols removed

**Example:**
```python
# Remove embedded punctuation
text = "user@domain.com"
result = normalize.drop_embedded_symbols_fn(text)
print(result)  # "userdomaincom"

# With replacement
text2 = "first-name"
result2 = normalize.drop_embedded_symbols_fn(text2, " ")
print(result2)  # "first name"

# Multiple embedded symbols
text3 = "a@b#c$d"
result3 = normalize.drop_embedded_symbols_fn(text3)
print(result3)  # "abcd"
```

### get_hyphen_slash_expansions_fn()
```python
def get_hyphen_slash_expansions_fn(
    text: str,
    subs: List[str] = ("-", " ", ""),
    add_self: bool = True,
    do_split: bool = True,
    min_split_token_len: int = 2,
    hyphen_slash_re=HYPHEN_SLASH_RE,
) -> Set[str]
```

Generate variations of hyphenated or slash-separated text.

**Parameters:**
- `text` (str): Input text with potential hyphens/slashes
- `subs` (List[str]): Characters to substitute for delimiters
- `add_self` (bool): Include original text in results
- `do_split` (bool): Include individual tokens
- `min_split_token_len` (int): Minimum token length for splitting
- `hyphen_slash_re` (Pattern): Regex pattern for matching delimiters

**Returns:** Set of text variations

**Example:**
```python
# Generate hyphen variations
text = "multi-word-phrase"
variations = normalize.get_hyphen_slash_expansions_fn(text)
print(variations)
# {'multi-word-phrase', 'multi word phrase', 'multiwordphrase', 'multi', 'word', 'phrase'}

# Custom substitutions
text2 = "data/science"
variations2 = normalize.get_hyphen_slash_expansions_fn(
    text2, subs=[" ", "_", ""], do_split=False
)
print(variations2)
# {'data/science', 'data science', 'data_science', 'datascience'}

# Without original text
text3 = "machine-learning"
variations3 = normalize.get_hyphen_slash_expansions_fn(
    text3, add_self=False, do_split=False
)
print(variations3)
# {'machine learning', 'machinelearning'}
```

### drop_parentheticals_fn()
```python
def drop_parentheticals_fn(text: str) -> str
```

Removes parenthetical expressions from text.

**Parameters:**
- `text` (str): Input text

**Returns:** Text with parentheticals removed

**Example:**
```python
# Remove parenthetical information
text = "Python (programming language) is popular"
result = normalize.drop_parentheticals_fn(text)
print(result)  # "Python  is popular"

# Multiple parentheticals
text2 = "AI (Artificial Intelligence) and ML (Machine Learning)"
result2 = normalize.drop_parentheticals_fn(text2)
print(result2)  # "AI  and ML "
```

### expand_ampersand_fn()
```python
def expand_ampersand_fn(text: str) -> str
```

Replaces ampersands with " and ".

**Parameters:**
- `text` (str): Input text

**Returns:** Text with ampersands expanded

**Example:**
```python
# Expand ampersands
text = "Research & Development"
result = normalize.expand_ampersand_fn(text)
print(result)  # "Research and Development"

# Multiple ampersands
text2 = "A&B&C"
result2 = normalize.expand_ampersand_fn(text2)
print(result2)  # "A and B and C"

# Handles whitespace
text3 = "cats&dogs"
result3 = normalize.expand_ampersand_fn(text3)
print(result3)  # "cats and dogs"
```

## Advanced Functions

### get_lexical_variations()
```python
def get_lexical_variations(
    text: str,
    include_self: bool = True,
    expand_camelcase: bool = True,
    drop_non_embedded_symbols: bool = True,
    drop_embedded_symbols: bool = True,
    spacify_embedded_symbols: bool = False,
    do_hyphen_expansion: bool = True,
    hyphen_subs: List[str] = (" ", ""),
    do_hyphen_split: bool = True,
    min_hyphen_split_token_len=2,
    do_slash_expansion: bool = True,
    slash_subs: List[str] = (" ", " or "),
    do_slash_split: bool = True,
    min_slash_split_token_len: int = 1,
    drop_parentheticals: bool = True,
    expand_ampersands: bool = True,
    add_eng_plurals: bool = True,
) -> Set[str]
```

Generate comprehensive lexical variations of input text using multiple normalization techniques.

**Parameters:** (extensive list of boolean flags and configuration options)

**Returns:** Set of text variations

**Example:**
```python
# Generate comprehensive variations
text = "multi-platform/cross-browser (compatible)"
variations = normalize.get_lexical_variations(text)
print(f"Generated {len(variations)} variations:")
for var in sorted(variations):
    print(f"  {var}")

# Custom configuration
text2 = "JavaScript"
variations2 = normalize.get_lexical_variations(
    text2,
    expand_camelcase=True,
    drop_non_embedded_symbols=False,
    add_eng_plurals=True
)
print(variations2)
# {'JavaScript', 'Java Script', 'JavaScripts', 'Java Scripts'}
```

### basic_normalization_fn()
```python
def basic_normalization_fn(text: str) -> str
```

Applies common normalization steps to text.

**Standard Operations:**
- Lowercase conversion
- Whitespace squashing
- Basic punctuation handling
- Trimming

**Example:**
```python
# Basic normalization
text = "  Hello,    WORLD!  \n\t How   are you?  "
result = normalize.basic_normalization_fn(text)
print(repr(result))  # 'hello, world! how are you?'
```

## Usage Patterns

### Text Cleaning Pipeline
```python
from dataknobs_xization import normalize

def clean_text_pipeline(text: str) -> Dict[str, str]:
    """Comprehensive text cleaning pipeline."""
    results = {"original": text}
    
    # Step 1: Expand camelCase
    step1 = normalize.expand_camelcase_fn(text)
    results["camelcase_expanded"] = step1
    
    # Step 2: Expand ampersands
    step2 = normalize.expand_ampersand_fn(step1)
    results["ampersands_expanded"] = step2
    
    # Step 3: Drop parentheticals
    step3 = normalize.drop_parentheticals_fn(step2)
    results["parentheticals_dropped"] = step3
    
    # Step 4: Handle embedded symbols
    step4 = normalize.drop_embedded_symbols_fn(step3, " ")
    results["embedded_symbols_replaced"] = step4
    
    # Step 5: Basic normalization
    step5 = normalize.basic_normalization_fn(step4)
    results["final_normalized"] = step5
    
    return results

# Example usage
text = "getUserName() & validateInput (required)"
results = clean_text_pipeline(text)
for step, result in results.items():
    print(f"{step}: {result}")
```

### Lexical Variation Generation
```python
from dataknobs_xization import normalize
from collections import Counter

def generate_search_terms(query: str) -> List[str]:
    """Generate search term variations for better matching."""
    # Get all variations
    variations = normalize.get_lexical_variations(
        query,
        expand_camelcase=True,
        do_hyphen_expansion=True,
        do_slash_expansion=True,
        add_eng_plurals=True
    )
    
    # Filter and rank by relevance
    filtered = []
    for var in variations:
        # Skip very short or very long variations
        if 2 <= len(var.split()) <= 10:
            filtered.append(var)
    
    # Sort by length and complexity (prefer simpler terms)
    filtered.sort(key=lambda x: (len(x.split()), len(x)))
    
    return filtered[:20]  # Return top 20 variations

# Example
query = "machine-learning/deep-learning"
search_terms = generate_search_terms(query)
print("Generated search terms:")
for i, term in enumerate(search_terms, 1):
    print(f"{i:2d}. {term}")
```

### Domain-Specific Normalization
```python
from dataknobs_xization import normalize
import re

class TechnicalTextNormalizer:
    """Specialized normalizer for technical text."""
    
    def __init__(self):
        self.tech_patterns = {
            'version_numbers': re.compile(r'v?\d+\.\d+(\.\d+)?'),
            'file_extensions': re.compile(r'\.[a-zA-Z0-9]+$'),
            'urls': re.compile(r'https?://[^\s]+'),
            'code_snippets': re.compile(r'`[^`]+`')
        }
    
    def normalize_technical(self, text: str) -> str:
        """Normalize technical text while preserving important patterns."""
        # Store special patterns
        preserved = {}
        placeholder_count = 0
        
        for pattern_name, pattern in self.tech_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                placeholder = f"__PRESERVE_{placeholder_count}__"
                preserved[placeholder] = match
                text = text.replace(match, placeholder, 1)
                placeholder_count += 1
        
        # Apply standard normalization
        normalized = normalize.basic_normalization_fn(text)
        
        # Expand technical camelCase
        normalized = normalize.expand_camelcase_fn(normalized)
        
        # Handle technical symbols differently
        normalized = normalize.drop_non_embedded_symbols_fn(normalized, " ")
        
        # Restore preserved patterns
        for placeholder, original in preserved.items():
            normalized = normalized.replace(placeholder, original)
        
        # Final cleanup
        normalized = normalize.SQUASH_WS_RE.sub(" ", normalized).strip()
        
        return normalized

# Usage
normalizer = TechnicalTextNormalizer()
tech_text = "Check out myLibrary v2.1.3 at https://github.com/user/repo and run `npm install`"
result = normalizer.normalize_technical(tech_text)
print(f"Original: {tech_text}")
print(f"Normalized: {result}")
```

### Batch Text Processing
```python
from dataknobs_xization import normalize
from dataknobs_utils import file_utils
from concurrent.futures import ThreadPoolExecutor
import json

def normalize_document(doc_data: dict) -> dict:
    """Normalize a single document."""
    text = doc_data.get('content', '')
    
    # Generate variations
    variations = normalize.get_lexical_variations(
        text,
        expand_camelcase=True,
        do_hyphen_expansion=True,
        drop_parentheticals=True
    )
    
    # Basic normalization
    normalized = normalize.basic_normalization_fn(text)
    
    return {
        **doc_data,
        'normalized_content': normalized,
        'variations': list(variations),
        'variation_count': len(variations)
    }

def batch_normalize_documents(input_dir: str, output_dir: str):
    """Process multiple documents in parallel."""
    documents = []
    
    # Load all documents
    for filepath in file_utils.filepath_generator(input_dir):
        if filepath.endswith('.json'):
            for line in file_utils.fileline_generator(filepath):
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError:
                    continue
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        normalized_docs = list(executor.map(normalize_document, documents))
    
    # Save results
    output_lines = []
    for doc in normalized_docs:
        output_lines.append(json.dumps(doc))
    
    file_utils.write_lines(f"{output_dir}/normalized_documents.jsonl", output_lines)
    
    print(f"Processed {len(normalized_docs)} documents")
    print(f"Average variations per document: {sum(d['variation_count'] for d in normalized_docs) / len(normalized_docs):.1f}")

# Usage
batch_normalize_documents('/input/docs', '/output/normalized')
```

## Error Handling

```python
from dataknobs_xization import normalize

def safe_normalize(text: str, method: str = "basic") -> str:
    """Safely normalize text with error handling."""
    if not text or not isinstance(text, str):
        return ""
    
    try:
        if method == "basic":
            return normalize.basic_normalization_fn(text)
        elif method == "camelcase":
            return normalize.expand_camelcase_fn(text)
        elif method == "comprehensive":
            variations = normalize.get_lexical_variations(text)
            return min(variations, key=len)  # Return shortest variation
        else:
            return text
            
    except Exception as e:
        print(f"Normalization failed for '{text[:50]}...': {e}")
        return text  # Return original on error

# Safe usage
result = safe_normalize("someWeirdText&Symbols(here)", "comprehensive")
print(result)
```

## Performance Considerations

- Regular expressions are pre-compiled for efficiency
- `get_lexical_variations()` can generate many variations - use selectively
- Consider caching results for frequently processed text
- Use appropriate batch sizes for parallel processing
- Monitor memory usage with large variation sets

## Integration Examples

### With Elasticsearch
```python
from dataknobs_xization import normalize
from dataknobs_utils import elasticsearch_utils

def create_searchable_document(title: str, content: str) -> dict:
    """Create document with normalized search fields."""
    # Generate title variations for better matching
    title_variations = normalize.get_lexical_variations(
        title, do_hyphen_expansion=True, expand_camelcase=True
    )
    
    # Normalize content
    normalized_content = normalize.basic_normalization_fn(content)
    
    # Expand technical terms
    expanded_content = normalize.expand_camelcase_fn(content)
    expanded_content = normalize.expand_ampersand_fn(expanded_content)
    
    return {
        'title': title,
        'content': content,
        'title_variations': list(title_variations),
        'normalized_content': normalized_content,
        'expanded_content': expanded_content,
        'searchable_title': ' '.join(title_variations),
        'searchable_content': f"{normalized_content} {expanded_content}"
    }

# Usage with Elasticsearch
doc = create_searchable_document(
    "JavaScript & Node.js",
    "Learn JavaScript (programming language) and Node.js development"
)

# Index with enhanced searchability
query = elasticsearch_utils.build_field_query_dict(
    ['searchable_title', 'searchable_content'],
    'java script nodejs'
)
```

### With File Processing
```python
from dataknobs_xization import normalize
from dataknobs_utils import file_utils

def normalize_text_files(input_dir: str, output_dir: str):
    """Normalize all text files in directory."""
    for filepath in file_utils.filepath_generator(input_dir):
        if filepath.endswith('.txt'):
            # Read file
            lines = list(file_utils.fileline_generator(filepath))
            
            # Normalize each line
            normalized_lines = []
            for line in lines:
                normalized = normalize.basic_normalization_fn(line)
                if normalized.strip():  # Skip empty lines
                    normalized_lines.append(normalized)
            
            # Save normalized version
            basename = file_utils.get_basename(filepath)
            output_path = f"{output_dir}/normalized_{basename}"
            file_utils.write_lines(output_path, normalized_lines)

# Process directory
normalize_text_files('/raw/text', '/processed/text')
```

The normalization module provides comprehensive text preprocessing capabilities that work seamlessly with other dataknobs components for complete text processing pipelines.