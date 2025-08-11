# Text Normalization Examples

This guide demonstrates text normalization and processing using the `dataknobs-xization` package.

## Basic Normalization

### Simple Text Cleaning

```python
from dataknobs_xization import normalize

# Basic normalization
text = "  Hello   World!   "
normalized = normalize.normalize_whitespace_fn(text)
print(f"Original: '{text}'")
print(f"Normalized: '{normalized}'")
# Output: 'Hello World!'
```

### Expanding CamelCase

```python
from dataknobs_xization import normalize

# Expand camelCase text
camel_text = "getUserNameAndEmail"
expanded = normalize.expand_camelcase_fn(camel_text)
print(f"Original: {camel_text}")
print(f"Expanded: {expanded}")
# Output: 'get User Name And Email'

# Works with acronyms
acronym_text = "XMLHttpRequest"
expanded = normalize.expand_camelcase_fn(acronym_text)
print(f"Expanded: {expanded}")
# Output: 'XML Http Request'
```

### Expanding Ampersands

```python
from dataknobs_xization import normalize

# Expand ampersands
text = "Research & Development"
expanded = normalize.expand_ampersand_fn(text)
print(f"Original: {text}")
print(f"Expanded: {expanded}")
# Output: 'Research and Development'

# Multiple ampersands
text = "A & B & C"
expanded = normalize.expand_ampersand_fn(text)
print(f"Expanded: {expanded}")
# Output: 'A and B and C'
```

## Combined Normalizations

### Full Text Normalization Pipeline

```python
from dataknobs_xization import normalize

def full_normalization(text):
    """Apply all normalization steps."""
    # Step 1: Expand camelCase
    text = normalize.expand_camelcase_fn(text)
    
    # Step 2: Expand ampersands
    text = normalize.expand_ampersand_fn(text)
    
    # Step 3: Normalize whitespace
    text = normalize.normalize_whitespace_fn(text)
    
    # Step 4: Convert to lowercase (optional)
    text = text.lower()
    
    return text

# Example usage
code_text = "getUserData&ProcessInput"
normalized = full_normalization(code_text)
print(f"Original: {code_text}")
print(f"Normalized: {normalized}")
# Output: 'get user data and process input'
```

### Custom Normalization Function

```python
from dataknobs_xization import normalize
import re

def custom_normalize(text):
    """Custom normalization with additional rules."""
    # Apply basic normalizations
    text = normalize.basic_normalization_fn(text)
    
    # Custom: Replace underscores with spaces
    text = text.replace('_', ' ')
    
    # Custom: Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Custom: Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Clean up whitespace
    text = normalize.normalize_whitespace_fn(text)
    
    return text

# Example
text = "user_data_123 & special@chars!"
normalized = custom_normalize(text)
print(f"Original: {text}")
print(f"Normalized: {normalized}")
```

## Tokenization Examples

### Basic Tokenization

```python
from dataknobs_xization.masking_tokenizer import TextFeatures

# Create text features for tokenization
text = "Hello World! How are you?"
features = TextFeatures(text, split_camelcase=True)

# Get tokens
tokens = features.get_tokens()

print("Tokens:")
for token in tokens:
    print(f"  '{token.token_text}' at position {token.start_pos}-{token.end_pos}")
```

### CamelCase-Aware Tokenization

```python
from dataknobs_xization.masking_tokenizer import TextFeatures

# Tokenize with camelCase splitting
code_text = "getUserNameById"
features = TextFeatures(code_text, split_camelcase=True)
tokens = features.get_tokens()

print("CamelCase tokens:")
for token in tokens:
    print(f"  '{token.token_text}'")
# Output: 'get', 'User', 'Name', 'By', 'Id'

# Without camelCase splitting
features_no_split = TextFeatures(code_text, split_camelcase=False)
tokens_no_split = features_no_split.get_tokens()

print("\nWithout splitting:")
for token in tokens_no_split:
    print(f"  '{token.token_text}'")
# Output: 'getUserNameById'
```

### Tokenization with Normalization

```python
from dataknobs_xization.masking_tokenizer import TextFeatures
from dataknobs_xization import normalize

def tokenize_and_normalize(text):
    """Tokenize and normalize each token."""
    features = TextFeatures(text, split_camelcase=True)
    
    # Define normalization function
    def normalize_token(token_text):
        return token_text.lower()
    
    # Get normalized tokens
    tokens = features.get_tokens(normalize_fn=normalize_token)
    
    return [token.norm_text for token in tokens]

# Example
text = "GetUserName AND ProcessData"
normalized_tokens = tokenize_and_normalize(text)
print(f"Original: {text}")
print(f"Normalized tokens: {normalized_tokens}")
```

## Pattern-Based Normalization

### Email Address Normalization

```python
import re
from dataknobs_xization import normalize

def normalize_email(text):
    """Normalize email addresses in text."""
    # Pattern for email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    def normalize_single_email(match):
        email = match.group(0)
        # Normalize to lowercase
        return email.lower()
    
    # Replace all emails with normalized version
    normalized = re.sub(email_pattern, normalize_single_email, text)
    
    return normalized

# Example
text = "Contact John.Doe@EXAMPLE.COM or Jane.Smith@Company.ORG"
normalized = normalize_email(text)
print(f"Original: {text}")
print(f"Normalized: {normalized}")
```

### URL Normalization

```python
import re
from urllib.parse import urlparse, urlunparse

def normalize_urls(text):
    """Normalize URLs in text."""
    url_pattern = r'https?://[^\s]+'
    
    def normalize_single_url(match):
        url = match.group(0)
        parsed = urlparse(url.lower())
        # Remove www. prefix if present
        netloc = parsed.netloc
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        # Rebuild URL
        normalized = urlunparse((
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))
        return normalized
    
    return re.sub(url_pattern, normalize_single_url, text)

# Example
text = "Visit HTTPS://WWW.EXAMPLE.COM/Page or http://Another-Site.org"
normalized = normalize_urls(text)
print(f"Original: {text}")
print(f"Normalized: {normalized}")
```

## Language-Specific Normalization

### Code Normalization

```python
from dataknobs_xization import normalize
import re

def normalize_code(code):
    """Normalize programming code for analysis."""
    # Remove comments (simple example for Python/Java style)
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)  # Single-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Multi-line comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # Python comments
    
    # Expand camelCase
    code = normalize.expand_camelcase_fn(code)
    
    # Normalize operators
    code = code.replace('&&', ' and ')
    code = code.replace('||', ' or ')
    code = code.replace('!=', ' not equal ')
    code = code.replace('==', ' equal ')
    
    # Normalize whitespace
    code = normalize.normalize_whitespace_fn(code)
    
    return code

# Example
code_snippet = """
// Get user data
function getUserData() {
    if (userName != null && userAge >= 18) {
        return true;  // Valid user
    }
}
"""

normalized = normalize_code(code_snippet)
print("Normalized code:")
print(normalized)
```

### Natural Language Processing

```python
from dataknobs_xization import normalize
import re

def normalize_for_nlp(text):
    """Normalize text for NLP processing."""
    # Convert to lowercase
    text = text.lower()
    
    # Expand contractions
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace
    text = normalize.normalize_whitespace_fn(text)
    
    return text

# Example
text = "I won't go there! She'll be happy, won't she?"
normalized = normalize_for_nlp(text)
print(f"Original: {text}")
print(f"Normalized: {normalized}")
```

## Batch Processing

### Normalizing Multiple Documents

```python
from dataknobs_xization import normalize
from dataknobs_structures import Document

def batch_normalize(documents):
    """Normalize a batch of documents."""
    normalized_docs = []
    
    for doc in documents:
        normalized_text = normalize.basic_normalization_fn(doc.text)
        normalized_doc = Document(
            normalized_text,
            metadata={**doc.metadata, "normalized": True}
        )
        normalized_docs.append(normalized_doc)
    
    return normalized_docs

# Example
documents = [
    Document("getUserName", metadata={"id": 1}),
    Document("processData&SaveResults", metadata={"id": 2}),
    Document("XMLHttpRequest", metadata={"id": 3})
]

normalized = batch_normalize(documents)
for doc in normalized:
    print(f"Doc {doc.metadata['id']}: {doc.text}")
```

### Parallel Normalization

```python
from dataknobs_xization import normalize
from concurrent.futures import ThreadPoolExecutor
import time

def normalize_large_text(text):
    """Normalize large text with all steps."""
    text = normalize.expand_camelcase_fn(text)
    text = normalize.expand_ampersand_fn(text)
    text = normalize.normalize_whitespace_fn(text)
    return text

def parallel_normalize(texts, max_workers=4):
    """Normalize multiple texts in parallel."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(normalize_large_text, texts))
    return results

# Example
texts = [
    "getUserData&ProcessInput" * 100,  # Large text
    "XMLHttpRequest&AjaxCall" * 100,
    "validateUserInput&SaveData" * 100
]

start = time.time()
normalized = parallel_normalize(texts)
print(f"Normalized {len(texts)} texts in {time.time() - start:.2f} seconds")
```

## Custom Token Processing

### Token-Level Normalization

```python
from dataknobs_xization.masking_tokenizer import TextFeatures

class CustomTokenProcessor:
    """Custom token processor with specific rules."""
    
    def __init__(self):
        self.abbreviations = {
            "Dr": "Doctor",
            "Mr": "Mister",
            "Mrs": "Missus",
            "Inc": "Incorporated"
        }
    
    def process(self, text):
        """Process text with custom token rules."""
        features = TextFeatures(text, split_camelcase=True)
        
        def custom_normalize(token_text):
            # Check for abbreviations
            if token_text in self.abbreviations:
                return self.abbreviations[token_text]
            # Default: lowercase
            return token_text.lower()
        
        tokens = features.get_tokens(normalize_fn=custom_normalize)
        
        # Reconstruct normalized text
        result = []
        for token in tokens:
            result.append(token.norm_text)
            if token.post_delims:
                result.append(token.post_delims)
        
        return ''.join(result)

# Example
processor = CustomTokenProcessor()
text = "Dr Smith from TechCorp Inc"
normalized = processor.process(text)
print(f"Original: {text}")
print(f"Processed: {normalized}")
```

## Best Practices

1. **Choose appropriate normalization level**: Don't over-normalize if you need to preserve information
2. **Consider context**: Different domains require different normalization rules
3. **Preserve original**: Always keep the original text for reference
4. **Test edge cases**: Include special characters, unicode, and edge cases in testing
5. **Performance optimization**: Use batch processing for large datasets
6. **Validation**: Validate normalized output to ensure quality

## Related Examples

- [Document Processing](document-processing.md)
- [Basic Tree Operations](basic-tree.md)
- [Elasticsearch Integration](elasticsearch-integration.md)