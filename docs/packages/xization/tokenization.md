# Tokenization API Documentation

The tokenization functionality in the `dataknobs_xization` package provides advanced text tokenization capabilities with character-level features and masking support.

## Overview

The tokenization system includes:

- Character-level feature extraction
- Text-level feature analysis
- Masking tokenization for privacy and data processing
- Integration with document structures
- Emoji and Unicode support

## Core Classes

### CharacterFeatures
```python
class CharacterFeatures(ABC):
    def __init__(
        self, 
        doctext: Union[dk_doc.Text, str], 
        roll_padding: int = 0
    )
```

Abstract base class for character-level text analysis and feature extraction.

**Parameters:**
- `doctext` (Union[dk_doc.Text, str]): The text to tokenize or a Text document with metadata
- `roll_padding` (int, default=0): Number of pad characters added to each end of text

**Properties:**
- `cdf` (pd.DataFrame): Character dataframe with each padded text character as a row
- `doctext` (dk_doc.Text): The document text wrapper
- `text_col` (str): Name of the cdf column holding text characters
- `text` (str): The text string
- `text_id` (Any): The ID of the text

**Example:**
```python
from dataknobs_xization import masking_tokenizer
from dataknobs_structures import document as dk_doc

# Create text document
text_doc = dk_doc.Text("Hello, World! 👋", text_id="greeting")

# Create character features (concrete implementation needed)
class SimpleCharFeatures(masking_tokenizer.CharacterFeatures):
    @property
    def cdf(self):
        import pandas as pd
        chars = list(self.text)
        return pd.DataFrame({
            self.text_col: chars,
            'position': range(len(chars)),
            'is_alpha': [c.isalpha() for c in chars],
            'is_digit': [c.isdigit() for c in chars]
        })

features = SimpleCharFeatures(text_doc)
print(f"Text: {features.text}")
print(f"Text ID: {features.text_id}")
print(features.cdf.head())
```

### TextFeatures
```python
class TextFeatures
```

Class for analyzing text-level features and patterns.

**Features:**
- Character-level analysis
- Token-level analysis  
- Pattern detection
- Statistical measures

**Example:**
```python
# Text-level feature analysis
text = "Hello, World! This is a test sentence."
features = masking_tokenizer.TextFeatures(text)

# Analyze various text properties
print(f"Character count: {len(text)}")
print(f"Word count: {len(text.split())}")
print(f"Contains punctuation: {any(not c.isalnum() and not c.isspace() for c in text)}")
```

## Tokenization Functions

### Character-Level Tokenization
```python
def tokenize_characters(
    text: str,
    include_whitespace: bool = True,
    include_punctuation: bool = True,
    pad_length: int = 0
) -> List[str]
```

Tokenize text into individual characters with optional filtering.

**Parameters:**
- `text` (str): Input text to tokenize
- `include_whitespace` (bool, default=True): Include whitespace characters
- `include_punctuation` (bool, default=True): Include punctuation characters
- `pad_length` (int, default=0): Add padding characters

**Returns:** List of character tokens

**Example:**
```python
from dataknobs_xization import tokenization

text = "Hello, World!"
chars = tokenization.tokenize_characters(text)
print(chars)  # ['H', 'e', 'l', 'l', 'o', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!']

# Without punctuation
chars_no_punct = tokenization.tokenize_characters(text, include_punctuation=False)
print(chars_no_punct)  # ['H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd']
```

### Word-Level Tokenization
```python
def tokenize_words(
    text: str,
    lowercase: bool = False,
    remove_punctuation: bool = True,
    split_pattern: str = None
) -> List[str]
```

Tokenize text into words with various preprocessing options.

**Parameters:**
- `text` (str): Input text to tokenize
- `lowercase` (bool, default=False): Convert to lowercase
- `remove_punctuation` (bool, default=True): Remove punctuation
- `split_pattern` (str, optional): Custom regex pattern for splitting

**Returns:** List of word tokens

**Example:**
```python
text = "Hello, World! How are you today?"
words = tokenization.tokenize_words(text)
print(words)  # ['Hello', 'World', 'How', 'are', 'you', 'today']

# With lowercase
words_lower = tokenization.tokenize_words(text, lowercase=True)
print(words_lower)  # ['hello', 'world', 'how', 'are', 'you', 'today']
```

### Sentence-Level Tokenization
```python
def tokenize_sentences(
    text: str,
    sentence_endings: List[str] = None
) -> List[str]
```

Tokenize text into sentences.

**Parameters:**
- `text` (str): Input text to tokenize
- `sentence_endings` (List[str], optional): Custom sentence ending patterns

**Returns:** List of sentence tokens

**Example:**
```python
text = "Hello world. How are you? I'm fine!"
sentences = tokenization.tokenize_sentences(text)
print(sentences)  # ['Hello world.', 'How are you?', "I'm fine!"]
```

## Feature Extraction

### Character Features
```python
def extract_character_features(text: str) -> pd.DataFrame
```

Extract detailed features for each character in the text.

**Features Extracted:**
- Character type (alphabetic, numeric, punctuation, whitespace)
- Case information (upper, lower, title)
- Unicode category
- Position information
- Emoji detection

**Example:**
```python
import pandas as pd
from dataknobs_xization import tokenization

text = "Hello, 123! 👋"
char_features = tokenization.extract_character_features(text)

print(char_features.head())
# Output includes columns: char, position, is_alpha, is_digit, is_upper, is_lower, etc.
```

### Token Features
```python
def extract_token_features(
    tokens: List[str],
    include_position: bool = True,
    include_length: bool = True,
    include_case: bool = True
) -> pd.DataFrame
```

Extract features for a list of tokens.

**Parameters:**
- `tokens` (List[str]): List of tokens to analyze
- `include_position` (bool): Include position information
- `include_length` (bool): Include length information
- `include_case` (bool): Include case information

**Returns:** DataFrame with token features

**Example:**
```python
tokens = ["Hello", "world", "123", "!"] 
token_features = tokenization.extract_token_features(tokens)
print(token_features)
```

## Advanced Tokenization

### Subword Tokenization
```python
def subword_tokenize(
    text: str,
    method: str = "bpe",
    vocab_size: int = 10000
) -> List[str]
```

Perform subword tokenization using various algorithms.

**Parameters:**
- `text` (str): Input text
- `method` (str): Tokenization method ("bpe", "wordpiece")
- `vocab_size` (int): Vocabulary size for training

**Returns:** List of subword tokens

**Example:**
```python
text = "tokenization"
subwords = tokenization.subword_tokenize(text, method="bpe")
print(subwords)  # ['token', 'ization'] or similar
```

### N-gram Generation
```python
def generate_ngrams(
    tokens: List[str],
    n: int,
    pad_start: bool = False,
    pad_end: bool = False
) -> List[Tuple[str, ...]]
```

Generate n-grams from a list of tokens.

**Parameters:**
- `tokens` (List[str]): Input tokens
- `n` (int): N-gram size
- `pad_start` (bool): Add start padding
- `pad_end` (bool): Add end padding

**Returns:** List of n-gram tuples

**Example:**
```python
tokens = ["hello", "world", "how", "are", "you"]
bigrams = tokenization.generate_ngrams(tokens, 2)
print(bigrams)  # [('hello', 'world'), ('world', 'how'), ('how', 'are'), ('are', 'you')]

trigrams = tokenization.generate_ngrams(tokens, 3)
print(trigrams)  # [('hello', 'world', 'how'), ('world', 'how', 'are'), ('how', 'are', 'you')]
```

## Integration with Document Processing

### Document Tokenization
```python
def tokenize_document(
    document: dk_doc.Document,
    level: str = "word",
    preserve_metadata: bool = True
) -> dk_doc.Document
```

Tokenize a document while preserving its structure and metadata.

**Parameters:**
- `document` (dk_doc.Document): Input document
- `level` (str): Tokenization level ("char", "word", "sentence")
- `preserve_metadata` (bool): Keep original metadata

**Returns:** Tokenized document

**Example:**
```python
from dataknobs_structures import document as dk_doc
from dataknobs_xization import tokenization

# Create document
doc = dk_doc.Document(
    text="Hello world. How are you?",
    metadata={"title": "Greeting", "author": "Alice"}
)

# Tokenize at word level
tokenized_doc = tokenization.tokenize_document(doc, level="word")
print(tokenized_doc.text)  # Tokenized version
print(tokenized_doc.metadata)  # Original metadata preserved
```

### Batch Document Processing
```python
def batch_tokenize_documents(
    documents: List[dk_doc.Document],
    tokenizer_config: Dict[str, Any]
) -> List[dk_doc.Document]
```

Tokenize multiple documents efficiently.

**Example:**
```python
documents = [doc1, doc2, doc3]
config = {
    "level": "word",
    "lowercase": True,
    "remove_punctuation": True
}

tokenized_docs = tokenization.batch_tokenize_documents(documents, config)
```

## Usage Patterns

### Text Preprocessing Pipeline
```python
from dataknobs_xization import tokenization, normalize
from dataknobs_utils import file_utils

def preprocess_text_pipeline(text: str) -> Dict[str, Any]:
    """Complete text preprocessing pipeline."""
    # Normalize text first
    normalized = normalize.basic_normalization_fn(text)
    
    # Tokenize at different levels
    chars = tokenization.tokenize_characters(normalized)
    words = tokenization.tokenize_words(normalized, lowercase=True)
    sentences = tokenization.tokenize_sentences(normalized)
    
    # Extract features
    char_features = tokenization.extract_character_features(normalized)
    token_features = tokenization.extract_token_features(words)
    
    # Generate n-grams
    bigrams = tokenization.generate_ngrams(words, 2)
    trigrams = tokenization.generate_ngrams(words, 3)
    
    return {
        "original": text,
        "normalized": normalized,
        "tokens": {
            "characters": chars,
            "words": words,
            "sentences": sentences
        },
        "features": {
            "character": char_features,
            "token": token_features
        },
        "ngrams": {
            "bigrams": bigrams,
            "trigrams": trigrams
        }
    }

# Process text
text = "Hello, World! This is a test sentence."
result = preprocess_text_pipeline(text)
print(f"Words: {result['tokens']['words']}")
print(f"Bigrams: {result['ngrams']['bigrams'][:3]}")
```

### Document Analysis
```python
from dataknobs_xization import tokenization
from dataknobs_structures import document as dk_doc
from collections import Counter

def analyze_document_tokens(document: dk_doc.Document) -> Dict[str, Any]:
    """Analyze tokenization patterns in a document."""
    text = document.text
    
    # Get different token types
    words = tokenization.tokenize_words(text, lowercase=True)
    chars = tokenization.tokenize_characters(text)
    
    # Analyze patterns
    word_freq = Counter(words)
    char_freq = Counter(chars)
    
    # Extract features
    char_features = tokenization.extract_character_features(text)
    
    # Calculate statistics
    stats = {
        "word_count": len(words),
        "unique_words": len(word_freq),
        "char_count": len(chars),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        "most_common_words": word_freq.most_common(10),
        "character_types": {
            "alpha": char_features['is_alpha'].sum(),
            "digit": char_features['is_digit'].sum(),
            "space": char_features['is_space'].sum(),
            "punct": char_features['is_punct'].sum()
        }
    }
    
    return stats

# Analyze document
doc = dk_doc.Document("The quick brown fox jumps over the lazy dog.")
analysis = analyze_document_tokens(doc)
print(f"Word count: {analysis['word_count']}")
print(f"Unique words: {analysis['unique_words']}")
print(f"Most common: {analysis['most_common_words'][:5]}")
```

### Custom Tokenizer
```python
from dataknobs_xization import tokenization
import re

class CustomTokenizer:
    """Custom tokenizer with domain-specific rules."""
    
    def __init__(self, preserve_entities: bool = True):
        self.preserve_entities = preserve_entities
        self.entity_pattern = re.compile(r'@\w+|#\w+|https?://\S+')
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with entity preservation."""
        if self.preserve_entities:
            # Find entities first
            entities = self.entity_pattern.findall(text)
            entity_placeholder = "__ENTITY__"
            
            # Replace entities with placeholders
            processed_text = self.entity_pattern.sub(entity_placeholder, text)
            
            # Regular tokenization
            tokens = tokenization.tokenize_words(processed_text, lowercase=True)
            
            # Restore entities
            entity_iter = iter(entities)
            final_tokens = []
            for token in tokens:
                if token == entity_placeholder:
                    try:
                        final_tokens.append(next(entity_iter))
                    except StopIteration:
                        final_tokens.append(token)
                else:
                    final_tokens.append(token)
            
            return final_tokens
        else:
            return tokenization.tokenize_words(text, lowercase=True)

# Usage
tokenizer = CustomTokenizer()
text = "Check out @username and visit https://example.com #hashtag"
tokens = tokenizer.tokenize(text)
print(tokens)  # Preserves @username, URL, and #hashtag
```

## Performance Considerations

- Character-level tokenization is memory intensive for large texts
- Use generators for processing large document collections
- Consider chunking very large texts for feature extraction
- Cache tokenization results for repeated processing
- Use appropriate data types for feature storage

## Error Handling

```python
from dataknobs_xization import tokenization

def safe_tokenization(text: str, method: str = "word") -> List[str]:
    """Safely tokenize text with error handling."""
    try:
        if not text or not isinstance(text, str):
            return []
        
        if method == "word":
            return tokenization.tokenize_words(text)
        elif method == "char":
            return tokenization.tokenize_characters(text)
        elif method == "sentence":
            return tokenization.tokenize_sentences(text)
        else:
            raise ValueError(f"Unknown tokenization method: {method}")
            
    except Exception as e:
        print(f"Tokenization failed: {e}")
        return []

# Safe usage
tokens = safe_tokenization("Hello world", "word")
print(tokens)
```

## Integration Examples

### With Tree Structures
```python
from dataknobs_xization import tokenization
from dataknobs_structures import Tree

# Build token tree
def build_token_tree(text: str) -> Tree:
    root = Tree("document")
    
    sentences = tokenization.tokenize_sentences(text)
    for i, sentence in enumerate(sentences):
        sent_node = root.add_child(f"sentence_{i}")
        
        words = tokenization.tokenize_words(sentence)
        for j, word in enumerate(words):
            word_node = sent_node.add_child(f"word_{j}")
            
            chars = tokenization.tokenize_characters(word)
            for k, char in enumerate(chars):
                word_node.add_child(char)
    
    return root

text = "Hello world. How are you?"
token_tree = build_token_tree(text)
print(token_tree.as_string(multiline=True))
```

### With File Processing
```python
from dataknobs_xization import tokenization
from dataknobs_utils import file_utils
import json

# Process files and extract tokens
def process_text_files(input_dir: str, output_dir: str):
    for filepath in file_utils.filepath_generator(input_dir):
        if filepath.endswith('.txt'):
            # Read file content
            content_lines = list(file_utils.fileline_generator(filepath))
            full_text = '\n'.join(content_lines)
            
            # Tokenize
            words = tokenization.tokenize_words(full_text, lowercase=True)
            sentences = tokenization.tokenize_sentences(full_text)
            
            # Save results
            result = {
                'filename': filepath,
                'word_count': len(words),
                'sentence_count': len(sentences),
                'words': words,
                'sentences': sentences
            }
            
            output_file = f"{output_dir}/{os.path.basename(filepath)}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

process_text_files('/input/texts', '/output/tokens')
```

The tokenization module provides comprehensive text tokenization capabilities that integrate seamlessly with other dataknobs components for complete text processing pipelines.