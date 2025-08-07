# Xization Package API Reference

Complete API reference for the `dataknobs_xization` package - text normalization and tokenization tools.

## Package Overview

```python
from dataknobs_xization import (
    annotations,
    authorities,
    lexicon,
    masking_tokenizer,
    normalize
)

# Import key classes
from dataknobs_xization.masking_tokenizer import CharacterFeatures, TextFeatures
```

## Module Index

### Core Modules
- [normalize](normalization.md) - Text normalization and standardization
- [masking_tokenizer](masking.md) - Character-level features and masking
- [tokenization](tokenization.md) - Advanced tokenization capabilities

### Supporting Modules
- **annotations** - Text annotation and markup tools
- **authorities** - Authority control and standardization
- **lexicon** - Lexical analysis and vocabulary management

## Quick Reference

### Text Normalization
```python
from dataknobs_xization import normalize

# Basic normalization
normalized = normalize.basic_normalization_fn("Hello, WORLD!")

# CamelCase expansion
expanded = normalize.expand_camelcase_fn("firstName")

# Generate lexical variations
variations = normalize.get_lexical_variations(
    "multi-platform/cross-browser"
)

# Symbol handling
cleaned = normalize.drop_non_embedded_symbols_fn("!Hello world?")
embedded = normalize.drop_embedded_symbols_fn("user@domain.com", " ")

# Expand patterns
ampersand = normalize.expand_ampersand_fn("Research & Development")
hyphen_vars = normalize.get_hyphen_slash_expansions_fn("data-science")
```

### Character Features and Masking
```python
from dataknobs_xization.masking_tokenizer import CharacterFeatures, TextFeatures
from dataknobs_structures import document as dk_doc

# Character-level analysis
class MyCharFeatures(CharacterFeatures):
    @property
    def cdf(self):
        # Implementation for character dataframe
        pass

features = MyCharFeatures("Sample text")
char_df = features.cdf

# Text-level features
text_features = TextFeatures("Analysis text")
```

### Tokenization
```python
from dataknobs_xization import tokenization

# Different tokenization levels
chars = tokenization.tokenize_characters("Hello world!")
words = tokenization.tokenize_words("Hello, world!", lowercase=True)
sentences = tokenization.tokenize_sentences("Hello world. How are you?")

# Feature extraction
char_features = tokenization.extract_character_features("Text")
token_features = tokenization.extract_token_features(["word1", "word2"])

# N-gram generation
bigrams = tokenization.generate_ngrams(["a", "b", "c", "d"], 2)
trigrams = tokenization.generate_ngrams(["a", "b", "c", "d"], 3)
```

## Detailed Module APIs

### normalize Module

**Regular Expression Patterns:**
- `SQUASH_WS_RE` - Collapse whitespace
- `ALL_SYMBOLS_RE` - Match all symbols
- `CAMELCASE_LU_RE` - CamelCase lower-upper transitions
- `CAMELCASE_UL_RE` - CamelCase upper-lower transitions
- `NON_EMBEDDED_WORD_SYMS_RE` - Non-embedded symbols
- `EMBEDDED_SYMS_RE` - Embedded symbols
- `HYPHEN_SLASH_RE` - Hyphen/slash patterns
- `HYPHEN_ONLY_RE` - Hyphen-only patterns
- `SLASH_ONLY_RE` - Slash-only patterns
- `PARENTHETICAL_RE` - Parenthetical expressions
- `AMPERSAND_RE` - Ampersand patterns

**Core Functions:**
- `expand_camelcase_fn(text: str) -> str`
- `drop_non_embedded_symbols_fn(text: str, repl: str = "") -> str`
- `drop_embedded_symbols_fn(text: str, repl: str = "") -> str`
- `get_hyphen_slash_expansions_fn(text: str, subs: List[str] = ("-", " ", ""), add_self: bool = True, do_split: bool = True, min_split_token_len: int = 2, hyphen_slash_re=HYPHEN_SLASH_RE) -> Set[str]`
- `drop_parentheticals_fn(text: str) -> str`
- `expand_ampersand_fn(text: str) -> str`
- `get_lexical_variations(text: str, **kwargs) -> Set[str]`
- `basic_normalization_fn(text: str) -> str`

### masking_tokenizer Module

**Abstract Classes:**
```python
class CharacterFeatures(ABC):
    def __init__(self, doctext: Union[dk_doc.Text, str], roll_padding: int = 0)
    
    @property
    @abstractmethod
    def cdf(self) -> pd.DataFrame:
        """Character dataframe with each padded text character as a row."""
        pass
    
    # Properties
    @property
    def doctext(self) -> dk_doc.Text
    @property
    def text_col(self) -> str
    @property
    def text(self) -> str
    @property
    def text_id(self) -> Any
```

```python
class TextFeatures:
    def __init__(self, doctext: Union[dk_doc.Text, str])
    
    # Methods for text-level feature extraction
    def extract_features(self) -> Dict[str, Any]
    def analyze_patterns(self) -> Dict[str, Any]
```

### annotations Module

**Functions:**
- Text annotation utilities
- Markup processing
- Annotation validation
- Format conversion

### authorities Module

**Functions:**
- Authority control for names and terms
- Standardization utilities
- Controlled vocabulary management
- Cross-reference resolution

### lexicon Module

**Functions:**
- Vocabulary analysis
- Lexical statistics
- Term frequency analysis
- Lexicon building utilities

## Usage Patterns

### Complete Text Processing Pipeline
```python
from dataknobs_xization import normalize, masking_tokenizer
from dataknobs_structures import document as dk_doc
from dataknobs_utils import file_utils
import pandas as pd

class TextProcessingPipeline:
    """Complete text processing with normalization, tokenization, and masking."""
    
    def __init__(self, config: dict):
        self.config = config
        self.normalize_config = config.get('normalize', {})
        self.mask_config = config.get('masking', {})
    
    def process_text(self, text: str) -> dict:
        """Process text through complete pipeline."""
        results = {'original': text}
        
        # Step 1: Normalization
        normalized = self._normalize_text(text)
        results['normalized'] = normalized
        
        # Step 2: Generate variations
        variations = normalize.get_lexical_variations(
            normalized, **self.normalize_config
        )
        results['variations'] = list(variations)
        
        # Step 3: Character-level analysis
        char_analysis = self._analyze_characters(normalized)
        results['character_analysis'] = char_analysis
        
        # Step 4: Tokenization
        tokens = self._tokenize_text(normalized)
        results['tokens'] = tokens
        
        return results
    
    def _normalize_text(self, text: str) -> str:
        """Apply normalization pipeline."""
        # Expand camelCase
        text = normalize.expand_camelcase_fn(text)
        
        # Expand ampersands
        text = normalize.expand_ampersand_fn(text)
        
        # Drop parentheticals if configured
        if self.normalize_config.get('drop_parentheticals', True):
            text = normalize.drop_parentheticals_fn(text)
        
        # Handle symbols
        if self.normalize_config.get('drop_non_embedded_symbols', True):
            text = normalize.drop_non_embedded_symbols_fn(text)
        
        # Basic normalization
        text = normalize.basic_normalization_fn(text)
        
        return text
    
    def _analyze_characters(self, text: str) -> dict:
        """Analyze character-level features."""
        # Create character features implementation
        class PipelineCharFeatures(masking_tokenizer.CharacterFeatures):
            @property
            def cdf(self):
                chars = list(self.text)
                return pd.DataFrame({
                    self.text_col: chars,
                    'position': range(len(chars)),
                    'is_alpha': [c.isalpha() for c in chars],
                    'is_digit': [c.isdigit() for c in chars],
                    'is_space': [c.isspace() for c in chars],
                    'is_punct': [not c.isalnum() and not c.isspace() for c in chars]
                })
        
        features = PipelineCharFeatures(text)
        cdf = features.cdf
        
        return {
            'total_chars': len(cdf),
            'alpha_chars': cdf['is_alpha'].sum(),
            'digit_chars': cdf['is_digit'].sum(),
            'space_chars': cdf['is_space'].sum(),
            'punct_chars': cdf['is_punct'].sum(),
            'alpha_ratio': cdf['is_alpha'].mean(),
            'digit_ratio': cdf['is_digit'].mean()
        }
    
    def _tokenize_text(self, text: str) -> dict:
        """Tokenize text at multiple levels."""
        from dataknobs_xization import tokenization
        
        return {
            'characters': tokenization.tokenize_characters(text),
            'words': tokenization.tokenize_words(text, lowercase=True),
            'sentences': tokenization.tokenize_sentences(text)
        }
    
    def process_documents(self, documents: List[dk_doc.Document]) -> List[dict]:
        """Process multiple documents."""
        results = []
        for doc in documents:
            doc_result = self.process_text(doc.text)
            doc_result['document_id'] = getattr(doc, 'text_id', None)
            doc_result['metadata'] = getattr(doc, 'metadata', {})
            results.append(doc_result)
        return results

# Usage
config = {
    'normalize': {
        'drop_parentheticals': True,
        'drop_non_embedded_symbols': True,
        'expand_camelcase': True,
        'expand_ampersands': True,
        'add_eng_plurals': True
    },
    'masking': {
        'mask_probability': 0.15,
        'preserve_structure': True
    }
}

pipeline = TextProcessingPipeline(config)

# Process single text
text = "getUserName() & validateInput (required)"
result = pipeline.process_text(text)
print(f"Original: {result['original']}")
print(f"Normalized: {result['normalized']}")
print(f"Variations: {len(result['variations'])}")
print(f"Character analysis: {result['character_analysis']}")

# Process documents
documents = [
    dk_doc.Document("JavaScript & Node.js development", text_id="doc1"),
    dk_doc.Document("Python (programming language) tutorial", text_id="doc2")
]

doc_results = pipeline.process_documents(documents)
for result in doc_results:
    print(f"Doc {result['document_id']}: {result['normalized']}")
```

### Privacy-Preserving Text Analytics
```python
from dataknobs_xization import normalize, masking_tokenizer
from dataknobs_utils import elasticsearch_utils
import json

class PrivacyPreservingAnalytics:
    """Analytics with built-in privacy preservation."""
    
    def __init__(self, privacy_config: dict):
        self.mask_probability = privacy_config.get('mask_probability', 0.15)
        self.preserve_patterns = privacy_config.get('preserve_patterns', [])
        self.differential_privacy = privacy_config.get('differential_privacy', {})
    
    def analyze_corpus(self, texts: List[str]) -> dict:
        """Analyze text corpus with privacy preservation."""
        # Step 1: Normalize all texts
        normalized_texts = []
        for text in texts:
            # Apply normalization pipeline
            normalized = normalize.basic_normalization_fn(text)
            normalized = normalize.expand_camelcase_fn(normalized)
            normalized_texts.append(normalized)
        
        # Step 2: Apply privacy-preserving masking
        masked_analytics = self._masked_analysis(normalized_texts)
        
        # Step 3: Generate aggregate statistics
        aggregate_stats = self._compute_aggregates(normalized_texts, masked_analytics)
        
        return {
            'corpus_size': len(texts),
            'privacy_parameters': {
                'mask_probability': self.mask_probability,
                'differential_privacy': self.differential_privacy
            },
            'masked_analytics': masked_analytics,
            'aggregate_statistics': aggregate_stats
        }
    
    def _masked_analysis(self, texts: List[str]) -> dict:
        """Perform analysis on masked text data."""
        # Character-level masking for each text
        class AnalyticsCharFeatures(masking_tokenizer.CharacterFeatures):
            def __init__(self, doctext, mask_prob):
                super().__init__(doctext)
                self.mask_prob = mask_prob
            
            @property
            def cdf(self):
                import numpy as np
                chars = list(self.text)
                np.random.seed(42)  # Reproducible for testing
                
                return pd.DataFrame({
                    self.text_col: chars,
                    'position': range(len(chars)),
                    'is_alpha': [c.isalpha() for c in chars],
                    'is_digit': [c.isdigit() for c in chars],
                    'is_masked': np.random.random(len(chars)) < self.mask_prob
                })
        
        total_chars = 0
        total_masked = 0
        feature_stats = {'alpha': 0, 'digit': 0, 'other': 0}
        
        for text in texts:
            features = AnalyticsCharFeatures(text, self.mask_probability)
            cdf = features.cdf
            
            total_chars += len(cdf)
            total_masked += cdf['is_masked'].sum()
            feature_stats['alpha'] += cdf['is_alpha'].sum()
            feature_stats['digit'] += cdf['is_digit'].sum()
            feature_stats['other'] += len(cdf) - cdf['is_alpha'].sum() - cdf['is_digit'].sum()
        
        return {
            'total_characters': total_chars,
            'total_masked': total_masked,
            'mask_ratio': total_masked / total_chars if total_chars > 0 else 0,
            'character_distribution': feature_stats
        }
    
    def _compute_aggregates(self, texts: List[str], masked_analytics: dict) -> dict:
        """Compute aggregate statistics with noise for differential privacy."""
        import numpy as np
        
        # Basic statistics
        word_counts = [len(text.split()) for text in texts]
        char_counts = [len(text) for text in texts]
        
        # Add differential privacy noise if configured
        epsilon = self.differential_privacy.get('epsilon', 1.0)
        if epsilon > 0:
            scale = 1.0 / epsilon
            word_count_noise = np.random.laplace(0, scale, len(word_counts))
            char_count_noise = np.random.laplace(0, scale, len(char_counts))
            
            word_counts = [max(0, wc + noise) for wc, noise in zip(word_counts, word_count_noise)]
            char_counts = [max(0, cc + noise) for cc, noise in zip(char_counts, char_count_noise)]
        
        return {
            'avg_word_count': np.mean(word_counts),
            'avg_char_count': np.mean(char_counts),
            'word_count_std': np.std(word_counts),
            'char_count_std': np.std(char_counts),
            'privacy_noise_added': epsilon > 0
        }

# Usage
privacy_config = {
    'mask_probability': 0.2,
    'differential_privacy': {'epsilon': 0.5},
    'preserve_patterns': ['email', 'phone']
}

analytics = PrivacyPreservingAnalytics(privacy_config)

texts = [
    "This document contains sensitive information about users.",
    "Financial data and personal details are stored here.",
    "Public information that can be analyzed without privacy concerns."
]

results = analytics.analyze_corpus(texts)
print(json.dumps(results, indent=2))
```

### Integration with Other Dataknobs Packages
```python
from dataknobs_xization import normalize, masking_tokenizer
from dataknobs_structures import Tree, document as dk_doc
from dataknobs_utils import elasticsearch_utils, file_utils

def build_normalized_search_index(
    documents: List[dk_doc.Document],
    index_name: str
) -> elasticsearch_utils.ElasticsearchIndex:
    """Build search index with normalized and varied text."""
    
    # Configure Elasticsearch
    table_settings = [
        elasticsearch_utils.TableSettings(
            index_name,
            {"number_of_shards": 1, "number_of_replicas": 0},
            {
                "properties": {
                    "original_text": {"type": "text"},
                    "normalized_text": {"type": "text", "analyzer": "english"},
                    "variations": {"type": "text"},
                    "character_features": {"type": "object"},
                    "document_id": {"type": "keyword"}
                }
            }
        )
    ]
    
    # Create index
    es_index = elasticsearch_utils.ElasticsearchIndex(None, table_settings)
    
    # Process documents and create batch file
    def document_generator():
        for doc in documents:
            # Normalize text
            normalized = normalize.basic_normalization_fn(doc.text)
            normalized = normalize.expand_camelcase_fn(normalized)
            normalized = normalize.expand_ampersand_fn(normalized)
            
            # Generate variations
            variations = normalize.get_lexical_variations(normalized)
            
            # Extract character features
            class IndexCharFeatures(masking_tokenizer.CharacterFeatures):
                @property
                def cdf(self):
                    chars = list(self.text)
                    return pd.DataFrame({
                        self.text_col: chars,
                        'is_alpha': [c.isalpha() for c in chars],
                        'is_digit': [c.isdigit() for c in chars]
                    })
            
            features = IndexCharFeatures(normalized)
            cdf = features.cdf
            
            yield {
                'original_text': doc.text,
                'normalized_text': normalized,
                'variations': ' '.join(variations),
                'character_features': {
                    'total_chars': len(cdf),
                    'alpha_ratio': cdf['is_alpha'].mean(),
                    'digit_ratio': cdf['is_digit'].mean()
                },
                'document_id': getattr(doc, 'text_id', str(hash(doc.text)))
            }
    
    # Create batch file and load
    with open('search_batch.jsonl', 'w') as f:
        elasticsearch_utils.add_batch_data(
            f, document_generator(), index_name
        )
    
    return es_index

# Usage
documents = [
    dk_doc.Document("JavaScript & Node.js Development", text_id="tech1"),
    dk_doc.Document("Machine Learning (ML) Algorithms", text_id="ai1"),
    dk_doc.Document("Data Science with Python/R", text_id="data1")
]

search_index = build_normalized_search_index(documents, "normalized_docs")
print("Search index created with normalized and varied text")
```

## Testing Utilities

```python
from dataknobs_xization import normalize, masking_tokenizer
import pytest
import tempfile

class TestXizationFunctions:
    """Test utilities for xization package."""
    
    def test_normalization(self):
        """Test normalization functions."""
        # Test camelCase expansion
        assert normalize.expand_camelcase_fn("firstName") == "first Name"
        assert normalize.expand_camelcase_fn("XMLParser") == "XML Parser"
        
        # Test symbol handling
        assert normalize.drop_non_embedded_symbols_fn("!Hello world?") == "Hello world"
        assert normalize.drop_embedded_symbols_fn("user@domain.com") == "userdomaincom"
        
        # Test ampersand expansion
        assert normalize.expand_ampersand_fn("A & B") == "A and B"
        
        # Test basic normalization
        result = normalize.basic_normalization_fn("  HELLO,   WORLD!  ")
        assert result.strip().lower() == "hello, world!"
    
    def test_variations(self):
        """Test lexical variation generation."""
        variations = normalize.get_lexical_variations("multi-platform")
        assert "multi platform" in variations
        assert "multiplatform" in variations
        assert "multi-platform" in variations
    
    def test_character_features(self):
        """Test character feature extraction."""
        class TestCharFeatures(masking_tokenizer.CharacterFeatures):
            @property
            def cdf(self):
                chars = list(self.text)
                return pd.DataFrame({
                    self.text_col: chars,
                    'is_alpha': [c.isalpha() for c in chars]
                })
        
        features = TestCharFeatures("Hello123")
        cdf = features.cdf
        assert len(cdf) == 8
        assert cdf['is_alpha'].sum() == 5  # "Hello"
    
    def test_integration(self):
        """Test integration between modules."""
        text = "getUserName() & validateInput"
        
        # Normalize
        normalized = normalize.expand_camelcase_fn(text)
        normalized = normalize.expand_ampersand_fn(normalized)
        normalized = normalize.basic_normalization_fn(normalized)
        
        # Extract features
        class IntegrationFeatures(masking_tokenizer.CharacterFeatures):
            @property
            def cdf(self):
                chars = list(self.text)
                return pd.DataFrame({self.text_col: chars})
        
        features = IntegrationFeatures(normalized)
        assert len(features.cdf) > 0

# Run tests
if __name__ == "__main__":
    test_suite = TestXizationFunctions()
    test_suite.test_normalization()
    test_suite.test_variations()
    test_suite.test_character_features()
    test_suite.test_integration()
    print("All tests passed!")
```

## Performance Considerations

- Normalization functions use pre-compiled regex patterns for efficiency
- Character feature extraction can be memory-intensive for large texts
- Lexical variation generation may produce many variations - use selectively
- Consider caching normalized results for frequently accessed texts
- Use appropriate batch sizes for bulk processing

## Best Practices

1. **Pipeline Design**: Design normalization pipelines appropriate for your domain
2. **Configuration**: Use configuration objects to manage normalization parameters
3. **Testing**: Test normalization results on representative data
4. **Caching**: Cache expensive operations like variation generation
5. **Memory Management**: Monitor memory usage with large-scale character analysis
6. **Integration**: Coordinate with other dataknobs packages for seamless workflows
7. **Documentation**: Document normalization decisions and their rationale

## Version Information

- **Package Version**: 1.0.0
- **Python Compatibility**: 3.8+
- **Dependencies**: pandas, numpy, dataknobs-structures, dataknobs-utils

For detailed documentation of individual modules, see their respective documentation pages.