# dataknobs-xization API Reference

Complete API documentation for the `dataknobs_xization` package.

> **💡 Quick Links:**
> - [Complete API Documentation](reference/xization.md) - Full auto-generated reference
> - [Source Code](https://github.com/kbs-labs/dataknobs/tree/main/packages/xization/src/dataknobs_xization) - Browse on GitHub
> - [Package Guide](../packages/xization/index.md) - Detailed documentation

## Package Information

- **Package Name**: `dataknobs_xization`
- **Version**: 1.0.0
- **Description**: Text normalization and tokenization tools
- **Python Requirements**: >=3.8

## Installation

```bash
pip install dataknobs-xization
```

## Import Statement

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

## Module Documentation

### normalize

#### Regular Expression Patterns

##### SQUASH_WS_RE
::: dataknobs_xization.normalize.SQUASH_WS_RE
    options:
      show_source: true

##### ALL_SYMBOLS_RE
::: dataknobs_xization.normalize.ALL_SYMBOLS_RE
    options:
      show_source: true

##### CAMELCASE_LU_RE
::: dataknobs_xization.normalize.CAMELCASE_LU_RE
    options:
      show_source: true

##### CAMELCASE_UL_RE
::: dataknobs_xization.normalize.CAMELCASE_UL_RE
    options:
      show_source: true

##### NON_EMBEDDED_WORD_SYMS_RE
::: dataknobs_xization.normalize.NON_EMBEDDED_WORD_SYMS_RE
    options:
      show_source: true

##### EMBEDDED_SYMS_RE
::: dataknobs_xization.normalize.EMBEDDED_SYMS_RE
    options:
      show_source: true

##### HYPHEN_SLASH_RE
::: dataknobs_xization.normalize.HYPHEN_SLASH_RE
    options:
      show_source: true

##### HYPHEN_ONLY_RE
::: dataknobs_xization.normalize.HYPHEN_ONLY_RE
    options:
      show_source: true

##### SLASH_ONLY_RE
::: dataknobs_xization.normalize.SLASH_ONLY_RE
    options:
      show_source: true

##### PARENTHETICAL_RE
::: dataknobs_xization.normalize.PARENTHETICAL_RE
    options:
      show_source: true

##### AMPERSAND_RE
::: dataknobs_xization.normalize.AMPERSAND_RE
    options:
      show_source: true

#### Functions

##### expand_camelcase_fn
::: dataknobs_xization.normalize.expand_camelcase_fn
    options:
      show_source: true

##### drop_non_embedded_symbols_fn
::: dataknobs_xization.normalize.drop_non_embedded_symbols_fn
    options:
      show_source: true

##### drop_embedded_symbols_fn
::: dataknobs_xization.normalize.drop_embedded_symbols_fn
    options:
      show_source: true

##### get_hyphen_slash_expansions_fn
::: dataknobs_xization.normalize.get_hyphen_slash_expansions_fn
    options:
      show_source: true

##### drop_parentheticals_fn
::: dataknobs_xization.normalize.drop_parentheticals_fn
    options:
      show_source: true

##### expand_ampersand_fn
::: dataknobs_xization.normalize.expand_ampersand_fn
    options:
      show_source: true

##### get_lexical_variations
::: dataknobs_xization.normalize.get_lexical_variations
    options:
      show_source: true

### masking_tokenizer

#### Classes

##### CharacterFeatures
::: dataknobs_xization.masking_tokenizer.CharacterFeatures
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - __init__
        - cdf
        - doctext
        - text_col
        - text
        - text_id

##### TextFeatures
::: dataknobs_xization.masking_tokenizer.TextFeatures
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

### annotations

#### Functions and Classes

::: dataknobs_xization.annotations
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

### authorities

#### Functions and Classes

::: dataknobs_xization.authorities
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

### lexicon

#### Functions and Classes

::: dataknobs_xization.lexicon
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

## Usage Examples

### Text Normalization Example

```python
from dataknobs_xization import normalize

# Basic text normalization
text = "  Hello,    WORLD!  \n\t How   are you?  "
normalized = normalize.basic_normalization_fn(text)
print(normalized)  # "hello, world! how are you?"

# CamelCase expansion
camel_text = "firstName"
expanded = normalize.expand_camelcase_fn(camel_text)
print(expanded)  # "first Name"

# Generate lexical variations
text_with_hyphens = "multi-platform/cross-browser"
variations = normalize.get_lexical_variations(text_with_hyphens)
print(f"Generated {len(variations)} variations:")
for var in sorted(variations):
    print(f"  {var}")

# Symbol handling
text_with_symbols = "!Hello world?"
cleaned = normalize.drop_non_embedded_symbols_fn(text_with_symbols)
print(cleaned)  # "Hello world"

embedded_text = "user@domain.com"
processed = normalize.drop_embedded_symbols_fn(embedded_text, " ")
print(processed)  # "user domain com"

# Ampersand expansion
ampersand_text = "Research & Development"
expanded_ampersand = normalize.expand_ampersand_fn(ampersand_text)
print(expanded_ampersand)  # "Research and Development"
```

### Character Features Example

```python
from dataknobs_xization.masking_tokenizer import CharacterFeatures
from dataknobs_structures import document as dk_doc
import pandas as pd

# Create a concrete implementation of CharacterFeatures
class BasicCharacterFeatures(CharacterFeatures):
    """Basic character-level feature extraction."""
    
    @property
    def cdf(self) -> pd.DataFrame:
        """Create character dataframe with features."""
        if not hasattr(self, '_cdf'):
            chars = list(self.text)
            
            # Add padding if specified
            if self._roll_padding > 0:
                pad_char = '<PAD>'
                chars = ([pad_char] * self._roll_padding + 
                        chars + 
                        [pad_char] * self._roll_padding)
            
            # Create feature dataframe
            self._cdf = pd.DataFrame({
                self.text_col: chars,
                'position': range(len(chars)),
                'is_alpha': [c.isalpha() if c != '<PAD>' else False for c in chars],
                'is_digit': [c.isdigit() if c != '<PAD>' else False for c in chars],
                'is_upper': [c.isupper() if c != '<PAD>' else False for c in chars],
                'is_lower': [c.islower() if c != '<PAD>' else False for c in chars],
                'is_space': [c.isspace() if c != '<PAD>' else False for c in chars],
                'is_punct': [not c.isalnum() and not c.isspace() if c != '<PAD>' else False for c in chars],
                'is_padding': [c == '<PAD>' for c in chars]
            })
        
        return self._cdf

# Usage
text = "Hello, World! 123 👋"
features = BasicCharacterFeatures(text, roll_padding=2)

print(f"Text: {features.text}")
print(f"Text column: {features.text_col}")
print("\nCharacter DataFrame:")
print(features.cdf.head(10))

# Analyze character distribution
cdf = features.cdf
print("\nCharacter Analysis:")
print(f"Total characters: {len(cdf)}")
print(f"Alphabetic: {cdf['is_alpha'].sum()}")
print(f"Digits: {cdf['is_digit'].sum()}")
print(f"Spaces: {cdf['is_space'].sum()}")
print(f"Punctuation: {cdf['is_punct'].sum()}")
print(f"Padding: {cdf['is_padding'].sum()}")
```

### Text Masking Example

```python
from dataknobs_xization.masking_tokenizer import CharacterFeatures
import pandas as pd
import numpy as np

class MaskingCharacterFeatures(CharacterFeatures):
    """Character features with masking capability."""
    
    def __init__(self, doctext, roll_padding=0, mask_probability=0.15):
        super().__init__(doctext, roll_padding)
        self.mask_probability = mask_probability
    
    @property
    def cdf(self) -> pd.DataFrame:
        """Character dataframe with masking features."""
        if not hasattr(self, '_cdf'):
            chars = list(self.text)
            
            if self._roll_padding > 0:
                pad_char = '<PAD>'
                chars = ([pad_char] * self._roll_padding + 
                        chars + 
                        [pad_char] * self._roll_padding)
            
            # Set random seed for reproducibility
            np.random.seed(42)
            
            self._cdf = pd.DataFrame({
                self.text_col: chars,
                'original_char': chars,
                'position': range(len(chars)),
                'is_alpha': [c.isalpha() if c != '<PAD>' else False for c in chars],
                'is_digit': [c.isdigit() if c != '<PAD>' else False for c in chars],
                'should_mask': np.random.random(len(chars)) < self.mask_probability,
                'is_padding': [c == '<PAD>' for c in chars]
            })
            
            # Apply masking
            mask_indices = self._cdf['should_mask'] & ~self._cdf['is_padding']
            self._cdf.loc[mask_indices, self.text_col] = '[MASK]'
        
        return self._cdf
    
    def get_masked_text(self) -> str:
        """Get the masked version of the text."""
        cdf = self.cdf
        masked_chars = cdf[~cdf['is_padding']][self.text_col].tolist()
        return ''.join(masked_chars)

# Usage
original_text = "This is a sample text for demonstration."
masker = MaskingCharacterFeatures(original_text, mask_probability=0.2)

print(f"Original: {original_text}")
print(f"Masked:   {masker.get_masked_text()}")
print(f"\nMask Statistics:")
cdf = masker.cdf
print(f"Total chars: {len(cdf)}")
print(f"Masked chars: {cdf['should_mask'].sum()}")
print(f"Mask ratio: {cdf['should_mask'].mean():.2%}")
```

### Complete Text Processing Pipeline

```python
from dataknobs_xization import normalize, masking_tokenizer
from dataknobs_structures import document as dk_doc
import pandas as pd

class TextProcessingPipeline:
    """Complete text processing with normalization and analysis."""
    
    def __init__(self, normalize_config=None, analysis_config=None):
        self.normalize_config = normalize_config or {}
        self.analysis_config = analysis_config or {}
    
    def process_document(self, doc: dk_doc.Document) -> dict:
        """Process a document through the complete pipeline."""
        original_text = doc.text
        results = {
            'document_id': getattr(doc, 'text_id', None),
            'original_text': original_text
        }
        
        # Step 1: Normalization
        normalized_text = self._normalize_text(original_text)
        results['normalized_text'] = normalized_text
        
        # Step 2: Generate variations
        variations = normalize.get_lexical_variations(
            normalized_text, **self.normalize_config
        )
        results['variations'] = list(variations)
        results['variation_count'] = len(variations)
        
        # Step 3: Character analysis
        char_analysis = self._analyze_characters(normalized_text)
        results['character_analysis'] = char_analysis
        
        return results
    
    def _normalize_text(self, text: str) -> str:
        """Apply normalization pipeline."""
        # Expand camelCase
        text = normalize.expand_camelcase_fn(text)
        
        # Expand ampersands
        text = normalize.expand_ampersand_fn(text)
        
        # Drop parentheticals
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
        class AnalysisCharFeatures(masking_tokenizer.CharacterFeatures):
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
        
        features = AnalysisCharFeatures(text)
        cdf = features.cdf
        
        return {
            'total_characters': len(cdf),
            'alphabetic_characters': cdf['is_alpha'].sum(),
            'digit_characters': cdf['is_digit'].sum(),
            'space_characters': cdf['is_space'].sum(),
            'punctuation_characters': cdf['is_punct'].sum(),
            'alphabetic_ratio': cdf['is_alpha'].mean(),
            'digit_ratio': cdf['is_digit'].mean(),
            'space_ratio': cdf['is_space'].mean(),
            'punctuation_ratio': cdf['is_punct'].mean()
        }
    
    def process_batch(self, documents: list) -> list:
        """Process multiple documents."""
        return [self.process_document(doc) for doc in documents]

# Usage example
config = {
    'drop_parentheticals': True,
    'drop_non_embedded_symbols': True,
    'expand_camelcase': True,
    'expand_ampersands': True,
    'add_eng_plurals': True
}

pipeline = TextProcessingPipeline(normalize_config=config)

# Create sample documents
documents = [
    dk_doc.Document(
        "getUserName() & validateInput (required)", 
        text_id="tech_doc_1"
    ),
    dk_doc.Document(
        "Machine Learning (ML) & Artificial Intelligence",
        text_id="ai_doc_1" 
    )
]

# Process documents
results = pipeline.process_batch(documents)

# Display results
for result in results:
    print(f"\nDocument: {result['document_id']}")
    print(f"Original: {result['original_text']}")
    print(f"Normalized: {result['normalized_text']}")
    print(f"Variations: {result['variation_count']}")
    print(f"Character Analysis: {result['character_analysis']}")
```

### Integration with Other Packages

```python
from dataknobs_xization import normalize, masking_tokenizer
from dataknobs_utils import file_utils, elasticsearch_utils
from dataknobs_structures import Tree, document as dk_doc
import json

def create_searchable_documents(input_dir: str) -> list:
    """Create searchable documents with normalized text."""
    searchable_docs = []
    
    # Process all text files
    for filepath in file_utils.filepath_generator(input_dir):
        if filepath.endswith('.txt'):
            # Read file content
            content_lines = list(file_utils.fileline_generator(filepath))
            full_text = '\n'.join(content_lines)
            
            # Normalize text
            normalized = normalize.basic_normalization_fn(full_text)
            normalized = normalize.expand_camelcase_fn(normalized)
            normalized = normalize.expand_ampersand_fn(normalized)
            
            # Generate search variations
            variations = normalize.get_lexical_variations(
                normalized,
                expand_camelcase=True,
                do_hyphen_expansion=True,
                do_slash_expansion=True
            )
            
            # Create searchable document
            searchable_doc = {
                'filepath': filepath,
                'original_text': full_text,
                'normalized_text': normalized,
                'search_variations': ' '.join(variations),
                'variation_count': len(variations)
            }
            
            searchable_docs.append(searchable_doc)
    
    return searchable_docs

# Create Elasticsearch index with normalized documents
def index_normalized_documents(documents: list, index_name: str):
    """Index normalized documents in Elasticsearch."""
    table_settings = elasticsearch_utils.TableSettings(
        index_name,
        {"number_of_shards": 1, "number_of_replicas": 0},
        {
            "properties": {
                "original_text": {"type": "text"},
                "normalized_text": {"type": "text", "analyzer": "english"},
                "search_variations": {"type": "text"},
                "filepath": {"type": "keyword"},
                "variation_count": {"type": "integer"}
            }
        }
    )
    
    index = elasticsearch_utils.ElasticsearchIndex(None, [table_settings])
    
    # Create batch file
    with open("normalized_batch.jsonl", "w") as f:
        elasticsearch_utils.add_batch_data(
            f, iter(documents), index_name
        )
    
    return index

# Usage
documents = create_searchable_documents("/path/to/text/files")
index = index_normalized_documents(documents, "normalized_texts")
print(f"Indexed {len(documents)} normalized documents")
```

## Error Handling

```python
from dataknobs_xization import normalize, masking_tokenizer
from dataknobs_structures import document as dk_doc

def safe_text_processing(text: str) -> dict:
    """Safely process text with comprehensive error handling."""
    results = {'original': text, 'errors': []}
    
    try:
        # Normalization with error handling
        normalized = normalize.basic_normalization_fn(text)
        results['normalized'] = normalized
    except Exception as e:
        results['errors'].append(f"Normalization failed: {e}")
        results['normalized'] = text
    
    try:
        # CamelCase expansion
        expanded = normalize.expand_camelcase_fn(results['normalized'])
        results['camelcase_expanded'] = expanded
    except Exception as e:
        results['errors'].append(f"CamelCase expansion failed: {e}")
        results['camelcase_expanded'] = results['normalized']
    
    try:
        # Variation generation
        variations = normalize.get_lexical_variations(results['camelcase_expanded'])
        results['variations'] = list(variations)
    except Exception as e:
        results['errors'].append(f"Variation generation failed: {e}")
        results['variations'] = [results['camelcase_expanded']]
    
    try:
        # Character analysis
        class SafeCharFeatures(masking_tokenizer.CharacterFeatures):
            @property
            def cdf(self):
                import pandas as pd
                chars = list(self.text) if self.text else []
                return pd.DataFrame({
                    self.text_col: chars,
                    'is_alpha': [c.isalpha() for c in chars]
                })
        
        features = SafeCharFeatures(results['camelcase_expanded'])
        results['character_count'] = len(features.cdf)
    except Exception as e:
        results['errors'].append(f"Character analysis failed: {e}")
        results['character_count'] = 0
    
    results['success'] = len(results['errors']) == 0
    return results

# Usage
test_texts = [
    "Normal text for processing",
    "camelCaseText & symbols!",
    "",  # Empty string
    None,  # None value
    "Special unicode: 👋🌍"
]

for i, text in enumerate(test_texts):
    try:
        result = safe_text_processing(text or "")
        print(f"\nTest {i+1}: {'SUCCESS' if result['success'] else 'ERRORS'}")
        print(f"Original: {repr(text)}")
        if result['success']:
            print(f"Normalized: {result['normalized']}")
            print(f"Variations: {len(result['variations'])}")
        else:
            print(f"Errors: {result['errors']}")
    except Exception as e:
        print(f"\nTest {i+1}: CRITICAL ERROR - {e}")
```

## Testing

```python
import pytest
from dataknobs_xization import normalize, masking_tokenizer
from dataknobs_structures import document as dk_doc
import pandas as pd

class TestXizationFunctions:
    """Test suite for xization functionality."""
    
    def test_normalization_functions(self):
        """Test core normalization functions."""
        # Test camelCase expansion
        assert normalize.expand_camelcase_fn("firstName") == "first Name"
        assert normalize.expand_camelcase_fn("XMLParser") == "XML Parser"
        
        # Test symbol handling
        assert normalize.drop_non_embedded_symbols_fn("!Hello world?") == "Hello world"
        assert normalize.drop_embedded_symbols_fn("user@domain.com") == "userdomaincom"
        
        # Test ampersand expansion
        assert normalize.expand_ampersand_fn("A & B") == "A and B"
        
        # Test parenthetical removal
        assert normalize.drop_parentheticals_fn("Text (with note)") == "Text "
    
    def test_lexical_variations(self):
        """Test lexical variation generation."""
        variations = normalize.get_lexical_variations("multi-platform")
        
        # Check expected variations are present
        assert "multi platform" in variations
        assert "multiplatform" in variations
        assert "multi-platform" in variations
        
        # Check it returns a set
        assert isinstance(variations, set)
        assert len(variations) > 1
    
    def test_character_features(self):
        """Test character feature extraction."""
        class TestCharFeatures(masking_tokenizer.CharacterFeatures):
            @property
            def cdf(self):
                chars = list(self.text)
                return pd.DataFrame({
                    self.text_col: chars,
                    'is_alpha': [c.isalpha() for c in chars],
                    'is_digit': [c.isdigit() for c in chars]
                })
        
        features = TestCharFeatures("Hello123")
        cdf = features.cdf
        
        # Test basic properties
        assert len(cdf) == 8
        assert cdf['is_alpha'].sum() == 5  # "Hello"
        assert cdf['is_digit'].sum() == 3  # "123"
        
        # Test text properties
        assert features.text == "Hello123"
        assert features.text_col == 'text'  # Default column name
    
    def test_document_integration(self):
        """Test integration with document structures."""
        doc = dk_doc.Text("Test document", text_id="test1")
        
        class DocCharFeatures(masking_tokenizer.CharacterFeatures):
            @property
            def cdf(self):
                chars = list(self.text)
                return pd.DataFrame({self.text_col: chars})
        
        features = DocCharFeatures(doc)
        assert features.text_id == "test1"
        assert features.text == "Test document"
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test empty text
        empty_variations = normalize.get_lexical_variations("")
        assert isinstance(empty_variations, set)
        
        # Test None handling in utility function
        from dataknobs_xization.normalize import basic_normalization_fn
        try:
            result = basic_normalization_fn("")
            assert isinstance(result, str)
        except Exception:
            pytest.fail("Should handle empty string gracefully")

# Run tests
if __name__ == "__main__":
    test_suite = TestXizationFunctions()
    test_suite.test_normalization_functions()
    test_suite.test_lexical_variations()
    test_suite.test_character_features()
    test_suite.test_document_integration()
    test_suite.test_error_handling()
    print("All tests passed!")
```

## Performance Notes

- **Regular Expressions**: Pre-compiled patterns for efficient text processing
- **Character Analysis**: Memory-intensive for large texts - use streaming for big documents
- **Variation Generation**: Can produce many variations - filter appropriately
- **Pandas DataFrames**: Efficient for character-level analysis but consider memory usage

## Dependencies

Core dependencies for dataknobs_xization:

```txt
pandas>=1.3.0
numpy>=1.20.0
dataknobs-structures>=1.0.0
dataknobs-utils>=1.0.0
```

## Contributing

For contributing to dataknobs_xization:

1. Fork the repository
2. Create feature branch for text processing enhancements
3. Add comprehensive tests for normalization functions
4. Test with various text types and edge cases
5. Submit pull request with documentation updates

See [Contributing Guide](../development/contributing.md) for detailed information.

## Changelog

### Version 1.0.0
- Initial release
- Text normalization functions
- Character-level feature extraction
- Lexical variation generation
- Masking tokenizer framework
- Integration with dataknobs-structures

## License

See [License](../license.md) for license information.