# Masking API Documentation

The masking functionality in the `dataknobs_xization` package provides advanced text masking and tokenization capabilities for privacy protection, data anonymization, and feature extraction.

## Overview

The masking system provides:

- Character-level feature extraction and analysis
- Text-level feature computation
- Masking tokenization for sensitive data
- Integration with document structures
- Advanced pattern recognition and masking
- Privacy-preserving text processing

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

Abstract base class representing features of text as a DataFrame with each character as a row and columns representing character features.

**Parameters:**
- `doctext` (Union[dk_doc.Text, str]): Text to tokenize or Text document with metadata
- `roll_padding` (int, default=0): Number of pad characters added to each end of text

**Abstract Properties:**
- `cdf` (pd.DataFrame): Character DataFrame with each padded text character as a row

**Properties:**
- `doctext` (dk_doc.Text): The document text wrapper
- `text_col` (str): Name of cdf column holding text characters
- `text` (str): The text string
- `text_id` (Any): The ID of the text

**Example Implementation:**
```python
from dataknobs_xization import masking_tokenizer
from dataknobs_structures import document as dk_doc
import pandas as pd
import numpy as np

class BasicCharacterFeatures(masking_tokenizer.CharacterFeatures):
    """Basic implementation of character features."""
    
    @property
    def cdf(self) -> pd.DataFrame:
        """Create character dataframe with basic features."""
        if not hasattr(self, '_cdf'):
            chars = list(self.text)
            
            # Add padding if specified
            if self._roll_padding > 0:
                pad_char = '<PAD>'
                chars = [pad_char] * self._roll_padding + chars + [pad_char] * self._roll_padding
            
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

# Usage example
text = "Hello, World! 123"
features = BasicCharacterFeatures(text, roll_padding=2)
print(features.cdf.head(10))
```

### TextFeatures
```python
class TextFeatures:
    def __init__(self, doctext: Union[dk_doc.Text, str])
```

Class for extracting and analyzing text-level features.

**Parameters:**
- `doctext` (Union[dk_doc.Text, str]): Input text or document

**Example:**
```python
class AdvancedTextFeatures(masking_tokenizer.TextFeatures):
    """Advanced text-level features."""
    
    def __init__(self, doctext):
        self.doctext = doctext if hasattr(doctext, 'text') else dk_doc.Text(doctext)
    
    def extract_features(self) -> dict:
        """Extract comprehensive text features."""
        text = self.doctext.text
        
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'alpha_ratio': sum(c.isalpha() for c in text) / len(text) if text else 0,
            'digit_ratio': sum(c.isdigit() for c in text) / len(text) if text else 0,
            'space_ratio': sum(c.isspace() for c in text) / len(text) if text else 0,
            'punct_ratio': sum(not c.isalnum() and not c.isspace() for c in text) / len(text) if text else 0,
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'contains_digits': any(c.isdigit() for c in text),
            'contains_special': any(not c.isalnum() and not c.isspace() for c in text),
            'is_uppercase': text.isupper(),
            'is_lowercase': text.islower(),
            'is_titlecase': text.istitle()
        }

# Usage
text_features = AdvancedTextFeatures("Hello, World! This has 123 numbers.")
features = text_features.extract_features()
print(features)
```

## Masking Functions

### Basic Masking
```python
def mask_text(
    text: str,
    mask_char: str = '*',
    preserve_length: bool = True,
    preserve_structure: bool = True
) -> str
```

Apply basic masking to text while optionally preserving structure.

**Parameters:**
- `text` (str): Input text to mask
- `mask_char` (str, default='*'): Character to use for masking
- `preserve_length` (bool, default=True): Keep original text length
- `preserve_structure` (bool, default=True): Preserve spaces and punctuation

**Returns:** Masked text string

**Example:**
```python
from dataknobs_xization import masking

# Basic masking
text = "John Doe, age 30"
masked = masking.mask_text(text)
print(masked)  # "**** ***, *** **"

# Custom mask character
masked2 = masking.mask_text(text, mask_char='X')
print(masked2)  # "XXXX XXX, XXX XX"

# Don't preserve structure
masked3 = masking.mask_text(text, preserve_structure=False)
print(masked3)  # "*************"
```

### Pattern-Based Masking
```python
def mask_patterns(
    text: str,
    patterns: Dict[str, str],
    mask_char: str = '*'
) -> str
```

Mask text based on regex patterns for specific data types.

**Parameters:**
- `text` (str): Input text
- `patterns` (Dict[str, str]): Dictionary of pattern names to regex patterns
- `mask_char` (str, default='*'): Masking character

**Returns:** Text with patterns masked

**Example:**
```python
import re

# Define patterns for sensitive data
patterns = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}-\d{3}-\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
}

text = "Contact John at john@example.com or 555-123-4567. SSN: 123-45-6789"
masked = masking.mask_patterns(text, patterns)
print(masked)
# "Contact John at ***************** or ************. SSN: ***********"
```

### Entity-Based Masking
```python
def mask_entities(
    text: str,
    entity_types: List[str],
    entity_detector: Callable = None
) -> Tuple[str, List[Dict]]
```

Mask named entities while preserving a mapping for later restoration.

**Parameters:**
- `text` (str): Input text
- `entity_types` (List[str]): Types of entities to mask
- `entity_detector` (Callable): Function to detect entities

**Returns:** Tuple of (masked_text, entity_mapping)

**Example:**
```python
def simple_entity_detector(text: str) -> List[Dict]:
    """Simple entity detector for demonstration."""
    import re
    entities = []
    
    # Detect names (capitalized words)
    for match in re.finditer(r'\b[A-Z][a-z]+\b', text):
        entities.append({
            'text': match.group(),
            'start': match.start(),
            'end': match.end(),
            'type': 'PERSON'
        })
    
    # Detect numbers
    for match in re.finditer(r'\b\d+\b', text):
        entities.append({
            'text': match.group(),
            'start': match.start(),
            'end': match.end(),
            'type': 'NUMBER'
        })
    
    return entities

text = "Alice has 25 apples and Bob has 30 oranges"
masked_text, entity_map = masking.mask_entities(
    text, 
    ['PERSON', 'NUMBER'],
    entity_detector=simple_entity_detector
)
print(f"Masked: {masked_text}")
print(f"Entities: {entity_map}")
# Masked: <PERSON_1> has <NUMBER_1> apples and <PERSON_2> has <NUMBER_2> oranges
```

## Advanced Masking Techniques

### Differential Privacy Masking
```python
class DifferentialPrivacyMasker:
    """Apply differential privacy techniques to text."""
    
    def __init__(self, epsilon: float = 1.0, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
    
    def add_noise(self, value: float) -> float:
        """Add Laplace noise for differential privacy."""
        import numpy as np
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def mask_frequencies(self, text: str) -> Dict[str, int]:
        """Return word frequencies with differential privacy."""
        from collections import Counter
        
        words = text.lower().split()
        true_counts = Counter(words)
        
        # Add noise to counts
        noisy_counts = {}
        for word, count in true_counts.items():
            noisy_count = max(0, int(self.add_noise(count)))
            if noisy_count > 0:
                noisy_counts[word] = noisy_count
        
        return noisy_counts

# Usage
masker = DifferentialPrivacyMasker(epsilon=0.5)
text = "the cat sat on the mat the cat was happy"
noisy_freq = masker.mask_frequencies(text)
print(noisy_freq)
```

### Contextual Masking
```python
class ContextualMasker:
    """Mask text while preserving grammatical structure."""
    
    def __init__(self):
        self.pos_tags = {}  # Part-of-speech tags
    
    def mask_by_pos(self, text: str, mask_pos: List[str]) -> str:
        """Mask words based on part-of-speech tags."""
        # Simplified POS tagging (in practice, use a proper NLP library)
        words = text.split()
        masked_words = []
        
        for word in words:
            # Simplified POS detection
            if word.lower() in ['john', 'mary', 'alice', 'bob']:  # Names
                if 'NOUN' in mask_pos:
                    masked_words.append('[NAME]')
                else:
                    masked_words.append(word)
            elif word.isdigit():  # Numbers
                if 'NUM' in mask_pos:
                    masked_words.append('[NUMBER]')
                else:
                    masked_words.append(word)
            else:
                masked_words.append(word)
        
        return ' '.join(masked_words)
    
    def preserve_syntax(self, text: str) -> str:
        """Mask content while preserving syntactic structure."""
        words = text.split()
        masked = []
        
        for word in words:
            if word.lower() in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to']:
                masked.append(word)  # Keep function words
            else:
                masked.append('*' * len(word))  # Mask content words
        
        return ' '.join(masked)

# Usage
masker = ContextualMasker()
text = "John went to the store and bought 5 apples"
pos_masked = masker.mask_by_pos(text, ['NOUN', 'NUM'])
print(pos_masked)  # "[NAME] went to the store and bought [NUMBER] apples"

syntax_preserved = masker.preserve_syntax(text)
print(syntax_preserved)  # "**** **** to the ***** and ****** * ******"
```

## Character-Level Masking

### Character Feature Masking
```python
class CharacterLevelMasker(masking_tokenizer.CharacterFeatures):
    """Character-level masking with feature preservation."""
    
    def __init__(self, doctext, roll_padding=0, mask_probability=0.15):
        super().__init__(doctext, roll_padding)
        self.mask_probability = mask_probability
    
    @property
    def cdf(self) -> pd.DataFrame:
        """Character dataframe with masking features."""
        if not hasattr(self, '_cdf'):
            chars = list(self.text)
            
            # Add padding
            if self._roll_padding > 0:
                pad_char = '<PAD>'
                chars = [pad_char] * self._roll_padding + chars + [pad_char] * self._roll_padding
            
            # Create base features
            import numpy as np
            np.random.seed(42)  # For reproducibility
            
            self._cdf = pd.DataFrame({
                self.text_col: chars,
                'original_char': chars,
                'position': range(len(chars)),
                'is_alpha': [c.isalpha() if c != '<PAD>' else False for c in chars],
                'is_digit': [c.isdigit() if c != '<PAD>' else False for c in chars],
                'is_upper': [c.isupper() if c != '<PAD>' else False for c in chars],
                'is_lower': [c.islower() if c != '<PAD>' else False for c in chars],
                'is_space': [c.isspace() if c != '<PAD>' else False for c in chars],
                'is_punct': [not c.isalnum() and not c.isspace() if c != '<PAD>' else False for c in chars],
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
    
    def get_mask_positions(self) -> List[int]:
        """Get positions of masked characters."""
        cdf = self.cdf
        mask_positions = cdf[cdf['should_mask'] & ~cdf['is_padding']]['position'].tolist()
        return [pos - self._roll_padding for pos in mask_positions]  # Adjust for padding

# Usage
text = "This is a sample text for masking demonstration."
masker = CharacterLevelMasker(text, roll_padding=1, mask_probability=0.2)
masked_text = masker.get_masked_text()
print(f"Original: {text}")
print(f"Masked:   {masked_text}")
print(f"Mask positions: {masker.get_mask_positions()}")
```

## Integration Patterns

### Document Processing Pipeline
```python
from dataknobs_xization import masking
from dataknobs_structures import document as dk_doc
from dataknobs_utils import file_utils
import json

class DocumentMaskingPipeline:
    """Complete document masking pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.entity_patterns = config.get('entity_patterns', {})
        self.mask_probability = config.get('mask_probability', 0.15)
    
    def process_document(self, doc: dk_doc.Document) -> Dict[str, Any]:
        """Process a single document with multiple masking strategies."""
        text = doc.text
        results = {
            'original_doc': doc,
            'original_text': text,
            'document_id': doc.text_id if hasattr(doc, 'text_id') else None
        }
        
        # Pattern-based masking
        if self.entity_patterns:
            pattern_masked = masking.mask_patterns(text, self.entity_patterns)
            results['pattern_masked'] = pattern_masked
        
        # Character-level masking
        char_masker = CharacterLevelMasker(text, mask_probability=self.mask_probability)
        char_masked = char_masker.get_masked_text()
        results['character_masked'] = char_masked
        results['mask_positions'] = char_masker.get_mask_positions()
        
        # Extract features
        features = char_masker.cdf
        results['character_features'] = {
            'total_chars': len(features),
            'masked_chars': features['should_mask'].sum(),
            'alpha_chars': features['is_alpha'].sum(),
            'digit_chars': features['is_digit'].sum(),
            'space_chars': features['is_space'].sum(),
            'punct_chars': features['is_punct'].sum()
        }
        
        return results
    
    def process_documents(self, documents: List[dk_doc.Document]) -> List[Dict[str, Any]]:
        """Process multiple documents."""
        return [self.process_document(doc) for doc in documents]
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save masking results to file."""
        output_lines = []
        for result in results:
            # Convert to serializable format
            serializable_result = {
                'document_id': result['document_id'],
                'original_text': result['original_text'],
                'pattern_masked': result.get('pattern_masked'),
                'character_masked': result['character_masked'],
                'mask_positions': result['mask_positions'],
                'character_features': result['character_features']
            }
            output_lines.append(json.dumps(serializable_result))
        
        file_utils.write_lines(output_path, output_lines)

# Usage
config = {
    'entity_patterns': {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}-\d{3}-\d{4}\b'
    },
    'mask_probability': 0.1
}

pipeline = DocumentMaskingPipeline(config)

# Process documents
documents = [
    dk_doc.Document("Contact Alice at alice@example.com or 555-123-4567", text_id="doc1"),
    dk_doc.Document("Meeting with Bob tomorrow at 2 PM", text_id="doc2")
]

results = pipeline.process_documents(documents)
pipeline.save_results(results, "masked_documents.jsonl")

for result in results:
    print(f"Doc {result['document_id']}:")
    print(f"  Original: {result['original_text']}")
    print(f"  Pattern:  {result.get('pattern_masked', 'N/A')}")
    print(f"  Character: {result['character_masked']}")
    print()
```

### Privacy-Preserving Analytics
```python
from dataknobs_xization import masking
from collections import Counter
import numpy as np

class PrivacyPreservingAnalytics:
    """Perform analytics on masked text data."""
    
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        self.dp_masker = DifferentialPrivacyMasker(epsilon)
    
    def analyze_masked_corpus(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze a corpus of texts with privacy preservation."""
        # Mask all texts
        masked_texts = []
        total_masks = 0
        
        for text in texts:
            char_masker = CharacterLevelMasker(text, mask_probability=0.15)
            masked_text = char_masker.get_masked_text()
            masked_texts.append(masked_text)
            total_masks += len(char_masker.get_mask_positions())
        
        # Compute statistics with differential privacy
        all_words = ' '.join(masked_texts).split()
        word_freq = self.dp_masker.mask_frequencies(' '.join(texts))
        
        return {
            'total_documents': len(texts),
            'total_masked_characters': total_masks,
            'average_mask_per_doc': total_masks / len(texts),
            'vocabulary_size': len(word_freq),
            'top_words': sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10],
            'privacy_epsilon': self.epsilon
        }

# Usage
analytics = PrivacyPreservingAnalytics(epsilon=0.5)
texts = [
    "This is a confidential document with sensitive information.",
    "Another document containing private data and personal details.",
    "Public information that can be shared without concerns."
]

analysis = analytics.analyze_masked_corpus(texts)
print(f"Analysis with privacy (ε={analysis['privacy_epsilon']}):")
print(f"Documents: {analysis['total_documents']}")
print(f"Masked characters: {analysis['total_masked_characters']}")
print(f"Top words: {analysis['top_words'][:5]}")
```

## Performance and Security Considerations

- Character-level masking is computationally intensive for large texts
- Consider streaming processing for very large documents
- Store masked data separately from original data for security
- Use cryptographically secure random number generators for production masking
- Implement proper key management for reversible masking
- Consider the trade-off between privacy and utility when setting masking parameters

## Best Practices

1. **Data Classification**: Classify data sensitivity before applying masking
2. **Masking Strategy**: Choose appropriate masking strategy based on use case
3. **Testing**: Test masking effectiveness with synthetic data
4. **Documentation**: Document masking procedures and parameters
5. **Access Control**: Implement proper access controls for unmasked data
6. **Audit Trail**: Maintain logs of masking operations
7. **Regular Review**: Regularly review and update masking strategies

The masking module provides comprehensive privacy-preserving text processing capabilities that integrate with the broader dataknobs ecosystem for secure data handling and analysis.