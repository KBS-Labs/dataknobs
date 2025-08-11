# dataknobs-xization

Text normalization and tokenization tools.

## Installation

```bash
pip install dataknobs-xization
```

## Features

- **Text Normalization**: Standardize text for consistent processing
- **Masking Tokenizer**: Advanced tokenization with masking capabilities
- **Annotations**: Text annotation system
- **Authorities**: Authority management for text processing
- **Lexicon**: Lexicon-based text analysis

## Usage

```python
from dataknobs_xization import normalize, MaskingTokenizer

# Text normalization
normalized = normalize.normalize_text("Hello, World!")

# Tokenization with masking
tokenizer = MaskingTokenizer()
tokens = tokenizer.tokenize("This is a sample text.")

# Working with annotations
from dataknobs_xization import annotations
doc = annotations.create_document("Sample text", {"metadata": "value"})
```

## Dependencies

This package depends on:
- `dataknobs-common`
- `dataknobs-structures`
- `dataknobs-utils`
- nltk

## License

See LICENSE file in the root repository.