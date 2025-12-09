# Testing Utilities

Test utilities for dataknobs packages including service availability checks, pytest markers, and configuration factories.

## Table of Contents

- [Service Availability Checks](#service-availability-checks)
- [Pytest Markers](#pytest-markers)
- [Configuration Factories](#configuration-factories)
- [File Helpers](#file-helpers)
- [Usage Examples](#usage-examples)

---

## Service Availability Checks

Functions to check if external services and packages are available.

### Ollama

```python
from dataknobs_common import is_ollama_available, is_ollama_model_available

# Check if Ollama service is running
if is_ollama_available():
    print("Ollama is available")

# Check if a specific model is available
if is_ollama_model_available("nomic-embed-text"):
    print("Embedding model ready")

if is_ollama_model_available("llama3.1:8b"):
    print("LLM model ready")
```

### Packages

```python
from dataknobs_common import (
    is_faiss_available,
    is_chromadb_available,
    is_package_available,
)

# Check vector store packages
if is_faiss_available():
    print("FAISS is installed")

if is_chromadb_available():
    print("ChromaDB is installed")

# Check any package
if is_package_available("torch"):
    print("PyTorch is installed")
```

### Services

```python
from dataknobs_common import is_redis_available

# Check Redis connection
if is_redis_available(host="localhost", port=6379):
    print("Redis is available")
```

---

## Pytest Markers

Pre-configured pytest markers to skip tests when dependencies are unavailable.

### Basic Markers

```python
import pytest
from dataknobs_common import (
    requires_ollama,
    requires_faiss,
    requires_chromadb,
    requires_redis,
)

@requires_ollama
def test_with_ollama():
    """Skipped if Ollama is not available."""
    # Your test code
    pass

@requires_faiss
def test_with_faiss():
    """Skipped if FAISS is not installed."""
    pass

@requires_chromadb
def test_with_chromadb():
    """Skipped if ChromaDB is not installed."""
    pass

@requires_redis
def test_with_redis():
    """Skipped if Redis is not available."""
    pass
```

### Dynamic Markers

```python
from dataknobs_common import requires_package, requires_ollama_model

@requires_package("torch")
def test_with_pytorch():
    """Skipped if PyTorch is not installed."""
    import torch
    # Test code
    pass

@requires_ollama_model("llama3.1:8b")
def test_with_llama():
    """Skipped if specific model is not available."""
    # Test code
    pass
```

### Manual Skipif

You can also use the availability functions directly:

```python
import pytest
from dataknobs_common import is_ollama_available

@pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama not available"
)
def test_ollama_integration():
    # Your test
    pass
```

---

## Configuration Factories

Pre-built configuration dictionaries for testing.

### Bot Configuration

```python
from dataknobs_common import get_test_bot_config

# Basic test config with echo LLM
config = get_test_bot_config()
# Returns:
# {
#     "llm": {"provider": "echo", "model": "test", "temperature": 0.7},
#     "conversation_storage": {"backend": "memory"}
# }

# With real LLM (for integration tests)
config = get_test_bot_config(use_echo_llm=False)

# With memory
config = get_test_bot_config(include_memory=True)

# With system prompt
config = get_test_bot_config(
    system_prompt="You are a test assistant."
)

# Full customization
config = get_test_bot_config(
    use_echo_llm=True,
    use_in_memory_storage=True,
    include_memory=True,
    system_prompt="Custom prompt"
)
```

### RAG Configuration

```python
from dataknobs_common import get_test_rag_config

# Basic RAG config
rag_config = get_test_rag_config()

# With FAISS backend (for persistence)
rag_config = get_test_rag_config(use_in_memory_store=False)

# Custom embedding
rag_config = get_test_rag_config(
    embedding_provider="openai",
    embedding_model="text-embedding-3-small"
)
```

---

## File Helpers

Create test files for ingestion testing.

### Markdown Files

```python
from dataknobs_common import create_test_markdown_files

def test_document_ingestion(tmp_path):
    # Create test markdown files
    files = create_test_markdown_files(tmp_path)

    # files contains paths to two markdown documents
    assert len(files) == 2

    # Use for testing
    for file_path in files:
        # Process documents
        pass
```

### JSON Files

```python
from dataknobs_common import create_test_json_files

def test_json_processing(tmp_path):
    # Create test JSON files
    files = create_test_json_files(tmp_path)

    # files contains paths to two JSON documents
    assert len(files) == 2
```

---

## Usage Examples

### Complete Test Setup

```python
# conftest.py
import pytest
from dataknobs_common import (
    is_ollama_available,
    get_test_bot_config,
    create_test_markdown_files,
)

@pytest.fixture
def bot_config():
    """Provide test bot configuration."""
    return get_test_bot_config(
        include_memory=True,
        system_prompt="Test assistant"
    )

@pytest.fixture
def test_documents(tmp_path):
    """Create test documents."""
    return create_test_markdown_files(tmp_path)

# Skip entire module if Ollama unavailable
pytestmark = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama required for integration tests"
)
```

### Unit Test Example

```python
# test_bot.py
import pytest
from dataknobs_bots import DynaBot, BotContext
from dataknobs_common import get_test_bot_config, requires_ollama

class TestDynaBot:
    @pytest.mark.asyncio
    async def test_chat_with_echo(self):
        """Test chat with echo provider (no external deps)."""
        config = get_test_bot_config()
        bot = await DynaBot.from_config(config)

        context = BotContext(
            conversation_id="test-1",
            client_id="test"
        )

        response = await bot.chat("Hello", context)
        assert response is not None

    @requires_ollama
    @pytest.mark.asyncio
    async def test_chat_with_ollama(self):
        """Test chat with Ollama (skipped if unavailable)."""
        config = get_test_bot_config(use_echo_llm=False)
        config["llm"]["provider"] = "ollama"
        config["llm"]["model"] = "gemma3:1b"

        bot = await DynaBot.from_config(config)
        # ...
```

### Integration Test Example

```python
# test_rag_integration.py
import pytest
from dataknobs_common import (
    requires_ollama,
    requires_ollama_model,
    get_test_bot_config,
    get_test_rag_config,
    create_test_markdown_files,
)

@requires_ollama
@requires_ollama_model("nomic-embed-text")
class TestRAGIntegration:
    @pytest.fixture
    def rag_bot_config(self, tmp_path):
        """Create RAG bot configuration with test docs."""
        docs = create_test_markdown_files(tmp_path)

        config = get_test_bot_config(use_echo_llm=False)
        config["llm"]["provider"] = "ollama"
        config["llm"]["model"] = "gemma3:1b"
        config["knowledge_base"] = get_test_rag_config()
        config["knowledge_base"]["documents_path"] = str(tmp_path)

        return config

    @pytest.mark.asyncio
    async def test_rag_query(self, rag_bot_config):
        """Test RAG retrieval."""
        # Test implementation
        pass
```

---

## Best Practices

1. **Use Echo LLM for Unit Tests**: Avoid external dependencies in unit tests.

2. **Mark Integration Tests**: Use markers to identify tests requiring services.

3. **Organize by Dependency**: Group tests by their external requirements.

4. **CI/CD Considerations**: Run integration tests only when services available.

5. **Fixture Composition**: Combine configuration factories with pytest fixtures.

---

## See Also

- [pytest documentation](https://docs.pytest.org/)
- [dataknobs_bots USER_GUIDE.md](../../bots/docs/USER_GUIDE.md)
