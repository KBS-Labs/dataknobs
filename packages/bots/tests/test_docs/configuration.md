# Configuration Guide

Learn how to configure DynaBot for your use case.

## LLM Configuration

Configure your LLM provider:

### OpenAI

```yaml
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7
  max_tokens: 1000
```

### Anthropic

```yaml
llm:
  provider: anthropic
  model: claude-3-sonnet
  temperature: 0.7
```

## Storage Configuration

Configure conversation storage:

### In-Memory Storage

For development and testing:

```yaml
conversation_storage:
  backend: memory
```

### PostgreSQL Storage

For production:

```yaml
conversation_storage:
  backend: postgres
  host: localhost
  port: 5432
  database: myapp
  user: postgres
  password: ${DB_PASSWORD}
```

## Knowledge Base Configuration

Enable RAG with a knowledge base:

```yaml
knowledge_base:
  enabled: true
  documents_path: /app/docs
  vector_store:
    backend: faiss
    dimension: 1536
    collection: knowledge
  embedding_provider: openai
  embedding_model: text-embedding-3-small
  chunking:
    max_chunk_size: 500
    chunk_overlap: 50
```

## Environment Variables

Use environment variables for secrets:

```bash
export OPENAI_API_KEY=your-key
export DB_PASSWORD=your-password
```

Reference them in config:

```yaml
llm:
  api_key: ${OPENAI_API_KEY}
```
