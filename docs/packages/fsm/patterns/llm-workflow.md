# LLM Workflow Pattern

## Overview

The LLM Workflow pattern provides pre-configured FSM patterns for Large Language Model workflows including simple generation, sequential chains, RAG (Retrieval-Augmented Generation), Chain-of-Thought reasoning, and multi-agent systems.

## Core Components

### WorkflowType

Supported LLM workflow types:

```python
from dataknobs_fsm.patterns.llm_workflow import WorkflowType

WorkflowType.SIMPLE       # Single LLM call
WorkflowType.CHAIN        # Sequential chain of LLM calls
WorkflowType.RAG          # Retrieval-augmented generation
WorkflowType.COT          # Chain-of-thought reasoning
WorkflowType.TREE         # Tree-of-thought reasoning
WorkflowType.AGENT        # Agent with tools
WorkflowType.MULTI_AGENT  # Multiple cooperating agents
```

### LLMStep

Configuration for individual workflow steps:

```python
from dataknobs_fsm.patterns.llm_workflow import LLMStep
from dataknobs_fsm.llm.utils import PromptTemplate

step = LLMStep(
    name="analysis",
    prompt_template=PromptTemplate("Analyze this text: {text}"),
    model_config=None,  # Use default
    pre_processor=lambda d: {"text": d["content"].strip()},
    post_processor=lambda r: r.content.upper(),
    validator=lambda r: len(r) > 100,
    retry_on_failure=True,
    max_retries=3,
    output_key="analysis",
    parse_json=False,
    extract_code=False
)
```

### LLMWorkflowConfig

Complete workflow configuration:

```python
from dataknobs_fsm.patterns.llm_workflow import LLMWorkflowConfig
from dataknobs_fsm.llm.base import LLMConfig

config = LLMWorkflowConfig(
    workflow_type=WorkflowType.CHAIN,
    steps=[step1, step2, step3],
    default_model_config=LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000
    ),
    max_iterations=10,
    maintain_history=True,
    max_history_length=20,
    context_window=4000,
    track_tokens=True,
    track_cost=False
)
```

## Basic Usage

### Simple LLM Workflow

```python
from dataknobs_fsm.patterns.llm_workflow import (
    LLMWorkflow,
    LLMWorkflowConfig,
    LLMStep,
    WorkflowType
)
from dataknobs_fsm.llm.base import LLMConfig
from dataknobs_fsm.llm.utils import PromptTemplate
import asyncio

async def simple_generation():
    # Configure single LLM call
    step = LLMStep(
        name="generate",
        prompt_template=PromptTemplate("Write a story about {topic}")
    )

    config = LLMWorkflowConfig(
        workflow_type=WorkflowType.SIMPLE,
        steps=[step],
        default_model_config=LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.8
        )
    )

    workflow = LLMWorkflow(config)
    result = await workflow.execute({"topic": "space exploration"})
    return result

# Run workflow
asyncio.run(simple_generation())
```

## Workflow Types

### Sequential Chain

Process data through multiple LLM steps:

```python
# Define chain steps
summarize = LLMStep(
    name="summarize",
    prompt_template=PromptTemplate("Summarize this text: {text}"),
    output_key="summary"
)

translate = LLMStep(
    name="translate",
    prompt_template=PromptTemplate("Translate to Spanish: {summary}"),
    depends_on=["summarize"],
    output_key="translation"
)

polish = LLMStep(
    name="polish",
    prompt_template=PromptTemplate("Polish this translation: {translation}"),
    depends_on=["translate"],
    output_key="final"
)

config = LLMWorkflowConfig(
    workflow_type=WorkflowType.CHAIN,
    steps=[summarize, translate, polish],
    default_model_config=llm_config
)
```

### RAG Pipeline

Retrieval-Augmented Generation workflow:

```python
from dataknobs_fsm.patterns.llm_workflow import RAGConfig

rag_config = RAGConfig(
    retriever_type="vector",
    index_path="/path/to/index",
    embedding_model="text-embedding-ada-002",
    top_k=5,
    similarity_threshold=0.7,
    rerank=True,
    max_context_length=2000,
    chunk_size=500,
    chunk_overlap=50
)

config = LLMWorkflowConfig(
    workflow_type=WorkflowType.RAG,
    steps=[],  # RAG steps are auto-generated
    default_model_config=llm_config,
    rag_config=rag_config
)

workflow = LLMWorkflow(config)

# Index documents
await workflow.index_documents(documents)

# Query with RAG
result = await workflow.execute({"query": "What is the capital of France?"})
```

### Chain-of-Thought Reasoning

Structured reasoning workflow:

```python
config = LLMWorkflowConfig(
    workflow_type=WorkflowType.COT,
    steps=[
        LLMStep(
            name="decompose",
            prompt_template=PromptTemplate(
                "Break down this problem into steps: {problem}"
            )
        ),
        LLMStep(
            name="reason",
            prompt_template=PromptTemplate(
                "Solve each step:\n{steps}"
            )
        ),
        LLMStep(
            name="synthesize",
            prompt_template=PromptTemplate(
                "Combine the solutions:\n{solutions}"
            )
        )
    ],
    default_model_config=llm_config
)
```

### Agent Workflow

LLM agent with tools:

```python
from dataknobs_fsm.patterns.llm_workflow import AgentConfig

agent = AgentConfig(
    agent_name="researcher",
    role="Research Assistant",
    capabilities=["search", "summarize", "analyze"],
    tools=[
        {"name": "search", "description": "Search the web"},
        {"name": "calculator", "description": "Perform calculations"}
    ],
    memory_type="buffer",
    memory_size=10,
    planning_enabled=True,
    reflection_enabled=True
)

config = LLMWorkflowConfig(
    workflow_type=WorkflowType.AGENT,
    steps=[],  # Agent steps are auto-generated
    default_model_config=llm_config,
    agent_configs=[agent]
)
```

## Prompt Templates

### Basic Templates

```python
from dataknobs_fsm.llm.utils import PromptTemplate

# Simple template
template = PromptTemplate("Translate '{text}' to {language}")

# Multi-line template
template = PromptTemplate("""
You are a helpful assistant.
Task: {task}
Context: {context}
Please provide a detailed response.
""")

# Template with defaults
template = PromptTemplate(
    "Analyze {text}",
    defaults={"language": "English"}
)
```

### Dynamic Templates

```python
def create_prompt(data):
    if data.get("technical"):
        return "Provide a technical analysis of: {content}"
    else:
        return "Explain in simple terms: {content}"

step = LLMStep(
    name="analyze",
    prompt_template=PromptTemplate(create_prompt)
)
```

## Response Processing

### JSON Parsing

```python
step = LLMStep(
    name="extract",
    prompt_template=PromptTemplate(
        "Extract data as JSON: {text}"
    ),
    parse_json=True,
    validator=lambda r: "name" in r and "age" in r
)
```

### Code Extraction

```python
step = LLMStep(
    name="generate_code",
    prompt_template=PromptTemplate(
        "Write Python code to: {task}"
    ),
    extract_code=True,
    post_processor=lambda code: compile(code, "generated", "exec")
)
```

### Custom Processing

```python
def process_response(response):
    # Custom processing logic
    lines = response.split("\n")
    return {
        "title": lines[0],
        "content": "\n".join(lines[1:])
    }

step = LLMStep(
    name="format",
    prompt_template=template,
    post_processor=process_response
)
```

## Memory and Context

### Conversation History

```python
config = LLMWorkflowConfig(
    workflow_type=WorkflowType.CHAIN,
    steps=steps,
    default_model_config=llm_config,
    maintain_history=True,
    max_history_length=20,  # Keep last 20 messages
    context_window=4000      # Token limit
)
```

### Context Management

```python
# Pass context between steps
step2 = LLMStep(
    name="continue",
    prompt_template=PromptTemplate(
        "Continue from: {previous_output}"
    ),
    pass_context=True,  # Receive previous step output
    depends_on=["step1"]
)
```

## Error Handling

### Retry Logic

```python
step = LLMStep(
    name="critical",
    prompt_template=template,
    retry_on_failure=True,
    max_retries=3,
    validator=lambda r: validate_response(r)
)
```

### Fallback Responses

```python
config = LLMWorkflowConfig(
    workflow_type=WorkflowType.SIMPLE,
    steps=[step],
    default_model_config=llm_config,
    error_handler=lambda e, s: f"Error in {s}: {str(e)}",
    fallback_response="I apologize, but I cannot process this request."
)
```

## Monitoring and Tracking

### Token Usage

```python
config = LLMWorkflowConfig(
    workflow_type=WorkflowType.CHAIN,
    steps=steps,
    default_model_config=llm_config,
    track_tokens=True,
    track_cost=True
)

workflow = LLMWorkflow(config)
result = await workflow.execute(data)

# Access metrics
metrics = workflow.get_metrics()
print(f"Total tokens: {metrics['total_tokens']}")
print(f"Estimated cost: ${metrics['estimated_cost']}")
```

### Logging

```python
config = LLMWorkflowConfig(
    workflow_type=WorkflowType.CHAIN,
    steps=steps,
    default_model_config=llm_config,
    log_prompts=True,
    log_responses=True
)
```

## Factory Functions

### Create Simple Chain

```python
from dataknobs_fsm.patterns.llm_workflow import create_simple_chain

workflow = create_simple_chain(
    steps=[
        ("summarize", "Summarize: {text}"),
        ("translate", "Translate to French: {summary}"),
        ("polish", "Polish: {translation}")
    ],
    model="gpt-3.5-turbo"
)

result = await workflow.execute({"text": "Long article..."})
```

### Create RAG Pipeline

```python
from dataknobs_fsm.patterns.llm_workflow import create_rag_pipeline

workflow = create_rag_pipeline(
    documents=["doc1.txt", "doc2.txt"],
    embedding_model="text-embedding-ada-002",
    llm_model="gpt-4",
    chunk_size=500
)

answer = await workflow.query("What is the main topic?")
```

## Complete Examples

### Example 1: Document Analysis Pipeline

```python
import asyncio
from dataknobs_fsm.patterns.llm_workflow import (
    LLMWorkflow, LLMWorkflowConfig, LLMStep, WorkflowType
)
from dataknobs_fsm.llm.base import LLMConfig
from dataknobs_fsm.llm.utils import PromptTemplate

async def analyze_document(document_path):
    # Read document
    with open(document_path, 'r') as f:
        content = f.read()

    # Define analysis steps
    extract_topics = LLMStep(
        name="topics",
        prompt_template=PromptTemplate(
            "Extract main topics from this document:\n{content}"
        ),
        output_key="topics"
    )

    sentiment_analysis = LLMStep(
        name="sentiment",
        prompt_template=PromptTemplate(
            "Analyze sentiment of these topics:\n{topics}"
        ),
        depends_on=["topics"],
        output_key="sentiment"
    )

    generate_summary = LLMStep(
        name="summary",
        prompt_template=PromptTemplate(
            "Create executive summary based on:\nTopics: {topics}\nSentiment: {sentiment}"
        ),
        depends_on=["topics", "sentiment"],
        output_key="summary"
    )

    # Configure workflow
    config = LLMWorkflowConfig(
        workflow_type=WorkflowType.CHAIN,
        steps=[extract_topics, sentiment_analysis, generate_summary],
        default_model_config=LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.3
        ),
        aggregate_outputs=True
    )

    # Execute workflow
    workflow = LLMWorkflow(config)
    results = await workflow.execute({"content": content})

    return {
        "topics": results["topics"],
        "sentiment": results["sentiment"],
        "summary": results["summary"]
    }

# Run analysis
result = asyncio.run(analyze_document("report.txt"))
```

### Example 2: Multi-Agent Collaboration

```python
async def multi_agent_research(topic):
    # Define agents
    researcher = AgentConfig(
        agent_name="researcher",
        role="Research Specialist",
        capabilities=["search", "extract", "summarize"]
    )

    analyst = AgentConfig(
        agent_name="analyst",
        role="Data Analyst",
        capabilities=["analyze", "visualize", "interpret"]
    )

    writer = AgentConfig(
        agent_name="writer",
        role="Content Writer",
        capabilities=["write", "edit", "format"]
    )

    # Configure multi-agent workflow
    config = LLMWorkflowConfig(
        workflow_type=WorkflowType.MULTI_AGENT,
        steps=[],  # Auto-generated for agents
        default_model_config=LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.5
        ),
        agent_configs=[researcher, analyst, writer],
        max_iterations=5
    )

    workflow = LLMWorkflow(config)
    report = await workflow.execute({
        "task": f"Create comprehensive report on {topic}",
        "requirements": ["data", "analysis", "recommendations"]
    })

    return report
```

### Example 3: RAG-Enhanced Q&A

```python
async def create_qa_system(knowledge_base_path):
    # Configure RAG
    rag_config = RAGConfig(
        retriever_type="vector",
        index_path=knowledge_base_path,
        embedding_model="text-embedding-ada-002",
        top_k=3,
        similarity_threshold=0.75,
        chunk_size=300,
        chunk_overlap=50,
        context_template=PromptTemplate(
            "Context:\n{context}\n\nQuestion: {question}"
        )
    )

    # Configure workflow
    config = LLMWorkflowConfig(
        workflow_type=WorkflowType.RAG,
        steps=[],
        default_model_config=LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.2,
            system_prompt="You are a helpful Q&A assistant. Answer based on the provided context."
        ),
        rag_config=rag_config,
        maintain_history=True
    )

    workflow = LLMWorkflow(config)

    # Index knowledge base
    documents = load_documents(knowledge_base_path)
    await workflow.index_documents(documents)

    # Q&A function
    async def answer_question(question):
        return await workflow.execute({"question": question})

    return answer_question

# Create and use Q&A system
qa_system = await create_qa_system("knowledge/")
answer = await qa_system("What is the company policy on remote work?")
```

## Best Practices

### 1. Choose Appropriate Workflow Type

- **SIMPLE**: Single generation tasks
- **CHAIN**: Multi-step processing
- **RAG**: Knowledge-grounded responses
- **COT**: Complex reasoning tasks
- **AGENT**: Tasks requiring tools

### 2. Optimize Prompts

```python
# Be specific and clear
template = PromptTemplate("""
Role: You are an expert data analyst.
Task: Analyze the following sales data.
Format: Provide insights as bullet points.
Data: {data}
""")
```

### 3. Validate Responses

```python
def validate_json_response(response):
    try:
        data = json.loads(response)
        return all(k in data for k in ["id", "name", "value"])
    except:
        return False

step = LLMStep(
    name="extract",
    prompt_template=template,
    validator=validate_json_response,
    retry_on_failure=True
)
```

### 4. Manage Context Window

```python
config = LLMWorkflowConfig(
    workflow_type=WorkflowType.CHAIN,
    steps=steps,
    default_model_config=llm_config,
    context_window=4000,  # Stay within model limits
    max_history_length=10  # Limit conversation history
)
```

### 5. Monitor Costs

```python
# Track token usage and costs
config = LLMWorkflowConfig(
    track_tokens=True,
    track_cost=True
)

# Set up alerts
if metrics['estimated_cost'] > 10.0:
    alert_high_cost()
```

## Troubleshooting

### Common Issues

1. **Rate Limits**
   - Implement retry logic
   - Add delays between requests
   - Use multiple API keys

2. **Context Length Exceeded**
   - Reduce prompt size
   - Limit history length
   - Use summarization

3. **Invalid Responses**
   - Add validation
   - Improve prompts
   - Use retry logic

4. **High Costs**
   - Use cheaper models
   - Optimize prompts
   - Cache responses

## Next Steps

- [API Orchestration Pattern](api-orchestration.md) - API integration
- [Error Recovery Pattern](error-recovery.md) - Error handling
- [Resources Guide](../guides/resources.md) - Resource management including LLM
- [Examples](../examples/index.md) - More LLM examples