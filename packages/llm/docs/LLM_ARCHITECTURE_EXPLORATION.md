# DataKnobs LLM Package Architecture Exploration

## Overview
The dataknobs-llm package provides a comprehensive abstraction layer for working with various LLM providers (OpenAI, Anthropic, Ollama, HuggingFace, Echo). It includes advanced prompt engineering, conversation management, and FSM integration.

**Package Structure:** 46 Python files organized into 4 major modules:
- `llm/` - Core LLM provider abstraction
- `prompts/` - Advanced prompt engineering and management
- `conversations/` - Multi-turn conversation handling with branching
- `fsm_integration/` - FSM workflow patterns and integration

---

## 1. MAIN CLASSES & RESPONSIBILITIES

### 1.1 LLM Provider Abstraction (`llm/base.py`)

#### Base Classes
```
LLMProvider (ABC)
├── AsyncLLMProvider - Async operations
└── SyncLLMProvider - Sync operations

LLMAdapter - Format conversion between providers
LLMMiddleware (Protocol) - Request/response processing
```

#### Core Data Classes
- **LLMMessage**: Represents a single message in conversation
  - Fields: `role` (system/user/assistant/function), `content`, `name`, `function_call`, `metadata`
  
- **LLMResponse**: Response from LLM
  - Fields: `content`, `model`, `finish_reason`, `usage`, `function_call`, `metadata`, `created_at`
  
- **LLMStreamResponse**: Streaming response chunks
  - Fields: `delta`, `is_final`, `finish_reason`, `usage`, `metadata`
  
- **LLMConfig**: Configuration dataclass
  - Provider settings: `provider`, `model`, `api_key`, `api_base`
  - Generation params: `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, etc.
  - Mode settings: `mode` (CHAT/TEXT/INSTRUCT/EMBEDDING/FUNCTION), `system_prompt`
  - Function calling: `functions`, `function_call`
  - Streaming: `stream`, `stream_callback`
  - Rate limiting: `rate_limit`, `retry_count`, `retry_delay`, `timeout`
  - Advanced: `seed`, `logit_bias`, `user_id`, `options` (provider-specific)

#### Provider Interface Methods
```python
# Completion Operations
async def complete(messages: List[LLMMessage]) -> LLMResponse
async def stream_complete(messages: List[LLMMessage]) -> AsyncIterator[LLMStreamResponse]

# Embeddings
async def embed(texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]

# Function Calling
async def function_call(messages: List[LLMMessage], functions: List[Dict]) -> LLMResponse

# Prompt Integration
async def render_and_complete(prompt_name: str, params: Dict, prompt_type: str) -> LLMResponse
async def render_and_stream(prompt_name: str, params: Dict, prompt_type: str) -> AsyncIterator[LLMStreamResponse]

# Lifecycle
def initialize() -> None
def close() -> None
def validate_model() -> bool
def get_capabilities() -> List[ModelCapability]
```

#### Enums
- **CompletionMode**: CHAT, TEXT, INSTRUCT, EMBEDDING, FUNCTION
- **ModelCapability**: TEXT_GENERATION, CHAT, EMBEDDINGS, FUNCTION_CALLING, VISION, CODE, JSON_MODE, STREAMING

#### Provider Implementations
- **OpenAIProvider**: GPT-3.5, GPT-4 models with streaming and function calling
- **AnthropicProvider**: Claude models with extended thinking
- **OllamaProvider**: Local LLM support
- **HuggingFaceProvider**: HF Transformers integration
- **EchoProvider**: Debug provider that echoes input
- **LLMProviderFactory**: Factory pattern for provider creation

---

### 1.2 Message and Template Utilities (`llm/utils.py`)

#### MessageTemplate
Flexible template system with two strategies:

```python
@dataclass
class MessageTemplate:
    template: str
    variables: List[str]
    strategy: TemplateStrategy  # SIMPLE or CONDITIONAL
    
    def format(**kwargs) -> str        # Render template
    def partial(**kwargs) -> MessageTemplate  # Fill partial variables
```

**SIMPLE Strategy:**
- Uses Python `str.format()` with `{variable}` syntax
- All variables must be provided
- Clean and straightforward

**CONDITIONAL Strategy:**
- Uses `{{variable}}` (double braces) for variables
- Uses `((conditional content))` for optional sections
- Variables can be optional
- Whitespace-aware substitution
- Example: `"Hello {{name}}((, you have {{count}} messages))"`
  - With name & count: "Hello Alice, you have 5 messages"
  - With name only: "Hello Bob"

#### MessageBuilder
Fluent builder for constructing message sequences:
```python
MessageBuilder()
    .system("You are helpful")
    .user("What is Python?")
    .assistant("Python is a programming language...")
    .from_template(role, template, **vars)
    .build() -> List[LLMMessage]
```

#### ResponseParser
Static methods for extracting information from LLM responses:
- `extract_json()` - Extract JSON objects/arrays
- `extract_code(language)` - Extract code blocks with optional language filter
- `extract_list(numbered)` - Extract bullet or numbered lists
- `extract_sections()` - Extract markdown sections with headers

#### TokenCounter
Token estimation for various models:
- `estimate_tokens(text, model)` - Estimate token count
- `estimate_messages_tokens(messages, model)` - Estimate for message list
- `fits_in_context(text, model, max_tokens)` - Check context window

#### CostCalculator
Cost estimation with pricing data:
- Pricing database for GPT-4, GPT-3.5, Claude models
- `calculate_cost(response, model)` - Calculate from usage
- `estimate_cost(text, model, expected_output_tokens)` - Pre-flight estimate

---

## 2. CONVERSATION HANDLING CAPABILITIES

### 2.1 Conversation Storage & State (`conversations/storage.py`)

#### ConversationNode
```python
@dataclass
class ConversationNode:
    message: LLMMessage
    node_id: str              # Dot-delimited path (e.g., "0.1.2")
    timestamp: datetime
    prompt_name: str | None   # Original prompt template name
    branch_name: str | None   # Human-readable branch label
    metadata: Dict[str, Any]  # Usage stats, model info, RAG metadata
```

#### ConversationState
```python
@dataclass
class ConversationState:
    conversation_id: str
    message_tree: Tree        # Tree-based branching structure
    current_node_id: str      # Current position in tree
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
```

#### Tree-Based Branching
- Messages stored in a tree structure allowing multiple alternative branches
- Node IDs use dot-notation: "" (root), "0", "0.1", "0.1.2", etc.
- Helper functions:
  - `calculate_node_id(node: Tree) -> str` - Get node path
  - `get_node_by_id(tree: Tree, node_id: str) -> Tree | None` - Navigate tree
  - `get_messages_for_llm(tree: Tree, node_id: str) -> List[LLMMessage]` - Get conversation path

#### Storage Abstraction
```python
class ConversationStorage(ABC):
    async def save_conversation(state: ConversationState) -> None
    async def load_conversation(conversation_id: str) -> ConversationState | None
    async def list_conversations(filters: Dict) -> List[Dict]
    async def delete_conversation(conversation_id: str) -> None
```

- **DataknobsConversationStorage**: Adapter for dataknobs backend databases

#### Schema Versioning
- Current version: 1.0.0 (MAJOR.MINOR.PATCH)
- Handles backward compatibility and migrations

---

### 2.2 Conversation Manager (`conversations/manager.py`)

```python
class ConversationManager:
    """Manages multi-turn conversations with persistence and branching."""
    
    # Creation
    @classmethod
    async def create(
        llm: AsyncLLMProvider,
        prompt_builder: AsyncPromptBuilder,
        storage: ConversationStorage,
        system_prompt_name: str | None = None,
        system_params: Dict | None = None,
        metadata: Dict | None = None,
        middleware: List[ConversationMiddleware] | None = None,
        cache_rag_results: bool = False,
        reuse_rag_on_branch: bool = False,
    ) -> ConversationManager
    
    # Resume existing conversation
    @classmethod
    async def resume(
        conversation_id: str,
        llm: AsyncLLMProvider,
        prompt_builder: AsyncPromptBuilder,
        storage: ConversationStorage,
    ) -> ConversationManager
    
    # Message operations
    async def add_message(
        content: str | None = None,
        prompt_name: str | None = None,
        params: Dict | None = None,
        role: str = "user",
        include_rag: bool = True,
    ) -> None
    
    # Completion
    async def complete(
        branch_name: str | None = None,
        save: bool = True,
    ) -> LLMResponse
    
    async def stream_complete(
        branch_name: str | None = None,
        save: bool = True,
    ) -> AsyncIterator[LLMStreamResponse]
    
    # Navigation
    async def switch_to_node(node_id: str) -> None
    async def get_conversation_history() -> List[LLMMessage]
    async def get_branch_names() -> List[str]
```

**Key Features:**
- Multi-turn conversation support
- Tree-based branching for alternative responses
- Prompt library integration
- RAG result caching for branching
- Middleware support for processing
- Automatic persistence

---

### 2.3 Conversation Middleware (`conversations/middleware.py`)

```python
class ConversationMiddleware(ABC):
    async def process_request(
        messages: List[LLMMessage],
        state: ConversationState
    ) -> List[LLMMessage]
    
    async def process_response(
        response: LLMResponse,
        state: ConversationState
    ) -> LLMResponse
```

**Built-in Implementations:**
- **LoggingMiddleware**: Log all requests/responses with conversation ID
- **ContentFilterMiddleware**: Filter inappropriate content
- **ValidationMiddleware**: Validate responses against criteria
- **MetadataMiddleware**: Add/extract metadata

---

### 2.4 Conversation Flow (`conversations/flow/flow.py`)

```python
@dataclass
class FlowState:
    """Single state in a conversation flow."""
    prompt_name: str
    transitions: Dict[str, str]              # condition -> next_state
    transition_conditions: Dict[str, TransitionCondition]
    max_loops: int | None
    prompt_params: Dict[str, Any]
    on_enter: Callable | None
    on_exit: Callable | None
    metadata: Dict[str, Any]

@dataclass
class ConversationFlow:
    """Complete conversation flow definition."""
    name: str
    description: str
    states: Dict[str, FlowState]
    initial_state: str
    final_states: List[str]
    metadata: Dict[str, Any]
```

**Transition Conditions** (`conversations/flow/conditions.py`):
- **AlwaysCondition**: Unconditional transition
- **KeywordCondition**: Match keywords in response
- **RegexCondition**: Match regex patterns
- **CustomCondition**: User-defined evaluation logic
- **JSONCondition**: Check JSON structure/values

---

## 3. PROMPT & MESSAGE TEMPLATE MANAGEMENT

### 3.1 Prompt System Overview (`prompts/__init__.py`)

The prompt library system is feature-rich with:
- **Resource Adapters**: Plug in any data source (dicts, databases, vector stores)
- **Template Rendering**: CONDITIONAL strategy with `{{variables}}` and `((conditionals))`
- **Validation System**: ERROR/WARN/IGNORE levels
- **RAG Integration**: Explicit placement with `{{RAG_CONTENT}}` placeholders
- **Prompt Libraries**: Filesystem, config, composite, versioned
- **Builder Pattern**: PromptBuilder (sync) and AsyncPromptBuilder (async)

### 3.2 Prompt Types & Validation (`prompts/base/types.py`)

```python
class ValidationLevel(Enum):
    ERROR = "error"    # Raise exception
    WARN = "warn"      # Log warning (default)
    IGNORE = "ignore"  # Silently ignore

class TemplateMode(Enum):
    MIXED = "mixed"    # Preprocess (()) then Jinja2
    JINJA2 = "jinja2"  # Pure Jinja2

@dataclass
class ValidationConfig:
    level: ValidationLevel | None
    required_params: Set[str]
    optional_params: Set[str]

class PromptTemplateDict(TypedDict, total=False):
    template: str
    defaults: Dict[str, Any]
    validation: ValidationConfig
    metadata: Dict[str, Any]
    sections: Dict[str, str]
    extends: str                    # Template inheritance
    rag_config_refs: List[str]
    rag_configs: List[RAGConfig]
    template_mode: str

class RAGConfig(TypedDict, total=False):
    adapter_name: str
    query: str                       # May contain {{variables}}
    k: int                          # Number of results
    filters: Dict[str, Any]
    placeholder: str                 # e.g., "RAG_CONTENT"
    header: str
    item_template: str

@dataclass
class RenderResult:
    content: str
    params_used: Dict[str, Any]
    params_missing: List[str]
    validation_warnings: List[str]
    metadata: Dict[str, Any]
    rag_metadata: Dict[str, Any] | None
```

### 3.3 Resource Adapters (`prompts/adapters/`)

Pluggable data sources for parameter resolution and RAG:

```python
class ResourceAdapter(ABC):
    async def search(query: str, k: int, filters: Dict) -> List[Any]
    def is_async() -> bool

# Implementations:
- DictResourceAdapter / AsyncDictResourceAdapter      # Python dicts
- InMemoryAdapter / InMemoryAsyncAdapter             # In-memory storage
- DataknobsBackendAdapter / AsyncDataknobsBackendAdapter  # Database backends
```

### 3.4 Prompt Libraries (`prompts/implementations/`)

```python
class AbstractPromptLibrary(ABC):
    get_system_prompt(name: str) -> PromptTemplateDict | None
    get_user_prompts(name: str) -> List[PromptTemplateDict]
    get_message_index(name: str) -> MessageIndex | None

# Implementations:
- FileSystemPromptLibrary    # YAML/JSON files
- ConfigPromptLibrary        # dataknobs Config objects
- CompositePromptLibrary     # Multiple sources with fallback
- VersionedPromptLibrary     # Version/A-B testing support
```

### 3.5 Prompt Builder (`prompts/builders/`)

```python
class PromptBuilder(BasePromptBuilder):
    """Synchronous builder."""
    
    async def render_system_prompt(
        name: str,
        params: Dict | None = None,
        include_rag: bool = True,
        validation_override: ValidationLevel | None = None,
        return_rag_metadata: bool = False,
        cached_rag: Dict | None = None,
    ) -> RenderResult
    
    async def render_user_prompt(
        name: str,
        index: int = 0,
        params: Dict | None = None,
        include_rag: bool = True,
        validation_override: ValidationLevel | None = None,
        return_rag_metadata: bool = False,
        cached_rag: Dict | None = None,
    ) -> RenderResult
    
    async def render_message_index(
        name: str,
        index: int = 0,
        params: Dict | None = None,
        include_rag: bool = True,
    ) -> List[LLMMessage]

class AsyncPromptBuilder(BasePromptBuilder):
    """Async version of PromptBuilder."""
    # Same methods as PromptBuilder but async
```

### 3.6 Template Rendering (`prompts/rendering/`)

```python
# Conditional template rendering from template_utils.py
def render_conditional_template(template: str, params: Dict[str, Any]) -> str
    """
    Handles:
    - {{variable}} substitution (double braces)
    - ((optional sections)) with whitespace handling
    - Nested conditionals
    - Missing/None values
    """
```

### 3.7 Versioning & A/B Testing (`prompts/versioning/`)

```python
class VersionManager:
    async def create_version(name: str, variant_id: str, template: str) -> str
    async def get_active_version(name: str) -> PromptVersion
    async def promote_version(version_id: str) -> None

class ABTestManager:
    async def create_experiment(name: str, variants: List[PromptVariant]) -> str
    async def get_variant(experiment_id: str, variant_id: str) -> PromptVariant
    async def record_metric(experiment_id: str, variant_id: str, metric: MetricEvent) -> None

class MetricsCollector:
    async def collect_metrics(experiment_id: str) -> PromptMetrics
    async def compare_variants(experiment_id: str) -> Dict[str, float]
```

---

## 4. CONFIGURATION PATTERNS

### 4.1 LLMConfig
- Can be created as dataclass directly: `LLMConfig(provider="openai", model="gpt-4")`
- Can be created from dict: `LLMConfig.from_dict(config_dict)`
- Can be created from dataknobs Config object: `normalize_llm_config(config_obj)`

**Key Pattern:** Flexible config handling
```python
def normalize_llm_config(config: Union[LLMConfig, Config, Dict]) -> LLMConfig
```

### 4.2 Provider Factory Pattern
```python
LLMProviderFactory.create(
    provider="openai",
    model="gpt-4",
    api_key="sk-...",
    temperature=0.7
)
# or
create_llm_provider(config: Union[LLMConfig, Dict, Config]) -> LLMProvider
```

### 4.3 Prompt Library Configuration
```python
library = CompositePromptLibrary([
    FileSystemPromptLibrary(Path("./prompts")),
    ConfigPromptLibrary(dataknobs_config),
])

builder = PromptBuilder(
    library=library,
    adapters={
        'config': DictResourceAdapter(config_dict),
        'docs': DataknobsBackendAdapter(database),
    }
)
```

---

## 5. EXISTING ABSTRACTIONS FOR CHAT/CONVERSATION

### 5.1 Multi-Turn Support
- **ConversationManager**: Manages full conversation lifecycle
- **ConversationState**: Persists conversation state
- **Message History**: Tree-based with branching support

### 5.2 Conversation Flow Abstraction
- **ConversationFlow**: Define states and transitions
- **FlowState**: Individual states with prompt templates
- **TransitionConditions**: Dynamic routing based on response

### 5.3 Middleware for Cross-Cutting Concerns
- Process requests before LLM
- Process responses after LLM
- Logging, validation, filtering, metadata extraction

### 5.4 Limitations/Gaps
1. **No Agent Abstraction**: No built-in agent/tool pattern yet
2. **Tool Calling**: Present in LLMConfig but no high-level abstractions
3. **Agent State**: No dedicated agent state management
4. **Tool Registry**: No built-in function/tool registry
5. **Agent Reasoning**: No chain-of-thought or reasoning patterns built-in
6. **Memory Patterns**: No explicit memory management (buffer, summary, vector)

---

## 6. TOOL/FUNCTION CALLING CAPABILITIES

### 6.1 LLM-Level Support
```python
class LLMConfig:
    functions: List[Dict[str, Any]] | None
    function_call: Union[str, Dict[str, str]] | None  # 'auto', 'none', or {name: ...}

class LLMResponse:
    function_call: Dict[str, Any] | None

class AsyncLLMProvider:
    async def function_call(
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse
```

### 6.2 FSM Integration (`fsm_integration/`)
FSM-based workflow patterns with LLM integration:

```python
class PromptBuilder(ITransformFunction):
    """Build prompts for LLM calls."""
    def transform(data: Dict[str, Any]) -> Dict[str, Any]

class LLMCaller(ITransformFunction):
    """Call an LLM with a prompt."""
    async def transform(data: Dict[str, Any]) -> Dict[str, Any]

class WorkflowType(Enum):
    SIMPLE = "simple"           # Single LLM call
    CHAIN = "chain"             # Sequential LLM calls
    RAG = "rag"                 # Retrieval-augmented
    COT = "cot"                 # Chain-of-thought
    TREE = "tree"               # Tree-of-thought
    AGENT = "agent"             # Agent with tools
    MULTI_AGENT = "multi_agent" # Multiple agents
```

### 6.3 AgentConfig (`fsm_integration/workflows.py`)
```python
@dataclass
class AgentConfig:
    agent_name: str
    role: str
    capabilities: List[str]
    tools: List[Dict[str, Any]] | None
    tool_descriptions: str | None
    memory_type: str | None      # 'buffer', 'summary', 'vector'
    memory_size: int = 10
    planning_enabled: bool = False
    planning_steps: int = 5
```

### 6.4 Limitation
- Tool support exists but is **not fully implemented at the agent level**
- Configuration exists but no higher-level agent abstraction over tools
- Function calling is supported at provider level but requires manual orchestration

---

## 7. STATE MANAGEMENT PATTERNS

### 7.1 Conversation State
```python
@dataclass
class ConversationState:
    conversation_id: str
    message_tree: Tree           # Branching message history
    current_node_id: str         # Current position
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]     # Extensible metadata
```

### 7.2 LLM Provider State
```python
class LLMProvider:
    _client = None               # Provider client (lazy initialized)
    _is_initialized: bool        # Initialization flag
    prompt_builder: PromptBuilder # Optional prompt builder
    config: LLMConfig            # Configuration
```

### 7.3 Middleware State
- Middleware operates on immutable message/response objects
- State changes passed through `ConversationState` parameter

### 7.4 Node Metadata
```python
@dataclass
class ConversationNode:
    metadata: Dict[str, Any]     # For storing:
                                 # - Usage statistics
                                 # - Model info
                                 # - RAG results
                                 # - Timing info
                                 # - Custom data
```

### 7.5 Limitations
1. **No Built-in Agent Memory**: No buffer/summary/vector memory patterns
2. **No Persistent Agent State**: Agent state not persisted to storage
3. **No Session Management**: No multi-session/user isolation
4. **Limited Context Management**: No explicit context window management

---

## 8. ARCHITECTURE INSIGHTS FOR CHATBOT/AGENT ABSTRACTION

### 8.1 Strengths to Build Upon
1. **Clean Provider Abstraction**: LLMProvider base class is well-designed
2. **Flexible Templating**: Conditional templates and RAG integration
3. **Conversation Persistence**: Tree-based branching with storage
4. **Middleware System**: Extensible request/response processing
5. **Modular Design**: Clear separation of concerns

### 8.2 Gaps for Agent Abstraction
1. **No Agent Class**: Need base Agent abstraction
2. **No Tool Registry**: Need way to register and discover tools
3. **No Memory Management**: Need buffer, summary, vector memory patterns
4. **No Agent Loop**: No built-in agent execution loop with tool calling
5. **No Reasoning Patterns**: No chain-of-thought, tree-of-thought built-in
6. **No Agent Metadata**: No tracking of agent capabilities, constraints

### 8.3 Recommended Architecture for Agent Layer
```
Agent (ABC)
├── memory: Memory (buffer/summary/vector)
├── tools: ToolRegistry
├── llm: AsyncLLMProvider
├── prompt_builder: AsyncPromptBuilder
├── state: AgentState (persistent)
├── config: AgentConfig
│
├── async def think(input: str) -> Thought
├── async def decide() -> Action
├── async def execute(action: Action) -> Result
├── async def reflect() -> None
└── async def run(input: str) -> str

Tool (ABC)
├── name: str
├── description: str
├── schema: Dict                  # Function definition
├── async def execute(**kwargs) -> Any

Memory (ABC)
├── async def add(message: str, role: str)
├── async def get_context(k: int) -> List[str]
├── async def summarize() -> str

ToolRegistry
├── register(tool: Tool)
├── get_tool(name: str) -> Tool
├── list_tools() -> List[Tool]
├── to_function_calls() -> List[Dict]  # For LLM
```

### 8.4 Integration Points
1. **With ConversationManager**: Agent could wrap or extend ConversationManager
2. **With Middleware**: Middleware could intercept agent decisions
3. **With FSM**: Agent loop could be FSM-based
4. **With Prompts**: Use prompt library for agent prompts
5. **With Storage**: Use ConversationStorage for agent memory persistence

---

## 9. KEY FILES & THEIR ROLES

| File | Purpose | Key Classes |
|------|---------|------------|
| `llm/base.py` | Provider abstraction | LLMProvider, AsyncLLMProvider, SyncLLMProvider, LLMConfig, LLMMessage |
| `llm/providers.py` | Provider implementations | OpenAIProvider, AnthropicProvider, etc. |
| `llm/utils.py` | Message/template utilities | MessageTemplate, MessageBuilder, ResponseParser, TokenCounter, CostCalculator |
| `conversations/manager.py` | Multi-turn conversation | ConversationManager |
| `conversations/storage.py` | State persistence | ConversationState, ConversationNode, ConversationStorage |
| `conversations/middleware.py` | Request/response processing | ConversationMiddleware, LoggingMiddleware, ValidationMiddleware |
| `conversations/flow/flow.py` | Conversation routing | ConversationFlow, FlowState, TransitionCondition |
| `prompts/base/types.py` | Type definitions | PromptTemplateDict, RAGConfig, ValidationConfig |
| `prompts/builders/prompt_builder.py` | Prompt construction | PromptBuilder, AsyncPromptBuilder |
| `prompts/adapters/*.py` | Data sources | ResourceAdapter implementations |
| `prompts/implementations/*.py` | Template sources | FileSystemPromptLibrary, ConfigPromptLibrary, etc. |
| `fsm_integration/functions.py` | FSM functions | PromptBuilder, LLMCaller |
| `fsm_integration/workflows.py` | Workflow patterns | WorkflowType, LLMStep, AgentConfig |
| `template_utils.py` | Shared utilities | TemplateStrategy, render_conditional_template |

---

## 10. DEPENDENCY FLOW

```
llm/ (no external dataknobs-llm deps)
  └─ uses prompts via import (one-way)

prompts/ (no external dataknobs-llm deps)
  └─ independent module

conversations/ (depends on llm, prompts)
  └─ orchestrates both

fsm_integration/ (depends on llm)
  └─ integrates with FSM

template_utils/ (no deps, used by both llm and prompts)
  └─ shared utilities
```

---

## 11. EXAMPLE USAGE PATTERNS

### Creating an LLM Provider
```python
config = LLMConfig(provider="openai", model="gpt-4", temperature=0.7)
llm = OpenAIProvider(config)
llm.initialize()
response = await llm.complete([LLMMessage(role="user", content="Hello")])
```

### Building Prompts with Templates
```python
template = MessageTemplate.from_conditional(
    "Hello {{name}}((, you have {{count}} messages))"
)
result = template.format(name="Alice", count=5)
```

### Managing Conversations
```python
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    system_prompt_name="helpful_assistant"
)
await manager.add_message(content="What is Python?", role="user")
response = await manager.complete()
```

### Using Conversation Flows
```python
flow = ConversationFlow(
    name="customer_support",
    initial_state="greeting",
    states={
        "greeting": FlowState(
            prompt_name="greeting_prompt",
            transitions={"continue": "ask_issue"},
            transition_conditions={"continue": AlwaysCondition()}
        ),
        "ask_issue": FlowState(
            prompt_name="issue_prompt",
            transitions={"resolved": "resolve"},
            transition_conditions={"resolved": KeywordCondition(["resolved", "solved"])}
        ),
    }
)
```

