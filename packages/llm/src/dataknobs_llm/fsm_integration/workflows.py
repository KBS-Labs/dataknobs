"""LLM workflow pattern implementation.

This module provides pre-configured FSM patterns for LLM-based workflows,
including RAG pipelines, chain-of-thought reasoning, and multi-agent systems.

Note: This module was migrated from dataknobs_fsm.patterns.llm_workflow to
consolidate all LLM functionality in the dataknobs-llm package.
"""

from typing import Any, Dict, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio

from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_llm.llm.base import LLMConfig, LLMMessage, LLMResponse
from dataknobs_llm.llm.providers import create_llm_provider
from dataknobs_llm.llm.utils import (
    MessageTemplate, MessageBuilder, ResponseParser
)


class WorkflowType(Enum):
    """LLM workflow types."""
    SIMPLE = "simple"  # Single LLM call
    CHAIN = "chain"  # Sequential chain of LLM calls
    RAG = "rag"  # Retrieval-augmented generation
    COT = "cot"  # Chain-of-thought reasoning
    TREE = "tree"  # Tree-of-thought reasoning
    AGENT = "agent"  # Agent with tools
    MULTI_AGENT = "multi_agent"  # Multiple cooperating agents


@dataclass 
class LLMStep:
    """Single step in LLM workflow."""
    name: str
    prompt_template: MessageTemplate
    model_config: LLMConfig | None = None  # Override default
    
    # Processing
    pre_processor: Callable[[Any], Any] | None = None
    post_processor: Callable[[LLMResponse], Any] | None = None
    
    # Validation
    validator: Callable[[Any], bool] | None = None
    retry_on_failure: bool = True
    max_retries: int = 3
    
    # Dependencies
    depends_on: List[str] | None = None
    pass_context: bool = True  # Pass previous results
    
    # Output
    output_key: str | None = None  # Key in results dict
    parse_json: bool = False
    extract_code: bool = False


@dataclass
class RAGConfig:
    """Configuration for RAG (Retrieval-Augmented Generation)."""
    retriever_type: str  # 'vector', 'keyword', 'hybrid'
    index_path: str | None = None
    embedding_model: str | None = None
    
    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.7
    rerank: bool = False
    rerank_model: str | None = None
    
    # Context settings
    max_context_length: int = 2000
    context_template: MessageTemplate | None = None
    
    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class AgentConfig:
    """Configuration for agent-based workflows."""
    agent_name: str
    role: str
    capabilities: List[str]
    
    # Tools
    tools: List[Dict[str, Any]] | None = None
    tool_descriptions: str | None = None
    
    # Memory
    memory_type: str | None = None  # 'buffer', 'summary', 'vector'
    memory_size: int = 10
    
    # Planning
    planning_enabled: bool = False
    planning_steps: int = 5
    
    # Reflection
    reflection_enabled: bool = False
    reflection_prompt: MessageTemplate | None = None


@dataclass
class LLMWorkflowConfig:
    """Configuration for LLM workflow."""
    workflow_type: WorkflowType
    steps: List[LLMStep]
    default_model_config: LLMConfig
    
    # Workflow settings
    max_iterations: int = 10
    early_stop_condition: Callable[[Dict[str, Any]], bool] | None = None
    
    # RAG settings (if applicable)
    rag_config: RAGConfig | None = None
    
    # Agent settings (if applicable)
    agent_configs: List[AgentConfig] | None = None
    
    # Memory and context
    maintain_history: bool = True
    max_history_length: int = 20
    context_window: int = 4000
    
    # Output settings
    aggregate_outputs: bool = False
    output_formatter: Callable[[Dict[str, Any]], Any] | None = None
    
    # Error handling
    error_handler: Callable[[Exception, str], Any] | None = None
    fallback_response: str | None = None
    
    # Monitoring
    log_prompts: bool = False
    log_responses: bool = False
    track_tokens: bool = True
    track_cost: bool = False


class VectorRetriever:
    """Simple vector-based retriever for RAG."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.documents = []
        self.embeddings = []
        
    async def index_documents(self, documents: List[str]) -> None:
        """Index documents for retrieval.
        
        Generates embeddings for documents using the configured LLM provider.
        In production, these would be stored in a vector database.
        
        Args:
            documents: List of documents to index
        """
        from dataknobs_fsm.llm.providers import get_provider
        
        self.documents = documents
        
        # Try to use a real embedding provider if available
        if self.config.provider_config:
            try:
                provider = get_provider(self.config.provider_config)
                
                # Check if provider supports embeddings
                if hasattr(provider, 'embed'):
                    # Generate embeddings for all documents
                    self.embeddings = await provider.embed(documents)
                    
                    # Normalize embeddings for cosine similarity
                    self.embeddings = [
                        self._normalize_embedding(emb) for emb in self.embeddings
                    ]
                else:
                    # Fallback to mock embeddings if provider doesn't support them
                    self.embeddings = self._generate_mock_embeddings(documents)
            except Exception as e:
                # Log error and fallback to mock embeddings
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to generate real embeddings: {e}. Using mock embeddings.")
                self.embeddings = self._generate_mock_embeddings(documents)
        else:
            # No provider configured, use mock embeddings
            self.embeddings = self._generate_mock_embeddings(documents)
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize an embedding vector for cosine similarity.
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            Normalized embedding vector
        """
        import math
        
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm == 0:
            return embedding
        return [x / norm for x in embedding]
    
    def _generate_mock_embeddings(self, documents: List[str]) -> List[List[float]]:
        """Generate mock embeddings for testing.
        
        Args:
            documents: Documents to generate embeddings for
            
        Returns:
            Mock embedding vectors
        """
        import hashlib
        
        embeddings = []
        for doc in documents:
            # Create deterministic mock embedding based on document content
            doc_hash = hashlib.sha256(doc.encode()).digest()
            # Convert hash to 768-dimensional embedding (standard size)
            embedding = []
            for i in range(96):  # 768 / 8 = 96
                # Take 8 bytes at a time and convert to float
                if i * 8 < len(doc_hash):
                    byte_chunk = doc_hash[i*8:(i+1)*8]
                    value = sum(b for b in byte_chunk) / 2040.0  # Normalize to ~[0, 1]
                else:
                    # Pad with deterministic values if needed
                    value = (i % 10) / 10.0
                
                # Expand to 8 dimensions
                for j in range(8):
                    embedding.append(value * (1 + j * 0.1))
            
            embeddings.append(self._normalize_embedding(embedding))
        
        return embeddings
        
    async def retrieve(self, query: str, top_k: int = None) -> List[str]:
        """Retrieve relevant documents using semantic similarity.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of most relevant documents
        """
        from dataknobs_fsm.llm.providers import get_provider
        
        top_k = top_k or self.config.top_k
        
        if not self.documents:
            return []
        
        # Generate embedding for query
        query_embedding = None
        
        if self.config.provider_config:
            try:
                provider = get_provider(self.config.provider_config)
                if hasattr(provider, 'embed'):
                    query_embedding = await provider.embed(query)
                    query_embedding = self._normalize_embedding(query_embedding)
            except Exception:
                pass
        
        if query_embedding is None:
            # Fallback to mock embedding
            query_embedding = self._generate_mock_embeddings([query])[0]
        
        # Calculate cosine similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity and return top-k documents
        similarities.sort(reverse=True)
        top_indices = [idx for _, idx in similarities[:top_k]]
        
        return [self.documents[idx] for idx in top_indices]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        if len(vec1) != len(vec2):
            # Handle dimension mismatch by padding or truncating
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        return dot_product  # Already normalized


class LLMWorkflow:
    """LLM workflow orchestrator using FSM pattern."""
    
    def __init__(self, config: LLMWorkflowConfig):
        """Initialize LLM workflow.
        
        Args:
            config: Workflow configuration
        """
        self.config = config
        self._fsm = self._build_fsm()
        self._providers = {}
        self._history = []
        self._context = {}
        self._retriever = None
        
        # Initialize retriever if RAG
        if config.workflow_type == WorkflowType.RAG and config.rag_config:
            self._retriever = VectorRetriever(config.rag_config)
            
    def _build_fsm(self) -> SimpleFSM:
        """Build FSM for LLM workflow."""
        # Add start state
        states = [{'name': 'start', 'type': 'initial', 'is_start': True}]
        arcs = []
        
        if self.config.workflow_type == WorkflowType.SIMPLE:
            # Single LLM call
            states.append({'name': 'llm_call', 'type': 'task'})
            arcs.append({'from': 'start', 'to': 'llm_call', 'name': 'init'})
            arcs.append({'from': 'llm_call', 'to': 'end', 'name': 'complete'})
            
        elif self.config.workflow_type == WorkflowType.CHAIN:
            # Sequential chain
            for i, step in enumerate(self.config.steps):
                state_name = f"step_{step.name}"
                states.append({'name': state_name, 'type': 'task'})
                
                if i == 0:
                    arcs.append({'from': 'start', 'to': state_name, 'name': f'init_{step.name}'})
                else:
                    prev_state = f"step_{self.config.steps[i-1].name}"
                    arcs.append({
                        'from': prev_state,
                        'to': state_name,
                        'name': f'{self.config.steps[i-1].name}_to_{step.name}'
                    })
                    
                if i == len(self.config.steps) - 1:
                    arcs.append({'from': state_name, 'to': 'end', 'name': f'{step.name}_complete'})
                    
        elif self.config.workflow_type == WorkflowType.RAG:
            # RAG pipeline
            states.extend([
                {'name': 'retrieve', 'type': 'task'},
                {'name': 'augment', 'type': 'task'},
                {'name': 'generate', 'type': 'task'}
            ])
            
            arcs.extend([
                {'from': 'start', 'to': 'retrieve', 'name': 'init_retrieval'},
                {'from': 'retrieve', 'to': 'augment', 'name': 'retrieve_to_augment'},
                {'from': 'augment', 'to': 'generate', 'name': 'augment_to_generate'},
                {'from': 'generate', 'to': 'end', 'name': 'generation_complete'}
            ])
            
        elif self.config.workflow_type == WorkflowType.COT:
            # Chain-of-thought reasoning
            states.extend([
                {'name': 'decompose', 'type': 'task'},
                {'name': 'reason', 'type': 'task'},
                {'name': 'synthesize', 'type': 'task'}
            ])
            
            arcs.extend([
                {'from': 'start', 'to': 'decompose', 'name': 'init_decompose'},
                {'from': 'decompose', 'to': 'reason', 'name': 'decompose_to_reason'},
                {'from': 'reason', 'to': 'synthesize', 'name': 'reason_to_synthesize'},
                {'from': 'synthesize', 'to': 'end', 'name': 'synthesis_complete'}
            ])
        
        # Add end state
        states.append({
            'name': 'end',
            'type': 'terminal'
        })
            
        # Build FSM configuration
        fsm_config = {
            'name': 'LLM_Workflow',
            'data_mode': DataHandlingMode.REFERENCE.value,
            'states': states,
            'arcs': arcs,
            'resources': []
        }
        
        return SimpleFSM(fsm_config)
        
    async def _get_provider(self, step: LLMStep | None = None):
        """Get LLM provider for step."""
        config = step.model_config if step and step.model_config else self.config.default_model_config
        
        key = f"{config.provider}_{config.model}"
        if key not in self._providers:
            self._providers[key] = create_llm_provider(config, is_async=True)
            await self._providers[key].initialize()
            
        return self._providers[key]
        
    async def _execute_step(
        self,
        step: LLMStep,
        input_data: Dict[str, Any]
    ) -> Any:
        """Execute a single workflow step.
        
        Args:
            step: Workflow step
            input_data: Input data with template variables
            
        Returns:
            Step output
        """
        # Pre-process input
        if step.pre_processor:
            input_data = step.pre_processor(input_data)
            
        # Format prompt
        prompt = step.prompt_template.format(**input_data)
        
        # Build messages
        builder = MessageBuilder()
        if self.config.default_model_config.system_prompt:
            builder.system(self.config.default_model_config.system_prompt)
            
        # Add history if maintaining
        if self.config.maintain_history and self._history:
            for msg in self._history[-self.config.max_history_length:]:
                builder.messages.append(msg)
                
        builder.user(prompt)
        messages = builder.build()
        
        # Get provider and generate
        provider = await self._get_provider(step)
        
        retry_count = 0
        while retry_count <= step.max_retries:
            try:
                # Generate response
                if self.config.default_model_config.stream:
                    response_text = ""
                    async for chunk in provider.stream_complete(messages):
                        response_text += chunk.delta
                        if self.config.default_model_config.stream_callback:
                            self.config.default_model_config.stream_callback(chunk)
                    response = LLMResponse(content=response_text, model=provider.config.model)
                else:
                    response = await provider.complete(messages)
                    
                # Validate response
                if step.validator and not step.validator(response):
                    if not step.retry_on_failure or retry_count >= step.max_retries:
                        raise ValueError(f"Validation failed for step {step.name}")
                    retry_count += 1
                    continue
                    
                # Parse response if needed
                result = response.content
                if step.parse_json:
                    result = ResponseParser.extract_json(response)
                elif step.extract_code:
                    result = ResponseParser.extract_code(response)
                    
                # Post-process
                if step.post_processor:
                    result = step.post_processor(result)  # type: ignore
                    
                # Update history
                if self.config.maintain_history:
                    self._history.append(LLMMessage(role='user', content=prompt))
                    self._history.append(LLMMessage(role='assistant', content=response.content))
                    
                # Track tokens and cost
                if self.config.track_tokens and response.usage:
                    self._context['total_tokens'] = self._context.get('total_tokens', 0) + response.usage.get('total_tokens', 0)
                    
                return result
                
            except Exception as e:
                if retry_count >= step.max_retries:
                    if self.config.error_handler:
                        return self.config.error_handler(e, step.name)
                    raise
                retry_count += 1
                await asyncio.sleep(1.0 * retry_count)  # Exponential backoff
                
    async def _execute_rag(self, query: str) -> str:
        """Execute RAG workflow.
        
        Args:
            query: User query
            
        Returns:
            Generated response
        """
        if not self._retriever:
            raise ValueError("RAG configuration not provided")
            
        # Retrieve relevant documents
        documents = await self._retriever.retrieve(query)
        
        # Build augmented prompt
        context = "\n\n".join(documents)
        if self.config.rag_config.context_template:
            augmented_prompt = self.config.rag_config.context_template.format(
                context=context,
                query=query
            )
        else:
            augmented_prompt = f"""Context:
{context}

Question: {query}

Answer based on the context provided:"""
        
        # Generate response
        provider = await self._get_provider()
        response = await provider.complete(augmented_prompt)
        
        return response.content
        
    async def _execute_cot(self, problem: str) -> str:
        """Execute chain-of-thought reasoning.
        
        Args:
            problem: Problem to solve
            
        Returns:
            Solution
        """
        provider = await self._get_provider()
        
        # Step 1: Decompose problem
        decompose_prompt = f"""Break down this problem into smaller steps:
{problem}

List the steps needed to solve this:"""
        
        decompose_response = await provider.complete(decompose_prompt)
        steps = ResponseParser.extract_list(decompose_response)
        
        # Step 2: Reason through each step
        reasoning = []
        for i, step in enumerate(steps, 1):
            reason_prompt = f"""Problem: {problem}
Step {i}: {step}

Explain how to complete this step:"""
            
            reason_response = await provider.complete(reason_prompt)
            reasoning.append(f"Step {i}: {step}\n{reason_response.content}")
            
        # Step 3: Synthesize solution
        synthesis_prompt = f"""Problem: {problem}

Reasoning:
{chr(10).join(reasoning)}

Based on the reasoning above, provide the final solution:"""
        
        synthesis_response = await provider.complete(synthesis_prompt)
        
        return synthesis_response.content
        
    async def execute(
        self,
        input_data: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute LLM workflow.
        
        Args:
            input_data: Input data or query
            
        Returns:
            Workflow results
        """
        # Normalize input
        if isinstance(input_data, str):
            input_data = {'query': input_data}
            
        results = {}
        
        if self.config.workflow_type == WorkflowType.SIMPLE:
            # Single step execution
            if self.config.steps:
                output = await self._execute_step(self.config.steps[0], input_data)
                results[self.config.steps[0].output_key or 'output'] = output
            else:
                # Direct LLM call
                provider = await self._get_provider()
                response = await provider.complete(input_data.get('query', ''))
                results['output'] = response.content
                
        elif self.config.workflow_type == WorkflowType.CHAIN:
            # Sequential chain execution
            current_context = input_data.copy()
            
            for step in self.config.steps:
                # Add dependencies to context
                if step.depends_on:
                    for dep in step.depends_on:
                        if dep in results:
                            current_context[dep] = results[dep]
                            
                # Execute step
                output = await self._execute_step(step, current_context)
                
                # Store result
                output_key = step.output_key or step.name
                results[output_key] = output
                
                # Update context if passing
                if step.pass_context:
                    current_context[output_key] = output
                    
        elif self.config.workflow_type == WorkflowType.RAG:
            # RAG pipeline
            output = await self._execute_rag(input_data.get('query', ''))
            results['output'] = output
            
        elif self.config.workflow_type == WorkflowType.COT:
            # Chain-of-thought
            output = await self._execute_cot(input_data.get('problem', input_data.get('query', '')))
            results['output'] = output
            
        # Format output if configured
        if self.config.output_formatter:
            results = self.config.output_formatter(results)
            
        # Add metadata
        if self.config.track_tokens:
            results['_tokens'] = self._context.get('total_tokens', 0)
            
        return results
        
    async def index_documents(self, documents: List[str]) -> None:
        """Index documents for RAG.
        
        Args:
            documents: Documents to index
        """
        if not self._retriever:
            raise ValueError("RAG configuration not provided")
        await self._retriever.index_documents(documents)
        
    async def close(self) -> None:
        """Close all providers."""
        for provider in self._providers.values():
            await provider.close()


def create_simple_llm_workflow(
    prompt_template: str,
    model: str = 'gpt-3.5-turbo',
    provider: str = 'openai',
    **kwargs
) -> LLMWorkflow:
    """Create simple LLM workflow.
    
    Args:
        prompt_template: Prompt template string
        model: Model name
        provider: Provider name
        **kwargs: Additional configuration
        
    Returns:
        Configured LLM workflow
    """
    template = MessageTemplate(prompt_template)
    
    config = LLMWorkflowConfig(
        workflow_type=WorkflowType.SIMPLE,
        steps=[
            LLMStep(
                name='generate',
                prompt_template=template
            )
        ],
        default_model_config=LLMConfig(
            provider=provider,
            model=model,
            **kwargs
        )
    )
    
    return LLMWorkflow(config)


def create_rag_workflow(
    model: str = 'gpt-3.5-turbo',
    provider: str = 'openai',
    retriever_type: str = 'vector',
    top_k: int = 5,
    **kwargs
) -> LLMWorkflow:
    """Create RAG workflow.
    
    Args:
        model: Model name
        provider: Provider name
        retriever_type: Type of retriever
        top_k: Number of documents to retrieve
        **kwargs: Additional configuration
        
    Returns:
        Configured RAG workflow
    """
    config = LLMWorkflowConfig(
        workflow_type=WorkflowType.RAG,
        steps=[],
        default_model_config=LLMConfig(
            provider=provider,
            model=model,
            **kwargs
        ),
        rag_config=RAGConfig(
            retriever_type=retriever_type,
            top_k=top_k
        )
    )
    
    return LLMWorkflow(config)


def create_chain_workflow(
    steps: List[Dict[str, Any]],
    model: str = 'gpt-3.5-turbo',
    provider: str = 'openai',
    **kwargs
) -> LLMWorkflow:
    """Create chain workflow.
    
    Args:
        steps: List of step configurations
        model: Model name
        provider: Provider name
        **kwargs: Additional configuration
        
    Returns:
        Configured chain workflow
    """
    llm_steps = []
    for step_config in steps:
        llm_steps.append(LLMStep(
            name=step_config['name'],
            prompt_template=MessageTemplate(step_config['prompt']),
            output_key=step_config.get('output_key'),
            parse_json=step_config.get('parse_json', False),
            depends_on=step_config.get('depends_on')
        ))
        
    config = LLMWorkflowConfig(
        workflow_type=WorkflowType.CHAIN,
        steps=llm_steps,
        default_model_config=LLMConfig(
            provider=provider,
            model=model,
            **kwargs
        )
    )
    
    return LLMWorkflow(config)
