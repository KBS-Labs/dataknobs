"""Tests for new pattern implementations."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta


@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

from dataknobs_fsm.patterns.api_orchestration import (
    OrchestrationMode, APIEndpoint, APIOrchestrationConfig,
    RateLimiter, CircuitBreaker, APIOrchestrator,
    create_rest_api_orchestrator, create_graphql_orchestrator
)
from dataknobs_fsm.patterns.llm_workflow import (
    WorkflowType, LLMStep, LLMWorkflowConfig, LLMWorkflow,
    create_simple_llm_workflow, create_rag_workflow, create_chain_workflow
)
from dataknobs_fsm.patterns.error_recovery import (
    RecoveryStrategy, BackoffStrategy, RetryConfig, CircuitBreakerConfig,
    BulkheadConfig, ErrorRecoveryConfig, ErrorRecoveryWorkflow, RetryExecutor,
    CircuitBreaker as ErrorCircuitBreaker, Bulkhead,
    create_retry_workflow, create_circuit_breaker_workflow, create_resilient_workflow
)
from dataknobs_fsm.llm.base import LLMConfig
from dataknobs_fsm.llm.utils import PromptTemplate


class TestAPIOrchestration:
    """Test API orchestration pattern."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter(self):
        """Test rate limiter."""
        limiter = RateLimiter(rate_limit=10, window=0.1)  # 10 requests per 0.1 second
        
        # Should allow first 10 requests immediately
        start = time.time()
        for _ in range(10):
            await limiter.acquire()
        elapsed = time.time() - start
        assert elapsed < 0.05  # Should be fast
        
        # 11th request should be delayed
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        assert elapsed >= 0.05  # Should wait at least 0.05 seconds
        
    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker."""
        breaker = CircuitBreaker(threshold=2, timeout=0.1)
        
        # Successful calls
        async def success():
            return "ok"
            
        result = await breaker.call(success)
        assert result == "ok"
        
        # Failing calls
        async def failure():
            raise ValueError("Error")
            
        # First failure
        with pytest.raises(ValueError):
            await breaker.call(failure)
            
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await breaker.call(failure)
            
        # Circuit should be open now
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await breaker.call(success)
            
        # Wait for timeout
        await asyncio.sleep(0.15)
        
        # Should be half-open, success should close it
        result = await breaker.call(success)
        assert result == "ok"
        
    def test_api_endpoint_creation(self):
        """Test API endpoint configuration."""
        endpoint = APIEndpoint(
            name="test_api",
            url="https://api.example.com/data",
            method="GET",
            headers={"Authorization": "Bearer token"},
            timeout=30.0,
            retry_count=3
        )
        
        assert endpoint.name == "test_api"
        assert endpoint.url == "https://api.example.com/data"
        assert endpoint.method == "GET"
        assert endpoint.retry_count == 3
        
    def test_orchestration_config(self):
        """Test orchestration configuration."""
        endpoints = [
            APIEndpoint(name="api1", url="https://api1.com"),
            APIEndpoint(name="api2", url="https://api2.com")
        ]
        
        config = APIOrchestrationConfig(
            endpoints=endpoints,
            mode=OrchestrationMode.PARALLEL,
            max_concurrent=5,
            global_rate_limit=60
        )
        
        assert len(config.endpoints) == 2
        assert config.mode == OrchestrationMode.PARALLEL
        assert config.max_concurrent == 5
        
    def test_create_rest_api_orchestrator(self):
        """Test REST API orchestrator creation."""
        orchestrator = create_rest_api_orchestrator(
            base_url="https://api.example.com",
            endpoints=[
                {"name": "users", "path": "/users", "method": "GET"},
                {"name": "posts", "path": "/posts", "method": "GET"}
            ],
            auth_token="test-token",
            rate_limit=60,
            mode=OrchestrationMode.SEQUENTIAL
        )
        
        assert orchestrator is not None
        assert isinstance(orchestrator, APIOrchestrator)
        assert len(orchestrator.config.endpoints) == 2
        assert orchestrator.config.mode == OrchestrationMode.SEQUENTIAL
        assert orchestrator.config.global_rate_limit == 60
        
    def test_create_graphql_orchestrator(self):
        """Test GraphQL orchestrator creation."""
        orchestrator = create_graphql_orchestrator(
            endpoint="https://api.example.com/graphql",
            queries=[
                {"name": "getUser", "query": "{ user { id name } }"},
                {"name": "getPosts", "query": "{ posts { id title } }"}
            ],
            auth_token="test-token",
            batch_queries=True
        )
        
        assert orchestrator is not None
        assert isinstance(orchestrator, APIOrchestrator)
        # Batched queries result in single endpoint
        assert len(orchestrator.config.endpoints) == 1
        assert orchestrator.config.endpoints[0].method == 'POST'


class TestLLMWorkflow:
    """Test LLM workflow pattern."""
    
    def test_llm_step_creation(self):
        """Test LLM step configuration."""
        template = PromptTemplate("Process {input}")
        step = LLMStep(
            name="process",
            prompt_template=template,
            parse_json=True,
            max_retries=3
        )
        
        assert step.name == "process"
        assert step.prompt_template == template
        assert step.parse_json is True
        assert step.max_retries == 3
        
    def test_workflow_config(self):
        """Test workflow configuration."""
        template = PromptTemplate("Test {query}")
        steps = [LLMStep(name="test", prompt_template=template)]
        
        config = LLMWorkflowConfig(
            workflow_type=WorkflowType.SIMPLE,
            steps=steps,
            default_model_config=LLMConfig(
                provider="openai",
                model="gpt-3.5-turbo"
            ),
            maintain_history=True,
            track_tokens=True
        )
        
        assert config.workflow_type == WorkflowType.SIMPLE
        assert len(config.steps) == 1
        assert config.maintain_history is True
        
    def test_create_simple_workflow(self):
        """Test simple LLM workflow creation."""
        workflow = create_simple_llm_workflow(
            prompt_template="Answer: {question}",
            model="gpt-3.5-turbo",
            provider="openai"
        )
        
        assert workflow is not None
        assert isinstance(workflow, LLMWorkflow)
        assert workflow.config.workflow_type == WorkflowType.SIMPLE
        assert len(workflow.config.steps) == 1
        assert workflow.config.steps[0].name == 'generate'
        
    def test_create_rag_workflow(self):
        """Test RAG workflow creation."""
        workflow = create_rag_workflow(
            model="gpt-3.5-turbo",
            provider="openai",
            retriever_type="vector",
            top_k=5
        )
        
        assert workflow is not None
        assert isinstance(workflow, LLMWorkflow)
        assert workflow.config.workflow_type == WorkflowType.RAG
        assert workflow.config.rag_config is not None
        assert workflow.config.rag_config.top_k == 5
        assert workflow.config.rag_config.retriever_type == "vector"
        
    def test_create_chain_workflow(self):
        """Test chain workflow creation."""
        steps = [
            {"name": "analyze", "prompt": "Analyze: {input}"},
            {"name": "summarize", "prompt": "Summarize: {analyze}"}
        ]
        
        workflow = create_chain_workflow(
            steps=steps,
            model="gpt-3.5-turbo",
            provider="openai"
        )
        
        assert workflow is not None
        assert isinstance(workflow, LLMWorkflow)
        assert workflow.config.workflow_type == WorkflowType.CHAIN
        assert len(workflow.config.steps) == 2
        assert workflow.config.steps[0].name == "analyze"
        assert workflow.config.steps[1].name == "summarize"


class TestErrorRecovery:
    """Test error recovery pattern."""
    
    def test_retry_config(self):
        """Test retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=1.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            backoff_multiplier=2.0
        )
        
        assert config.max_attempts == 5
        assert config.initial_delay == 1.0
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL
        
    def test_retry_executor_delay_calculation(self):
        """Test retry delay calculation."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            initial_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=10.0
        )
        executor = RetryExecutor(config)
        
        # Exponential backoff
        assert executor._calculate_delay(1) == 1.0  # 1 * 2^0
        assert executor._calculate_delay(2) == 2.0  # 1 * 2^1
        assert executor._calculate_delay(3) == 4.0  # 1 * 2^2
        assert executor._calculate_delay(4) == 8.0  # 1 * 2^3
        assert executor._calculate_delay(5) == 10.0  # Capped at max_delay
        
    @pytest.mark.asyncio
    async def test_retry_executor(self):
        """Test retry executor."""
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
            
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.01,
            backoff_strategy=BackoffStrategy.FIXED
        )
        executor = RetryExecutor(config)
        
        result = await executor.execute(failing_func)
        assert result == "success"
        assert call_count == 3
        
    @pytest.mark.asyncio
    async def test_error_circuit_breaker(self):
        """Test error recovery circuit breaker."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout=0.1
        )
        breaker = ErrorCircuitBreaker(config)
        
        # Successful call
        async def success():
            return "ok"
            
        result = await breaker.call(success)
        assert result == "ok"
        
        # Failing calls to open circuit
        async def failure():
            raise ValueError("Error")
            
        with pytest.raises(ValueError):
            await breaker.call(failure)
            
        with pytest.raises(ValueError):
            await breaker.call(failure)
            
        # Circuit should be open
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await breaker.call(success)
            
    @pytest.mark.asyncio
    async def test_bulkhead(self):
        """Test bulkhead isolation."""
        config = BulkheadConfig(
            max_concurrent=2,
            queue_timeout=0.1,
            track_metrics=True
        )
        bulkhead = Bulkhead(config)
        
        # Function that takes time
        async def slow_func():
            await asyncio.sleep(0.01)
            return "done"
            
        # Should handle concurrent executions
        tasks = [bulkhead.execute(slow_func) for _ in range(2)]
        results = await asyncio.gather(*tasks)
        assert all(r == "done" for r in results)
        
        # Check metrics
        assert bulkhead.metrics['executed'] == 2
        
    def test_create_retry_workflow(self):
        """Test retry workflow creation."""
        workflow = create_retry_workflow(
            max_attempts=5,
            backoff_strategy=BackoffStrategy.EXPONENTIAL
        )
        
        assert workflow is not None
        assert isinstance(workflow, ErrorRecoveryWorkflow)
        assert workflow.config.primary_strategy == RecoveryStrategy.RETRY
        assert workflow.config.retry_config.max_attempts == 5
        assert workflow.config.retry_config.backoff_strategy == BackoffStrategy.EXPONENTIAL
        
    def test_create_circuit_breaker_workflow(self):
        """Test circuit breaker workflow creation."""
        workflow = create_circuit_breaker_workflow(
            failure_threshold=3,
            timeout=60.0
        )
        
        assert workflow is not None
        assert isinstance(workflow, ErrorRecoveryWorkflow)
        assert workflow.config.primary_strategy == RecoveryStrategy.CIRCUIT_BREAKER
        assert workflow.config.circuit_breaker_config.failure_threshold == 3
        assert workflow.config.circuit_breaker_config.timeout == 60.0
        
    def test_create_resilient_workflow(self):
        """Test resilient workflow creation."""
        workflow = create_resilient_workflow(
            primary_strategy=RecoveryStrategy.RETRY,
            enable_circuit_breaker=True,
            enable_fallback=True,
            enable_bulkhead=False
        )
        
        assert workflow is not None
        assert isinstance(workflow, ErrorRecoveryWorkflow)
        assert workflow.config.primary_strategy == RecoveryStrategy.RETRY
        assert workflow.config.circuit_breaker_config is not None
        assert workflow.config.fallback_config is not None
        assert workflow.config.bulkhead_config is None
        assert RecoveryStrategy.FALLBACK in workflow.config.secondary_strategies
        
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """Test error recovery workflow execution."""
        config = ErrorRecoveryConfig(
            primary_strategy=RecoveryStrategy.RETRY,
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay=0.01,
                backoff_strategy=BackoffStrategy.FIXED
            ),
            metrics_enabled=True
        )
        workflow = ErrorRecoveryWorkflow(config)
        
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
            
        result = await workflow.execute(failing_func)
        assert result == "success"
        assert call_count == 3
        
        # Check metrics
        metrics = workflow.get_metrics()
        assert metrics['attempts'] == 1
        assert metrics['successes'] == 1
        assert metrics['failures'] == 0