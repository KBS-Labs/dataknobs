"""Error recovery pattern implementation.

This module provides pre-configured FSM patterns for error recovery and resilience,
including retry strategies, circuit breakers, fallback mechanisms, and compensation.
"""

from typing import Any, Dict, List, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
from datetime import datetime
import random
import logging

from ..api.simple import SimpleFSM
from ..core.data_modes import DataHandlingMode

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"  # Simple retry with backoff
    CIRCUIT_BREAKER = "circuit_breaker"  # Circuit breaker pattern
    FALLBACK = "fallback"  # Use fallback value/service
    COMPENSATE = "compensate"  # Compensation/rollback
    DEADLINE = "deadline"  # Deadline-based timeout
    BULKHEAD = "bulkhead"  # Isolate failures
    CACHE = "cache"  # Use cached results


class BackoffStrategy(Enum):
    """Backoff strategies for retries."""
    FIXED = "fixed"  # Fixed delay
    LINEAR = "linear"  # Linear increase
    EXPONENTIAL = "exponential"  # Exponential increase
    JITTER = "jitter"  # Random jitter added
    DECORRELATED = "decorrelated"  # Decorrelated jitter


@dataclass
class RetryConfig:
    """Configuration for retry strategy."""
    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.1  # 10% jitter
    
    # Retry conditions
    retry_on_exceptions: List[type] | None = None
    retry_on_result: Callable[[Any], bool] | None = None
    
    # Hooks
    on_retry: Callable[[int, Exception], None] | None = None
    on_failure: Callable[[Exception], None] | None = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures to open circuit
    success_threshold: int = 2  # Successes to close circuit
    timeout: float = 60.0  # Time before half-open state
    
    # Monitoring window
    window_size: int = 10  # Rolling window size
    window_duration: float = 60.0  # Window duration in seconds
    
    # Callbacks
    on_open: Callable[[], None] | None = None
    on_close: Callable[[], None] | None = None
    on_half_open: Callable[[], None] | None = None


@dataclass
class FallbackConfig:
    """Configuration for fallback strategy."""
    fallback_value: Any | None = None
    fallback_function: Callable[[Exception], Any] | None = None
    fallback_service: str | None = None  # Alternative service URL
    
    # Cache fallback
    use_cache: bool = False
    cache_ttl: float = 300.0  # 5 minutes
    
    # Conditions
    fallback_on_exceptions: List[type] | None = None
    fallback_on_timeout: bool = True


@dataclass
class CompensationConfig:
    """Configuration for compensation strategy."""
    compensation_actions: List[Callable[[Any], None]]
    save_state: bool = True  # Save state before operation
    
    # Sagas pattern
    use_sagas: bool = False
    saga_timeout: float = 300.0
    
    # Callbacks
    on_compensation_start: Callable[[], None] | None = None
    on_compensation_complete: Callable[[], None] | None = None


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation."""
    max_concurrent: int = 10
    max_queue_size: int = 100
    queue_timeout: float = 30.0
    
    # Thread pool isolation
    use_thread_pool: bool = False
    thread_pool_size: int = 5
    
    # Metrics
    track_metrics: bool = True


@dataclass
class ErrorRecoveryConfig:
    """Configuration for error recovery workflow."""
    primary_strategy: RecoveryStrategy
    secondary_strategies: List[RecoveryStrategy] | None = None
    
    # Strategy configurations
    retry_config: RetryConfig | None = None
    circuit_breaker_config: CircuitBreakerConfig | None = None
    fallback_config: FallbackConfig | None = None
    compensation_config: CompensationConfig | None = None
    bulkhead_config: BulkheadConfig | None = None
    
    # Global settings
    max_total_attempts: int = 10
    global_timeout: float = 300.0
    
    # Error classification
    transient_errors: List[type] | None = None
    permanent_errors: List[type] | None = None
    
    # Monitoring
    log_errors: bool = True
    metrics_enabled: bool = True
    alert_threshold: int = 10  # Errors before alerting


class RetryExecutor:
    """Executor for retry logic."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        
    def _calculate_delay(self, attempt: int, previous_delay: float = None) -> float:
        """Calculate delay for next retry."""
        if self.config.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.config.initial_delay
            
        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.config.initial_delay * attempt
            
        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.config.initial_delay * (self.config.backoff_multiplier ** (attempt - 1))
            
        elif self.config.backoff_strategy == BackoffStrategy.JITTER:
            base_delay = self.config.initial_delay * (self.config.backoff_multiplier ** (attempt - 1))
            jitter = random.uniform(-self.config.jitter_range, self.config.jitter_range)
            delay = base_delay * (1 + jitter)
            
        elif self.config.backoff_strategy == BackoffStrategy.DECORRELATED:
            if previous_delay is None:
                delay = self.config.initial_delay  # type: ignore[unreachable]
            else:
                delay = random.uniform(self.config.initial_delay, previous_delay * 3)
                
        else:
            delay = self.config.initial_delay  # type: ignore[unreachable]
            
        return min(delay, self.config.max_delay)
        
    async def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        previous_delay = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Check if should retry based on result
                if self.config.retry_on_result and self.config.retry_on_result(result):
                    if attempt < self.config.max_attempts:
                        delay = self._calculate_delay(attempt, previous_delay)  # type: ignore
                        previous_delay = delay
                        await asyncio.sleep(delay)
                        continue
                        
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if should retry this exception
                if self.config.retry_on_exceptions:
                    if not any(isinstance(e, exc_type) for exc_type in self.config.retry_on_exceptions):
                        raise
                        
                if attempt < self.config.max_attempts:
                    # Calculate delay
                    delay = self._calculate_delay(attempt, previous_delay)  # type: ignore
                    previous_delay = delay
                    
                    # Call retry hook
                    if self.config.on_retry:
                        self.config.on_retry(attempt, e)
                        
                    await asyncio.sleep(delay)
                else:
                    # Final failure
                    if self.config.on_failure:
                        self.config.on_failure(e)
                    raise
                    
        raise last_exception  # type: ignore


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.window_start = time.time()
        self.window_failures = []
        self._lock = asyncio.Lock()
        
    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check state
            if self.state == CircuitBreakerState.OPEN:
                # Check if should transition to half-open
                if self.last_failure_time:
                    elapsed = time.time() - self.last_failure_time  # type: ignore[unreachable]
                    if elapsed >= self.config.timeout:
                        self.state = CircuitBreakerState.HALF_OPEN
                        if self.config.on_half_open:
                            self.config.on_half_open()
                    else:
                        from ..core.exceptions import CircuitBreakerError
                        raise CircuitBreakerError(wait_time=self.config.timeout - elapsed)
                else:
                    from ..core.exceptions import CircuitBreakerError
                    raise CircuitBreakerError()
                    
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            # Success
            async with self._lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.config.success_threshold:
                        self.state = CircuitBreakerState.CLOSED
                        self.failure_count = 0
                        self.success_count = 0
                        if self.config.on_close:
                            self.config.on_close()
                            
                elif self.state == CircuitBreakerState.CLOSED:
                    # Reset failure count on success
                    self.failure_count = 0
                    
            return result
            
        except Exception:
            # Failure
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Add to window
                self.window_failures.append(time.time())
                
                # Clean old failures from window
                cutoff = time.time() - self.config.window_duration
                self.window_failures = [t for t in self.window_failures if t > cutoff]
                
                # Check if should open circuit
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.OPEN
                    self.success_count = 0
                    if self.config.on_open:
                        self.config.on_open()
                        
                elif self.state == CircuitBreakerState.CLOSED:
                    if len(self.window_failures) >= self.config.failure_threshold:
                        self.state = CircuitBreakerState.OPEN
                        if self.config.on_open:
                            self.config.on_open()
                            
            raise


class Bulkhead:
    """Bulkhead isolation pattern."""
    
    def __init__(self, config: BulkheadConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.active_count = 0
        self.queued_count = 0
        self.metrics = {
            'executed': 0,
            'rejected': 0,
            'timeout': 0
        } if config.track_metrics else None
        
    async def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute function with bulkhead isolation."""
        # Try to acquire semaphore
        try:
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.config.queue_timeout
            )
        except asyncio.TimeoutError:
            if self.metrics:
                self.metrics['timeout'] += 1
            from ..core.exceptions import BulkheadTimeoutError
            raise BulkheadTimeoutError("Bulkhead queue timeout") from None
            
        self.active_count += 1
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            if self.metrics:
                self.metrics['executed'] += 1
                
            return result
            
        finally:
            self.active_count -= 1
            self.semaphore.release()


class ErrorRecoveryWorkflow:
    """Error recovery workflow orchestrator."""
    
    def __init__(self, config: ErrorRecoveryConfig):
        """Initialize error recovery workflow.
        
        Args:
            config: Error recovery configuration
        """
        self.config = config
        self._fsm = self._build_fsm()
        self._retry_executor = None
        self._circuit_breaker = None
        self._bulkhead = None
        self._cache = {}
        self._state_history = []
        self._error_count = 0
        self._metrics = {
            'attempts': 0,
            'successes': 0,
            'failures': 0,
            'fallbacks': 0,
            'compensations': 0
        }
        
        # Initialize components
        if config.retry_config:
            self._retry_executor = RetryExecutor(config.retry_config)
        if config.circuit_breaker_config:
            self._circuit_breaker = CircuitBreaker(config.circuit_breaker_config)
        if config.bulkhead_config:
            self._bulkhead = Bulkhead(config.bulkhead_config)
            
    def _build_fsm(self) -> SimpleFSM:
        """Build FSM for error recovery workflow."""
        # Add start state
        states = [{'name': 'start', 'type': 'initial', 'is_start': True}]
        arcs = []
        
        # Main execution state
        states.append({'name': 'execute', 'type': 'task'})
        arcs.append({'from': 'start', 'to': 'execute', 'name': 'init'})
        
        # Recovery states based on strategy
        if self.config.primary_strategy == RecoveryStrategy.RETRY:
            states.append({'name': 'retry', 'type': 'task'})
            arcs.append({
                'from': 'execute', 
                'to': 'retry', 
                'name': 'on_error',
                'condition': {'type': 'inline', 'code': 'data.get("error") is not None'}  # type: ignore
            })
            arcs.append({'from': 'retry', 'to': 'execute', 'name': 'retry_attempt'})
            arcs.append({'from': 'retry', 'to': 'end', 'name': 'max_retries_reached'})
            
        elif self.config.primary_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            states.append({'name': 'circuit_check', 'type': 'decision'})
            arcs.append({'from': 'execute', 'to': 'circuit_check', 'name': 'check_circuit'})
            arcs.append({'from': 'circuit_check', 'to': 'end', 'name': 'circuit_open'})
            arcs.append({'from': 'circuit_check', 'to': 'execute', 'name': 'circuit_closed'})
            
        elif self.config.primary_strategy == RecoveryStrategy.FALLBACK:
            states.append({'name': 'fallback', 'type': 'task'})
            arcs.append({
                'from': 'execute', 
                'to': 'fallback', 
                'name': 'on_error',
                'condition': {'type': 'inline', 'code': 'data.get("error") is not None'}  # type: ignore
            })
            arcs.append({'from': 'fallback', 'to': 'end', 'name': 'fallback_complete'})
            
        elif self.config.primary_strategy == RecoveryStrategy.COMPENSATE:
            states.extend([
                {'name': 'save_state', 'type': 'task'},
                {'name': 'compensate', 'type': 'task'}
            ])
            arcs.append({'from': 'start', 'to': 'save_state', 'name': 'init'})
            arcs.append({'from': 'save_state', 'to': 'execute', 'name': 'state_saved'})
            arcs.append({
                'from': 'execute', 
                'to': 'compensate', 
                'name': 'on_error',
                'condition': {'type': 'inline', 'code': 'data.get("error") is not None'}  # type: ignore
            })
            arcs.append({'from': 'compensate', 'to': 'end', 'name': 'compensation_complete'})
            
        # Success path
        arcs.append({
            'from': 'execute', 
            'to': 'end', 
            'name': 'success',
            'condition': {'type': 'inline', 'code': 'data.get("error") is None'}  # type: ignore
        })
        
        # Add end state
        states.append({
            'name': 'end',
            'type': 'terminal'
        })
        
        # Build FSM configuration
        fsm_config = {
            'name': 'Error_Recovery',
            'data_mode': DataHandlingMode.COPY.value,
            'states': states,
            'arcs': arcs,
            'resources': []
        }
        
        return SimpleFSM(fsm_config)
        
    async def _execute_with_retry(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute with retry strategy."""
        if not self._retry_executor:
            self._retry_executor = RetryExecutor(self.config.retry_config or RetryConfig())
            
        return await self._retry_executor.execute(func, *args, **kwargs)
        
    async def _execute_with_circuit_breaker(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute with circuit breaker."""
        if not self._circuit_breaker:
            self._circuit_breaker = CircuitBreaker(
                self.config.circuit_breaker_config or CircuitBreakerConfig()
            )
            
        return await self._circuit_breaker.call(func, *args, **kwargs)
        
    async def _execute_with_fallback(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute with fallback."""
        try:
            # Try primary function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            return result
            
        except Exception as e:
            # Check if should use fallback
            if self.config.fallback_config:
                config = self.config.fallback_config
                
                # Check exception type
                if config.fallback_on_exceptions:
                    if not any(isinstance(e, exc_type) for exc_type in config.fallback_on_exceptions):
                        raise
                        
                # Use fallback
                self._metrics['fallbacks'] += 1
                
                if config.fallback_value is not None:
                    return config.fallback_value
                elif config.fallback_function:
                    return config.fallback_function(e)
                elif config.use_cache and self._cache:
                    # Return last cached result
                    return self._cache.get('last_result')
                    
            raise
            
    async def _execute_with_compensation(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute with compensation."""
        # Save state if configured
        saved_state = None
        if self.config.compensation_config and self.config.compensation_config.save_state:
            saved_state = {'args': args, 'kwargs': kwargs, 'timestamp': datetime.now()}
            self._state_history.append(saved_state)
            
        try:
            # Execute function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            return result
            
        except Exception:
            # Execute compensation
            if self.config.compensation_config:
                self._metrics['compensations'] += 1
                
                if self.config.compensation_config.on_compensation_start:
                    self.config.compensation_config.on_compensation_start()
                    
                # Run compensation actions
                for action in self.config.compensation_config.compensation_actions:
                    try:
                        if asyncio.iscoroutinefunction(action):
                            await action(saved_state)
                        else:
                            action(saved_state)
                    except Exception as comp_error:
                        # Log compensation error
                        if self.config.log_errors:
                            logger.error(f"Compensation error: {comp_error}")
                            
                if self.config.compensation_config.on_compensation_complete:
                    self.config.compensation_config.on_compensation_complete()
                    
            raise
            
    async def _execute_with_bulkhead(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute with bulkhead isolation."""
        if not self._bulkhead:
            self._bulkhead = Bulkhead(self.config.bulkhead_config or BulkheadConfig())
            
        return await self._bulkhead.execute(func, *args, **kwargs)
        
    async def execute(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Execute function with error recovery.
        
        Args:
            func: Function to execute
            *args: Any, **kwargs: Any: Function arguments
            
        Returns:
            Function result or fallback value
        """
        self._metrics['attempts'] += 1
        start_time = time.time()
        
        try:
            # Apply primary strategy
            if self.config.primary_strategy == RecoveryStrategy.RETRY:
                result = await self._execute_with_retry(func, *args, **kwargs)
                
            elif self.config.primary_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                result = await self._execute_with_circuit_breaker(func, *args, **kwargs)
                
            elif self.config.primary_strategy == RecoveryStrategy.FALLBACK:
                result = await self._execute_with_fallback(func, *args, **kwargs)
                
            elif self.config.primary_strategy == RecoveryStrategy.COMPENSATE:
                result = await self._execute_with_compensation(func, *args, **kwargs)
                
            elif self.config.primary_strategy == RecoveryStrategy.BULKHEAD:
                result = await self._execute_with_bulkhead(func, *args, **kwargs)
                
            elif self.config.primary_strategy == RecoveryStrategy.DEADLINE:
                # Execute with timeout
                timeout = self.config.global_timeout
                result = await asyncio.wait_for(
                    func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else asyncio.create_task(func(*args, **kwargs)),
                    timeout=timeout
                )
                
            else:
                # Direct execution
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
            # Cache successful result
            if self.config.fallback_config and self.config.fallback_config.use_cache:
                self._cache['last_result'] = result
                self._cache['last_success_time'] = time.time()
                
            # Track execution time
            execution_time = time.time() - start_time
            self._metrics['successes'] += 1
            self._metrics['last_execution_time'] = execution_time
            if 'total_execution_time' not in self._metrics:
                self._metrics['total_execution_time'] = 0
            self._metrics['total_execution_time'] += execution_time
            
            return result
            
        except Exception as e:
            self._error_count += 1
            self._metrics['failures'] += 1
            
            # Track execution time even on failure
            execution_time = time.time() - start_time
            self._metrics['last_execution_time'] = execution_time
            if 'total_execution_time' not in self._metrics:
                self._metrics['total_execution_time'] = 0
            self._metrics['total_execution_time'] += execution_time
            
            # Log error
            if self.config.log_errors:
                logger.error(f"Error in recovery workflow: {e}")
                
            # Check if should alert
            if self._error_count >= self.config.alert_threshold:
                logger.warning(f"Alert: Error threshold reached ({self._error_count} errors)")
                
            # Apply secondary strategies
            if self.config.secondary_strategies:
                for strategy in self.config.secondary_strategies:
                    try:
                        if strategy == RecoveryStrategy.FALLBACK:
                            return await self._execute_with_fallback(func, *args, **kwargs)
                        elif strategy == RecoveryStrategy.CACHE:
                            if 'last_result' in self._cache:
                                return self._cache['last_result']
                    except Exception:
                        continue
                        
            raise
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics.
        
        Returns:
            Metrics dictionary
        """
        metrics = self._metrics.copy()
        
        if self._bulkhead:
            metrics['bulkhead'] = self._bulkhead.metrics
            
        if self._circuit_breaker:
            metrics['circuit_breaker'] = {
                'state': self._circuit_breaker.state.value,
                'failure_count': self._circuit_breaker.failure_count
            }
            
        return metrics


def create_retry_workflow(
    max_attempts: int = 3,
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    **kwargs
) -> ErrorRecoveryWorkflow:
    """Create retry-based error recovery workflow.
    
    Args:
        max_attempts: Maximum retry attempts
        backoff_strategy: Backoff strategy
        **kwargs: Additional configuration
        
    Returns:
        Configured error recovery workflow
    """
    config = ErrorRecoveryConfig(
        primary_strategy=RecoveryStrategy.RETRY,
        retry_config=RetryConfig(
            max_attempts=max_attempts,
            backoff_strategy=backoff_strategy,
            **kwargs
        )
    )
    
    return ErrorRecoveryWorkflow(config)


def create_circuit_breaker_workflow(
    failure_threshold: int = 5,
    timeout: float = 60.0,
    **kwargs
) -> ErrorRecoveryWorkflow:
    """Create circuit breaker workflow.
    
    Args:
        failure_threshold: Failures before opening circuit
        timeout: Time before attempting recovery
        **kwargs: Additional configuration
        
    Returns:
        Configured error recovery workflow
    """
    config = ErrorRecoveryConfig(
        primary_strategy=RecoveryStrategy.CIRCUIT_BREAKER,
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            timeout=timeout,
            **kwargs
        )
    )
    
    return ErrorRecoveryWorkflow(config)


def create_resilient_workflow(
    primary_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    enable_circuit_breaker: bool = True,
    enable_fallback: bool = True,
    enable_bulkhead: bool = False
) -> ErrorRecoveryWorkflow:
    """Create fully resilient workflow with multiple strategies.
    
    Args:
        primary_strategy: Primary recovery strategy
        enable_circuit_breaker: Enable circuit breaker
        enable_fallback: Enable fallback
        enable_bulkhead: Enable bulkhead isolation
        
    Returns:
        Configured error recovery workflow
    """
    secondary_strategies = []
    
    if enable_fallback:
        secondary_strategies.append(RecoveryStrategy.FALLBACK)
    if enable_circuit_breaker and primary_strategy != RecoveryStrategy.CIRCUIT_BREAKER:
        secondary_strategies.append(RecoveryStrategy.CIRCUIT_BREAKER)
        
    config = ErrorRecoveryConfig(
        primary_strategy=primary_strategy,
        secondary_strategies=secondary_strategies,
        retry_config=RetryConfig() if primary_strategy == RecoveryStrategy.RETRY else None,
        circuit_breaker_config=CircuitBreakerConfig() if enable_circuit_breaker else None,
        fallback_config=FallbackConfig(use_cache=True) if enable_fallback else None,
        bulkhead_config=BulkheadConfig() if enable_bulkhead else None
    )
    
    return ErrorRecoveryWorkflow(config)
