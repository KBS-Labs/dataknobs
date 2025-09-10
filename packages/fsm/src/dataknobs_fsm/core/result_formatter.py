"""Utility for formatting execution results in a standardized way.

This module provides consistent result formatting across Simple and Advanced APIs,
eliminating code duplication and ensuring uniform output structure.
"""

from typing import Any, Dict, List
from ..execution.context import ExecutionContext


class ResultFormatter:
    """Formatter for execution results across different processing modes."""
    
    @staticmethod
    def format_single_result(
        context: ExecutionContext,
        success: bool,
        result: Any = None,
        error: Exception | None = None
    ) -> Dict[str, Any]:
        """Format a single execution result.
        
        Args:
            context: The execution context
            success: Whether execution was successful
            result: Execution result (if any)
            error: Exception if execution failed
            
        Returns:
            Formatted result dictionary
        """
        return {
            'final_state': context.current_state,
            'data': context.data,
            'path': ResultFormatter._get_complete_path(context),
            'success': success,
            'error': str(error) if error else (str(result) if not success and result else None),
            'metadata': context.metadata.copy() if context.metadata else {}
        }
    
    @staticmethod
    def format_batch_result(
        context: ExecutionContext,
        batch_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format batch execution results.
        
        Args:
            context: The execution context
            batch_results: Raw batch results
            
        Returns:
            List of formatted result dictionaries
        """
        formatted_results = []
        
        for i, raw_result in enumerate(batch_results):
            # Check if this item had an error
            error = None
            for err_idx, err_exc in context.batch_errors:
                if err_idx == i:
                    error = err_exc
                    break
            
            formatted_result = {
                'index': i,
                'final_state': raw_result.get('final_state', context.current_state),
                'data': raw_result.get('data', {}),
                'path': raw_result.get('path', []),
                'success': error is None,
                'error': str(error) if error else None
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    @staticmethod
    def format_stream_result(
        context: ExecutionContext,
        chunk_result: Any,
        chunk_index: int
    ) -> Dict[str, Any]:
        """Format a stream chunk result.
        
        Args:
            context: The execution context
            chunk_result: Result from processing a chunk
            chunk_index: Index of the processed chunk
            
        Returns:
            Formatted chunk result dictionary
        """
        return {
            'chunk_index': chunk_index,
            'chunks_processed': context.processed_chunks,
            'current_state': context.current_state,
            'data': chunk_result,
            'metadata': context.metadata.copy() if context.metadata else {}
        }
    
    @staticmethod
    def format_async_result(
        context: ExecutionContext,
        success: bool,
        result: Any = None,
        error: Exception | None = None
    ) -> Dict[str, Any]:
        """Format an async execution result.
        
        This is identical to format_single_result but provided for clarity.
        
        Args:
            context: The execution context
            success: Whether execution was successful
            result: Execution result (if any)
            error: Exception if execution failed
            
        Returns:
            Formatted result dictionary
        """
        return ResultFormatter.format_single_result(context, success, result, error)
    
    @staticmethod
    def format_step_result(
        context: ExecutionContext,
        new_state: str | None = None,
        transition_taken: bool = False
    ) -> Dict[str, Any]:
        """Format a step-by-step execution result.
        
        Args:
            context: The execution context
            new_state: The new state after the step (if any)
            transition_taken: Whether a transition was taken
            
        Returns:
            Formatted step result dictionary
        """
        return {
            'previous_state': context.previous_state,
            'current_state': context.current_state,
            'new_state': new_state,
            'transition_taken': transition_taken,
            'path': ResultFormatter._get_complete_path(context),
            'data': context.data,
            'metadata': context.metadata.copy() if context.metadata else {}
        }
    
    @staticmethod
    def format_error_result(
        context: ExecutionContext,
        error: Exception,
        error_state: str | None = None
    ) -> Dict[str, Any]:
        """Format an error result with context.
        
        Args:
            context: The execution context
            error: The exception that occurred
            error_state: State where error occurred (if known)
            
        Returns:
            Formatted error result dictionary
        """
        return {
            'success': False,
            'error': str(error),
            'error_type': type(error).__name__,
            'error_state': error_state or context.current_state,
            'final_state': context.current_state,
            'path': ResultFormatter._get_complete_path(context),
            'data': context.data,
            'metadata': context.metadata.copy() if context.metadata else {}
        }
    
    @staticmethod
    def _get_complete_path(context: ExecutionContext) -> List[str]:
        """Get the complete state traversal path.
        
        Args:
            context: The execution context
            
        Returns:
            List of state names in traversal order
        """
        # Build complete path from history plus current state
        path = context.state_history.copy() if context.state_history else []
        
        # Add current state if not already in path and if it exists
        if context.current_state and (not path or path[-1] != context.current_state):
            path.append(context.current_state)
        
        return path
    
    @staticmethod
    def format_performance_result(
        context: ExecutionContext,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format result with performance metrics.
        
        Args:
            context: The execution context
            result: Base result dictionary
            
        Returns:
            Result with added performance metrics
        """
        # Add performance stats to existing result
        result['performance'] = context.get_performance_stats()
        result['resource_usage'] = context.get_resource_usage()
        
        return result
