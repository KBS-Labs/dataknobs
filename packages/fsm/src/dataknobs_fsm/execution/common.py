"""Common utilities for sync and async execution engines.

This module provides shared logic for both synchronous and asynchronous
execution engines, including network selection, arc scoring, and 
transition selection strategies.
"""

import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from enum import Enum

from dataknobs_fsm.core.arc import ArcDefinition
from dataknobs_fsm.core.fsm import FSM, StateNetwork
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.execution.context import ExecutionContext

if TYPE_CHECKING:
    from dataknobs_fsm.execution.engine import TraversalStrategy


class TransitionSelectionMode(Enum):
    """Mode for selecting transitions."""
    STRATEGY_BASED = "strategy"  # Use TraversalStrategy (depth-first, breadth-first, etc.)
    PRIORITY_SCORING = "scoring"  # Use sophisticated multi-factor scoring
    HYBRID = "hybrid"  # Combine both approaches


class NetworkSelector:
    """Common network selection logic for execution engines."""
    
    @staticmethod
    def get_current_network(
        fsm: FSM,
        context: ExecutionContext,
        enable_intelligent_selection: bool = True
    ) -> StateNetwork | None:
        """Get the current network from context with intelligent selection.
        
        Network selection priority:
        1. If network stack is not empty, use the top network
        2. If a specific network is set in context metadata, use it
        3. Use the main network if defined
        4. If intelligent selection enabled, use context hints and data mode
        5. Fall back to the first available network
        
        Args:
            fsm: The FSM instance.
            context: Execution context.
            enable_intelligent_selection: Whether to use intelligent selection.
            
        Returns:
            Current network or None.
        """
        # Priority 1: Check if we're in a pushed network (network stack)
        if context.network_stack:
            network_name = context.network_stack[-1][0]
            if network_name in fsm.networks:
                return fsm.networks[network_name]
            # Log warning if network not found but continue
            if hasattr(context, 'metadata'):
                context.metadata['network_selection_warning'] = f"Network '{network_name}' not found in stack"
        
        # Priority 2: Check for explicitly set network in metadata
        if hasattr(context, 'metadata') and 'current_network' in context.metadata:
            network_name = context.metadata['current_network']
            if isinstance(network_name, str) and network_name in fsm.networks:
                return fsm.networks[network_name]
        
        # Priority 3: Use main network if defined
        main_network_ref = None
        
        # Handle both wrapper and core FSM structures
        if hasattr(fsm, 'core_fsm'):
            # This is a wrapper FSM
            main_network_ref = fsm.core_fsm.main_network
        else:
            # This is a core FSM
            main_network_ref = getattr(fsm, 'main_network', None)
        
        if main_network_ref:
            # Handle direct network object
            if hasattr(main_network_ref, 'states'):
                return main_network_ref
            # Handle network name reference
            elif isinstance(main_network_ref, str) and main_network_ref in fsm.networks:
                return fsm.networks[main_network_ref]
        
        # If intelligent selection is disabled, just return first network
        if not enable_intelligent_selection:
            if fsm.networks:
                return next(iter(fsm.networks.values()))
            return None
        
        # Priority 4: Select based on context hints
        # Check if there's a preferred network type in metadata
        if hasattr(context, 'metadata'):
            network_type = context.metadata.get('preferred_network_type')
            if network_type:
                # Find networks matching the type
                for network in fsm.networks.values():
                    if hasattr(network, 'type') and network.type == network_type:
                        return network
        
        # Priority 5: Use data mode to select appropriate network
        if hasattr(context, 'data_mode'):
            # Map processing modes to network name patterns
            mode_to_pattern = {
                ProcessingMode.BATCH: ['batch', 'parallel', 'bulk'],
                ProcessingMode.STREAM: ['stream', 'flow', 'pipeline'],
                ProcessingMode.SINGLE: ['main', 'default', 'single']
            }
            
            patterns = mode_to_pattern.get(context.data_mode, [])
            for pattern in patterns:
                for name, network in fsm.networks.items():
                    if pattern in name.lower():
                        return network
        
        # Priority 6: Fallback to first network with initial state
        for network in fsm.networks.values():
            if hasattr(network, 'initial_state') and network.initial_state:
                return network
        
        # Final fallback: Return first available network
        if fsm.networks:
            return next(iter(fsm.networks.values()))
        
        return None


class ArcScorer:
    """Common arc scoring logic for transition selection."""
    
    @staticmethod
    def score_arc(
        arc: ArcDefinition,
        context: ExecutionContext,
        source_state: str,
        include_resource_check: bool = True,
        include_history: bool = True,
        include_load_balancing: bool = True
    ) -> float:
        """Score an arc based on multiple factors.
        
        Factors considered:
        1. Arc priority (higher priority = higher score)
        2. Resource availability (available resources = bonus)
        3. Historical success rate (higher success = higher score)
        4. Load balancing (frequently used = penalty)
        5. Deterministic preference (deterministic = bonus)
        
        Args:
            arc: The arc to score.
            context: Execution context.
            include_resource_check: Whether to include resource availability.
            include_history: Whether to include historical success rate.
            include_load_balancing: Whether to include load balancing.
            
        Returns:
            Numeric score for the arc.
        """
        score = 0.0
        
        # Factor 1: Base priority (weighted heavily)
        priority = getattr(arc, 'priority', 0)
        score += priority * 1000  # High weight for explicit priority
        
        # Factor 2: Resource availability
        if include_resource_check and hasattr(arc, 'resources') and arc.resources:
            # Check if required resources are available
            resources_available = all(
                res in context.resources and 
                context.resources[res].status == 'available'
                for res in arc.resources
            )
            if resources_available:
                score += 100  # Bonus for available resources
        
        # Factor 3: Historical success rate from metadata
        if include_history:
            arc_key = f"arc_{source_state}_{arc.target_state}_stats"
            if arc_key in context.metadata:
                stats = context.metadata[arc_key]
                success_rate = stats.get('success_rate', 0.5)
                score += success_rate * 50  # Weight by success rate
        
        # Factor 4: Load balancing - penalize frequently used arcs
        if include_load_balancing:
            usage_key = f"arc_{source_state}_{arc.target_state}_usage"
            if usage_key in context.metadata:
                usage_count = context.metadata[usage_key]
                # Logarithmic penalty for overuse
                score -= math.log(usage_count + 1) * 10
        
        # Factor 5: Prefer deterministic arcs over non-deterministic
        if hasattr(arc, 'is_deterministic') and arc.is_deterministic:
            score += 25
        
        return score
    
    @staticmethod
    def update_arc_usage(arc: ArcDefinition, context: ExecutionContext, source_state: str) -> None:
        """Update usage statistics for an arc.
        
        Args:
            arc: The arc that was used.
            context: Execution context.
            source_state: The source state of the arc.
        """
        usage_key = f"arc_{source_state}_{arc.target_state}_usage"
        context.metadata[usage_key] = context.metadata.get(usage_key, 0) + 1


class TransitionSelector:
    """Common transition selection logic for execution engines."""
    
    def __init__(
        self,
        mode: TransitionSelectionMode = TransitionSelectionMode.HYBRID,
        default_strategy: Optional['TraversalStrategy'] = None
    ):
        """Initialize transition selector.
        
        Args:
            mode: Selection mode to use.
            default_strategy: Default traversal strategy for strategy-based mode.
        """
        self.mode = mode
        self.default_strategy = default_strategy
    
    def select_transition(
        self,
        available: List[ArcDefinition],
        context: ExecutionContext,
        strategy: Optional['TraversalStrategy'] = None
    ) -> ArcDefinition | None:
        """Select which transition to take from available options.
        
        Args:
            available: Available transitions.
            context: Execution context.
            strategy: Traversal strategy to use (for strategy-based mode).
            
        Returns:
            Selected arc or None.
        """
        if not available:
            return None
        
        # If only one option, return it
        if len(available) == 1:
            selected = available[0]
            state_name = context.current_state or ""
            ArcScorer.update_arc_usage(selected, context, state_name)
            return selected
        
        # Get the effective strategy
        effective_strategy = strategy or self.default_strategy
        
        # Select based on mode
        if self.mode == TransitionSelectionMode.STRATEGY_BASED:
            return self._select_by_strategy(available, context, effective_strategy)
        elif self.mode == TransitionSelectionMode.PRIORITY_SCORING:
            return self._select_by_scoring(available, context)
        else:  # HYBRID mode
            # Use strategy if specified, otherwise use scoring
            if effective_strategy:
                return self._select_by_strategy(available, context, effective_strategy)
            else:
                return self._select_by_scoring(available, context)
    
    def _select_by_strategy(
        self,
        available: List[ArcDefinition],
        context: ExecutionContext,
        strategy: Optional['TraversalStrategy']
    ) -> ArcDefinition | None:
        """Select transition based on traversal strategy.
        
        Args:
            available: Available transitions.
            context: Execution context.
            strategy: Traversal strategy.
            
        Returns:
            Selected arc or None.
        """
        from dataknobs_fsm.execution.engine import TraversalStrategy
        
        selected = None
        
        if strategy == TraversalStrategy.DEPTH_FIRST:
            # Take first available (highest priority)
            selected = available[0]
            
        elif strategy == TraversalStrategy.BREADTH_FIRST:
            # Prefer transitions to unvisited states
            for arc in available:
                if arc.target_state not in context.state_history:
                    selected = arc
                    break
            if not selected:
                selected = available[0]
                
        elif strategy == TraversalStrategy.RESOURCE_OPTIMIZED:
            # Choose transition with least resource requirements
            best_arc = None
            min_resources = float('inf')
            
            for arc in available:
                resource_count = len(arc.resource_requirements) if hasattr(arc, 'resource_requirements') else 0
                if resource_count < min_resources:
                    min_resources = resource_count
                    best_arc = arc
            
            selected = best_arc or available[0]
            
        elif strategy == TraversalStrategy.STREAM_OPTIMIZED:
            # Prefer transitions that support streaming
            for arc in available:
                if hasattr(arc, 'supports_streaming') and arc.supports_streaming:
                    selected = arc
                    break
            if not selected:
                selected = available[0]
        else:
            # Default to first available
            selected = available[0]
        
        state_name = context.current_state or ""
        if selected:
            ArcScorer.update_arc_usage(selected, context, state_name)
        
        return selected
    
    def _select_by_scoring(
        self,
        available: List[ArcDefinition],
        context: ExecutionContext
    ) -> ArcDefinition | None:
        """Select transition based on multi-factor scoring.
        
        Args:
            available: Available transitions.
            context: Execution context.
            
        Returns:
            Selected arc or None.
        """
        # Score each arc
        state_name = context.current_state or ""
        arc_scores = []
        for arc in available:
            score = ArcScorer.score_arc(arc, context, state_name)
            arc_scores.append((arc, score))
        
        # Sort by score (highest first)
        arc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # If top scores are tied, use round-robin for load balancing
        top_score = arc_scores[0][1]
        tied_arcs = [arc for arc, score in arc_scores if abs(score - top_score) < 0.01]
        
        if len(tied_arcs) > 1:
            # Use round-robin selection for tied arcs
            state_name = context.current_state or ""
            round_robin_key = f"state_{state_name}_round_robin"
            current_index = context.metadata.get(round_robin_key, 0)
            selected_arc = tied_arcs[current_index % len(tied_arcs)]
            context.metadata[round_robin_key] = current_index + 1
        else:
            # Return highest scoring arc
            selected_arc = arc_scores[0][0]
        
        # Update usage count
        state_name = context.current_state or ""
        ArcScorer.update_arc_usage(selected_arc, context, state_name)
        
        return selected_arc


def extract_metrics_from_context(context: ExecutionContext) -> Dict[str, Any]:
    """Extract performance metrics from execution context.
    
    Args:
        context: Execution context.
        
    Returns:
        Dictionary of metrics.
    """
    metrics = {}
    
    # Extract arc usage statistics
    arc_metrics = {}
    for key, value in context.metadata.items():
        if key.startswith('arc_') and key.endswith('_usage'):
            arc_name = key.replace('arc_', '').replace('_usage', '')
            arc_metrics[arc_name] = value
    
    if arc_metrics:
        metrics['arc_usage'] = arc_metrics
    
    # Extract network selection warnings
    if 'network_selection_warning' in context.metadata:
        metrics['warnings'] = [context.metadata['network_selection_warning']]
    
    # Extract batch information
    if 'batch_info' in context.metadata:
        metrics['batch'] = context.metadata['batch_info']
    
    # Extract execution times
    exec_times = {}
    for key, value in context.metadata.items():
        if 'execution_time' in key or 'acquisition_time' in key:
            exec_times[key] = value
    
    if exec_times:
        metrics['timing'] = exec_times
    
    return metrics
