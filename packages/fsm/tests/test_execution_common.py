"""Tests for execution/common.py module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataknobs_fsm.execution.common import (
    NetworkSelector,
    ArcScorer,
    TransitionSelector,
    TransitionSelectionMode,
    extract_metrics_from_context
)
from dataknobs_fsm.core.arc import ArcDefinition
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.engine import TraversalStrategy


class TestNetworkSelector:
    """Tests for NetworkSelector class."""
    
    def test_get_current_network_from_stack(self):
        """Test network selection from network stack."""
        fsm = Mock()
        fsm.networks = {
            'network1': Mock(name='network1'),
            'network2': Mock(name='network2')
        }
        
        context = Mock()
        context.network_stack = [('network2', 'return_state')]
        context.metadata = {}
        
        network = NetworkSelector.get_current_network(fsm, context)
        assert network == fsm.networks['network2']
    
    def test_get_current_network_from_metadata(self):
        """Test network selection from context metadata."""
        fsm = Mock()
        fsm.networks = {
            'network1': Mock(name='network1'),
            'network2': Mock(name='network2')
        }
        
        context = Mock()
        context.network_stack = []
        context.metadata = {'current_network': 'network1'}
        
        network = NetworkSelector.get_current_network(fsm, context)
        assert network == fsm.networks['network1']
    
    def test_get_current_network_main_network_core_fsm(self):
        """Test network selection using main_network from core FSM."""
        network1 = Mock(name='network1')
        network1.states = {'state1': Mock()}
        
        fsm = Mock()
        fsm.networks = {'network1': network1}
        fsm.main_network = 'network1'
        del fsm.core_fsm  # Ensure it's treated as core FSM
        
        context = Mock()
        context.network_stack = []
        context.metadata = {}
        
        network = NetworkSelector.get_current_network(fsm, context)
        assert network == network1
    
    def test_get_current_network_main_network_wrapper_fsm(self):
        """Test network selection using main_network from wrapper FSM."""
        network1 = Mock(name='network1')
        network1.states = {'state1': Mock()}
        
        core_fsm = Mock()
        core_fsm.main_network = 'network1'
        
        fsm = Mock()
        fsm.core_fsm = core_fsm
        fsm.networks = {'network1': network1}
        
        context = Mock()
        context.network_stack = []
        context.metadata = {}
        
        network = NetworkSelector.get_current_network(fsm, context)
        assert network == network1
    
    def test_get_current_network_intelligent_selection_disabled(self):
        """Test network selection with intelligent selection disabled."""
        network1 = Mock(name='network1')
        network2 = Mock(name='network2')
        
        fsm = Mock()
        fsm.networks = {'network1': network1, 'network2': network2}
        del fsm.core_fsm
        fsm.main_network = None
        
        context = Mock()
        context.network_stack = []
        context.metadata = {}
        
        network = NetworkSelector.get_current_network(fsm, context, enable_intelligent_selection=False)
        assert network in [network1, network2]  # Should return first network
    
    def test_get_current_network_by_type_preference(self):
        """Test network selection by preferred network type."""
        network1 = Mock(name='network1')
        network1.type = 'batch'
        network2 = Mock(name='network2')
        network2.type = 'stream'
        
        fsm = Mock()
        fsm.networks = {'network1': network1, 'network2': network2}
        del fsm.core_fsm
        fsm.main_network = None
        
        context = Mock()
        context.network_stack = []
        context.metadata = {'preferred_network_type': 'stream'}
        
        network = NetworkSelector.get_current_network(fsm, context)
        assert network == network2
    
    def test_get_current_network_by_data_mode_batch(self):
        """Test network selection based on batch data mode."""
        batch_network = Mock(name='batch_network')
        other_network = Mock(name='other_network')
        
        fsm = Mock()
        fsm.networks = {
            'batch_network': batch_network,
            'other_network': other_network
        }
        del fsm.core_fsm
        fsm.main_network = None
        
        context = Mock()
        context.network_stack = []
        context.metadata = {}
        context.data_mode = ProcessingMode.BATCH
        
        network = NetworkSelector.get_current_network(fsm, context)
        assert network == batch_network
    
    def test_get_current_network_by_data_mode_stream(self):
        """Test network selection based on stream data mode."""
        stream_network = Mock(name='stream_processing')
        other_network = Mock(name='other_network')
        
        fsm = Mock()
        fsm.networks = {
            'stream_processing': stream_network,
            'other_network': other_network
        }
        del fsm.core_fsm
        fsm.main_network = None
        
        context = Mock()
        context.network_stack = []
        context.metadata = {}
        context.data_mode = ProcessingMode.STREAM
        
        network = NetworkSelector.get_current_network(fsm, context)
        assert network == stream_network
    
    def test_get_current_network_fallback_to_initial_state(self):
        """Test network selection fallback to network with initial state."""
        network1 = Mock(name='network1')
        network1.initial_state = None
        network2 = Mock(name='network2')
        network2.initial_state = 'start'
        
        fsm = Mock()
        fsm.networks = {'network1': network1, 'network2': network2}
        del fsm.core_fsm
        fsm.main_network = None
        
        context = Mock()
        context.network_stack = []
        context.metadata = {}
        del context.data_mode
        
        network = NetworkSelector.get_current_network(fsm, context)
        assert network == network2
    
    def test_get_current_network_empty_fsm(self):
        """Test network selection with empty FSM."""
        fsm = Mock()
        fsm.networks = {}
        del fsm.core_fsm
        fsm.main_network = None
        
        context = Mock()
        context.network_stack = []
        context.metadata = {}
        
        network = NetworkSelector.get_current_network(fsm, context)
        assert network is None
    
    def test_get_current_network_warning_on_missing_stack_network(self):
        """Test warning is added when network in stack is not found."""
        fsm = Mock()
        fsm.networks = {'network1': Mock()}
        
        context = Mock()
        context.network_stack = [('missing_network', 'return_state')]
        context.metadata = {}
        
        NetworkSelector.get_current_network(fsm, context)
        assert 'network_selection_warning' in context.metadata
        assert 'missing_network' in context.metadata['network_selection_warning']


class TestArcScorer:
    """Tests for ArcScorer class."""
    
    def test_score_arc_basic_priority(self):
        """Test arc scoring with basic priority."""
        arc = Mock()
        arc.priority = 5
        arc.target_state = 'state2'
        del arc.resources
        del arc.is_deterministic
        
        context = Mock()
        context.metadata = {}
        context.resources = {}
        
        score = ArcScorer.score_arc(arc, context, 'state1')
        assert score == 5000  # priority * 1000
    
    def test_score_arc_with_resources_available(self):
        """Test arc scoring with available resources."""
        arc = Mock()
        arc.priority = 0
        arc.target_state = 'state2'
        arc.resources = ['resource1', 'resource2']
        del arc.is_deterministic
        
        resource1 = Mock()
        resource1.status = 'available'
        resource2 = Mock()
        resource2.status = 'available'
        
        context = Mock()
        context.metadata = {}
        context.resources = {
            'resource1': resource1,
            'resource2': resource2
        }
        
        score = ArcScorer.score_arc(arc, context, 'state1')
        assert score == 100  # resource bonus
    
    def test_score_arc_with_resources_unavailable(self):
        """Test arc scoring with unavailable resources."""
        arc = Mock()
        arc.priority = 0
        arc.target_state = 'state2'
        arc.resources = ['resource1']
        del arc.is_deterministic
        
        resource1 = Mock()
        resource1.status = 'busy'
        
        context = Mock()
        context.metadata = {}
        context.resources = {'resource1': resource1}
        
        score = ArcScorer.score_arc(arc, context, 'state1')
        assert score == 0  # no resource bonus
    
    def test_score_arc_with_history(self):
        """Test arc scoring with historical success rate."""
        arc = Mock()
        arc.priority = 0
        arc.target_state = 'state2'
        del arc.resources
        del arc.is_deterministic
        
        context = Mock()
        context.metadata = {
            'arc_state1_state2_stats': {'success_rate': 0.8}
        }
        context.resources = {}
        
        score = ArcScorer.score_arc(arc, context, 'state1')
        assert score == 40  # 0.8 * 50
    
    def test_score_arc_with_load_balancing(self):
        """Test arc scoring with load balancing penalty."""
        arc = Mock()
        arc.priority = 0
        arc.target_state = 'state2'
        del arc.resources
        del arc.is_deterministic
        
        context = Mock()
        context.metadata = {
            'arc_state1_state2_usage': 10
        }
        context.resources = {}
        
        score = ArcScorer.score_arc(arc, context, 'state1')
        # Score should be negative due to usage penalty
        assert score < 0
        assert score == pytest.approx(-23.978, rel=0.01)  # -log(11) * 10
    
    def test_score_arc_deterministic_bonus(self):
        """Test arc scoring with deterministic bonus."""
        arc = Mock()
        arc.priority = 0
        arc.target_state = 'state2'
        arc.is_deterministic = True
        del arc.resources
        
        context = Mock()
        context.metadata = {}
        context.resources = {}
        
        score = ArcScorer.score_arc(arc, context, 'state1')
        assert score == 25  # deterministic bonus
    
    def test_score_arc_all_factors(self):
        """Test arc scoring with all factors combined."""
        arc = Mock()
        arc.priority = 2
        arc.target_state = 'state2'
        arc.resources = ['resource1']
        arc.is_deterministic = True
        
        resource1 = Mock()
        resource1.status = 'available'
        
        context = Mock()
        context.metadata = {
            'arc_state1_state2_stats': {'success_rate': 0.6},
            'arc_state1_state2_usage': 5
        }
        context.resources = {'resource1': resource1}
        
        score = ArcScorer.score_arc(arc, context, 'state1')
        # priority(2000) + resources(100) + history(30) + deterministic(25) - usage(~17.9)
        assert score == pytest.approx(2137.08, rel=0.01)
    
    def test_update_arc_usage_new_arc(self):
        """Test updating arc usage for new arc."""
        arc = Mock()
        arc.target_state = 'state2'
        
        context = Mock()
        context.metadata = {}
        
        ArcScorer.update_arc_usage(arc, context, 'state1')
        assert context.metadata['arc_state1_state2_usage'] == 1
    
    def test_update_arc_usage_existing_arc(self):
        """Test updating arc usage for existing arc."""
        arc = Mock()
        arc.target_state = 'state2'
        
        context = Mock()
        context.metadata = {'arc_state1_state2_usage': 5}
        
        ArcScorer.update_arc_usage(arc, context, 'state1')
        assert context.metadata['arc_state1_state2_usage'] == 6


class TestTransitionSelector:
    """Tests for TransitionSelector class."""
    
    def test_select_transition_empty_list(self):
        """Test transition selection with empty list."""
        selector = TransitionSelector()
        context = Mock()
        
        result = selector.select_transition([], context)
        assert result is None
    
    def test_select_transition_single_option(self):
        """Test transition selection with single option."""
        selector = TransitionSelector()
        arc = Mock()
        arc.target_state = 'state2'
        
        context = Mock()
        context.current_state = 'state1'
        context.metadata = {}
        
        result = selector.select_transition([arc], context)
        assert result == arc
        assert 'arc_state1_state2_usage' in context.metadata
    
    def test_select_transition_strategy_based_depth_first(self):
        """Test strategy-based selection with depth-first."""
        selector = TransitionSelector(
            mode=TransitionSelectionMode.STRATEGY_BASED,
            default_strategy=TraversalStrategy.DEPTH_FIRST
        )
        
        arc1 = Mock()
        arc1.target_state = 'state2'
        arc2 = Mock()
        arc2.target_state = 'state3'
        
        context = Mock()
        context.current_state = 'state1'
        context.metadata = {}
        context.state_history = []
        
        result = selector.select_transition([arc1, arc2], context)
        assert result == arc1  # First arc in depth-first
    
    def test_select_transition_strategy_based_breadth_first(self):
        """Test strategy-based selection with breadth-first."""
        selector = TransitionSelector(mode=TransitionSelectionMode.STRATEGY_BASED)
        
        arc1 = Mock()
        arc1.target_state = 'visited_state'
        arc2 = Mock()
        arc2.target_state = 'unvisited_state'
        
        context = Mock()
        context.current_state = 'state1'
        context.metadata = {}
        context.state_history = ['visited_state']
        
        result = selector.select_transition(
            [arc1, arc2], 
            context, 
            strategy=TraversalStrategy.BREADTH_FIRST
        )
        assert result == arc2  # Unvisited state preferred
    
    def test_select_transition_strategy_resource_optimized(self):
        """Test strategy-based selection with resource optimization."""
        selector = TransitionSelector(mode=TransitionSelectionMode.STRATEGY_BASED)
        
        arc1 = Mock()
        arc1.target_state = 'state2'
        arc1.resource_requirements = ['r1', 'r2', 'r3']
        
        arc2 = Mock()
        arc2.target_state = 'state3'
        arc2.resource_requirements = ['r1']
        
        context = Mock()
        context.current_state = 'state1'
        context.metadata = {}
        
        result = selector.select_transition(
            [arc1, arc2],
            context,
            strategy=TraversalStrategy.RESOURCE_OPTIMIZED
        )
        assert result == arc2  # Fewer resources required
    
    def test_select_transition_strategy_stream_optimized(self):
        """Test strategy-based selection with stream optimization."""
        selector = TransitionSelector(mode=TransitionSelectionMode.STRATEGY_BASED)
        
        arc1 = Mock()
        arc1.target_state = 'state2'
        arc1.supports_streaming = False
        
        arc2 = Mock()
        arc2.target_state = 'state3'
        arc2.supports_streaming = True
        
        context = Mock()
        context.current_state = 'state1'
        context.metadata = {}
        
        result = selector.select_transition(
            [arc1, arc2],
            context,
            strategy=TraversalStrategy.STREAM_OPTIMIZED
        )
        assert result == arc2  # Streaming support preferred
    
    def test_select_transition_scoring_based(self):
        """Test scoring-based selection."""
        selector = TransitionSelector(mode=TransitionSelectionMode.PRIORITY_SCORING)
        
        arc1 = Mock()
        arc1.priority = 1
        arc1.target_state = 'state2'
        del arc1.resources
        del arc1.is_deterministic
        
        arc2 = Mock()
        arc2.priority = 5
        arc2.target_state = 'state3'
        del arc2.resources
        del arc2.is_deterministic
        
        context = Mock()
        context.current_state = 'state1'
        context.metadata = {}
        context.resources = {}
        
        result = selector.select_transition([arc1, arc2], context)
        assert result == arc2  # Higher priority wins
    
    def test_select_transition_scoring_round_robin_on_tie(self):
        """Test scoring-based selection with round-robin on tie."""
        # We need to patch ArcScorer.score_arc to disable load balancing
        # so arcs remain tied across multiple calls
        with patch.object(ArcScorer, 'score_arc') as mock_score:
            # Always return same score to ensure ties
            mock_score.return_value = 100.0
            
            selector = TransitionSelector(mode=TransitionSelectionMode.PRIORITY_SCORING)
            
            arc1 = Mock()
            arc1.priority = 1
            arc1.target_state = 'state2'
            
            arc2 = Mock()
            arc2.priority = 1
            arc2.target_state = 'state3'
            
            context = Mock()
            context.current_state = 'state1'
            context.metadata = {}
            context.resources = {}
            
            # Multiple calls should cycle through tied arcs via round-robin
            results = []
            for i in range(4):
                result = selector.select_transition([arc1, arc2], context)
                results.append(result)
            
            # Check round-robin counter was incremented
            assert 'state_state1_round_robin' in context.metadata
            assert context.metadata['state_state1_round_robin'] == 4
            
            # Due to round-robin, we should see both arcs selected
            assert arc1 in results
            assert arc2 in results
            
            # Verify alternating selection pattern
            # With round-robin, we should see arc1, arc2, arc1, arc2
            assert results[0] == arc1
            assert results[1] == arc2
            assert results[2] == arc1
            assert results[3] == arc2
    
    def test_select_transition_hybrid_mode_with_strategy(self):
        """Test hybrid mode when strategy is provided."""
        selector = TransitionSelector(mode=TransitionSelectionMode.HYBRID)
        
        arc1 = Mock()
        arc1.target_state = 'state2'
        arc2 = Mock()
        arc2.target_state = 'state3'
        
        context = Mock()
        context.current_state = 'state1'
        context.metadata = {}
        context.state_history = []
        
        result = selector.select_transition(
            [arc1, arc2],
            context,
            strategy=TraversalStrategy.DEPTH_FIRST
        )
        assert result == arc1  # Uses strategy when provided
    
    def test_select_transition_hybrid_mode_without_strategy(self):
        """Test hybrid mode when no strategy is provided."""
        selector = TransitionSelector(mode=TransitionSelectionMode.HYBRID)
        
        arc1 = Mock()
        arc1.priority = 1
        arc1.target_state = 'state2'
        del arc1.resources
        del arc1.is_deterministic
        
        arc2 = Mock()
        arc2.priority = 5
        arc2.target_state = 'state3'
        del arc2.resources
        del arc2.is_deterministic
        
        context = Mock()
        context.current_state = 'state1'
        context.metadata = {}
        context.resources = {}
        
        result = selector.select_transition([arc1, arc2], context)
        assert result == arc2  # Uses scoring when no strategy


class TestExtractMetricsFromContext:
    """Tests for extract_metrics_from_context function."""
    
    def test_extract_metrics_empty_context(self):
        """Test extracting metrics from empty context."""
        context = Mock()
        context.metadata = {}
        
        metrics = extract_metrics_from_context(context)
        assert metrics == {}
    
    def test_extract_arc_usage_metrics(self):
        """Test extracting arc usage metrics."""
        context = Mock()
        context.metadata = {
            'arc_state1_state2_usage': 5,
            'arc_state2_state3_usage': 3,
            'other_data': 'value'
        }
        
        metrics = extract_metrics_from_context(context)
        assert 'arc_usage' in metrics
        assert metrics['arc_usage'] == {
            'state1_state2': 5,
            'state2_state3': 3
        }
    
    def test_extract_network_warnings(self):
        """Test extracting network selection warnings."""
        context = Mock()
        context.metadata = {
            'network_selection_warning': 'Network not found'
        }
        
        metrics = extract_metrics_from_context(context)
        assert 'warnings' in metrics
        assert metrics['warnings'] == ['Network not found']
    
    def test_extract_batch_info(self):
        """Test extracting batch information."""
        context = Mock()
        context.metadata = {
            'batch_info': {'size': 100, 'processed': 95}
        }
        
        metrics = extract_metrics_from_context(context)
        assert 'batch' in metrics
        assert metrics['batch'] == {'size': 100, 'processed': 95}
    
    def test_extract_timing_metrics(self):
        """Test extracting timing metrics."""
        context = Mock()
        context.metadata = {
            'state1_execution_time': 1.5,
            'resource_acquisition_time': 0.3,
            'other_time': 2.1
        }
        
        metrics = extract_metrics_from_context(context)
        assert 'timing' in metrics
        assert metrics['timing'] == {
            'state1_execution_time': 1.5,
            'resource_acquisition_time': 0.3
        }
    
    def test_extract_all_metrics(self):
        """Test extracting all types of metrics."""
        context = Mock()
        context.metadata = {
            'arc_state1_state2_usage': 10,
            'network_selection_warning': 'Warning message',
            'batch_info': {'size': 50},
            'execution_time_total': 5.5,
            'unrelated_key': 'ignored'
        }
        
        metrics = extract_metrics_from_context(context)
        assert 'arc_usage' in metrics
        assert 'warnings' in metrics
        assert 'batch' in metrics
        assert 'timing' in metrics
        assert len(metrics) == 4