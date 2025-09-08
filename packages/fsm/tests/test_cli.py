"""Tests for FSM CLI commands."""

import pytest
from click.testing import CliRunner
from unittest.mock import Mock, AsyncMock, patch, mock_open, MagicMock
import json
import yaml
from pathlib import Path
import tempfile

from dataknobs_fsm.cli.main import (
    cli, config, run, debug,
    history, pattern
)


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_config():
    """Sample FSM configuration."""
    return {
        'name': 'test_fsm',
        'states': [
            {'name': 'start', 'is_start': True},
            {'name': 'end', 'is_end': True}
        ],
        'arcs': [
            {'from': 'start', 'to': 'end', 'name': 'complete'}
        ]
    }


@pytest.fixture
def temp_config_file(sample_config):
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


class TestCLIMain:
    """Test main CLI command."""
    
    def test_cli_version(self, runner):
        """Test version display."""
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '0.1.0' in result.output
    
    def test_cli_help(self, runner):
        """Test help display."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'FSM CLI' in result.output
        assert 'config' in result.output
        assert 'run' in result.output
        assert 'debug' in result.output


class TestConfigCommands:
    """Test config command group."""
    
    def test_config_create_basic(self, runner):
        """Test creating basic configuration."""
        with patch('builtins.open', mock_open()) as mock_file:
            result = runner.invoke(cli, [
                'config', 'create',
                '--name', 'my_fsm',
                '--output', 'config.yaml'
            ])
            
            assert result.exit_code == 0
            assert 'Configuration created' in result.output
            mock_file.assert_called_once()
    
    def test_config_create_from_template(self, runner):
        """Test creating configuration from template."""
        with patch('builtins.open', mock_open()) as mock_file:
            result = runner.invoke(cli, [
                'config', 'create',
                '--template', 'workflow',
                '--output', 'workflow.yaml'
            ])
            
            assert result.exit_code == 0
            mock_file.assert_called_once()
    
    def test_config_validate_valid(self, runner, temp_config_file):
        """Test validating valid configuration."""
        with patch('dataknobs_fsm.api.simple.SimpleFSM') as mock_fsm:
            mock_instance = Mock()
            mock_instance.validate.return_value = []
            mock_fsm.return_value = mock_instance
            
            result = runner.invoke(cli, [
                'config', 'validate',
                temp_config_file
            ])
            
            assert result.exit_code == 0
            assert 'valid' in result.output.lower()
    
    def test_config_validate_invalid(self, runner, temp_config_file):
        """Test validating invalid configuration."""
        with patch('dataknobs_fsm.api.simple.SimpleFSM') as mock_fsm:
            mock_instance = Mock()
            mock_instance.validate.return_value = ['Missing start state', 'Disconnected state']
            mock_fsm.return_value = mock_instance
            
            result = runner.invoke(cli, [
                'config', 'validate',
                temp_config_file
            ])
            
            assert result.exit_code == 1
            assert 'Missing start state' in result.output
    
    def test_config_visualize(self, runner, temp_config_file):
        """Test configuration visualization."""
        with patch('dataknobs_fsm.api.simple.SimpleFSM') as mock_fsm:
            mock_instance = Mock()
            mock_instance.get_visualization.return_value = {
                'states': [
                    {'id': 'start', 'label': 'Start', 'is_start': True},
                    {'id': 'end', 'label': 'End', 'is_end': True}
                ],
                'transitions': [
                    {'from': 'start', 'to': 'end', 'label': 'complete'}
                ]
            }
            mock_fsm.return_value = mock_instance
            
            result = runner.invoke(cli, [
                'config', 'visualize',
                temp_config_file
            ])
            
            assert result.exit_code == 0
            assert 'start' in result.output.lower()
            assert 'end' in result.output.lower()


class TestRunCommands:
    """Test run command group."""
    
    @pytest.mark.asyncio
    def test_run_execute(self, runner, temp_config_file):
        """Test executing FSM."""
        with patch('dataknobs_fsm.api.simple.SimpleFSM') as mock_fsm:
            mock_instance = Mock()
            mock_instance.execute = AsyncMock(return_value={
                'success': True,
                'final_state': 'end',
                'execution_time': 0.5,
                'data': {'result': 'completed'}
            })
            mock_fsm.return_value = mock_instance
            
            result = runner.invoke(cli, [
                'run', 'execute',
                temp_config_file,
                '--data', '{"input": "test"}'
            ])
            
            assert result.exit_code == 0
            assert 'Success' in result.output
            assert 'end' in result.output
    
    @pytest.mark.asyncio
    def test_run_batch(self, runner, temp_config_file):
        """Test batch execution."""
        batch_data = [{'id': i} for i in range(3)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(batch_data, f)
            batch_file = f.name
        
        try:
            with patch('dataknobs_fsm.api.simple.SimpleFSM') as mock_fsm:
                mock_instance = Mock()
                mock_instance.execute_batch = AsyncMock(return_value=[
                    {'success': True, 'final_state': 'end', 'data': {'id': i}}
                    for i in range(3)
                ])
                mock_fsm.return_value = mock_instance
                
                result = runner.invoke(cli, [
                    'run', 'batch',
                    temp_config_file,
                    '--data-file', batch_file,
                    '--batch-size', '2'
                ])
                
                assert result.exit_code == 0
                assert 'Processed: 3' in result.output
                assert 'Successful: 3' in result.output
        finally:
            Path(batch_file).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    def test_run_stream(self, runner, temp_config_file):
        """Test stream execution."""
        with patch('dataknobs_fsm.api.simple.SimpleFSM') as mock_fsm:
            mock_instance = Mock()
            
            async def mock_stream(data_gen):
                for i in range(3):
                    yield {'success': True, 'data': {'id': i}}
            
            mock_instance.execute_stream = mock_stream
            mock_fsm.return_value = mock_instance
            
            # Create stream data file
            stream_data = [{'id': i} for i in range(3)]
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for item in stream_data:
                    f.write(json.dumps(item) + '\n')
                stream_file = f.name
            
            try:
                result = runner.invoke(cli, [
                    'run', 'stream',
                    temp_config_file,
                    '--source', stream_file
                ])
                
                assert result.exit_code == 0
                assert 'Stream processing started' in result.output
            finally:
                Path(stream_file).unlink(missing_ok=True)


class TestDebugCommands:
    """Test debug command group."""
    
    @pytest.mark.asyncio
    def test_debug_trace(self, runner, temp_config_file):
        """Test trace execution."""
        with patch('dataknobs_fsm.api.advanced.AdvancedFSM') as mock_fsm:
            mock_instance = Mock()
            mock_instance.execute_workflow = AsyncMock(return_value={
                'success': True,
                'final_state': 'end'
            })
            mock_instance.get_trace = Mock(return_value=[
                {'timestamp': '2024-01-01T00:00:00', 'state': 'start', 'data': {}},
                {'timestamp': '2024-01-01T00:00:01', 'state': 'end', 'data': {}}
            ])
            mock_fsm.return_value = mock_instance
            
            result = runner.invoke(cli, [
                'debug', 'trace',
                temp_config_file,
                '--data', '{}'
            ])
            
            assert result.exit_code == 0
            assert 'start' in result.output
            assert 'end' in result.output
    
    def test_debug_breakpoint(self, runner, temp_config_file):
        """Test breakpoint debugging."""
        with patch('dataknobs_fsm.api.advanced.AdvancedFSM') as mock_fsm:
            mock_instance = Mock()
            mock_instance.set_breakpoint = Mock()
            mock_instance.step = AsyncMock(return_value={
                'state': 'start',
                'data': {},
                'can_continue': True
            })
            mock_fsm.return_value = mock_instance
            
            # Simulate user input for interactive debugging
            with patch('click.confirm', return_value=False):
                result = runner.invoke(cli, [
                    'debug', 'breakpoint',
                    temp_config_file,
                    '--state', 'start',
                    '--data', '{}'
                ])
                
                assert result.exit_code == 0
                mock_instance.set_breakpoint.assert_called_with('start')
    
    def test_debug_profile(self, runner, temp_config_file):
        """Test profiling execution."""
        with patch('dataknobs_fsm.api.advanced.AdvancedFSM') as mock_fsm:
            mock_instance = Mock()
            mock_instance.execute_workflow = AsyncMock(return_value={
                'success': True,
                'final_state': 'end'
            })
            mock_instance.get_profile = Mock(return_value={
                'states': {
                    'start': {'count': 1, 'total_time': 0.1, 'avg_time': 0.1},
                    'end': {'count': 1, 'total_time': 0.05, 'avg_time': 0.05}
                },
                'total_time': 0.15
            })
            mock_fsm.return_value = mock_instance
            
            result = runner.invoke(cli, [
                'debug', 'profile',
                temp_config_file,
                '--data', '{}',
                '--iterations', '1'
            ])
            
            assert result.exit_code == 0
            assert 'start' in result.output
            assert '0.1' in result.output


class TestHistoryCommands:
    """Test history command group."""
    
    def test_history_list(self, runner):
        """Test listing execution history."""
        with patch('dataknobs_fsm.api.advanced.AdvancedFSM') as mock_fsm:
            mock_instance = Mock()
            mock_instance.get_history = Mock(return_value=[
                {
                    'execution_id': 'exec1',
                    'timestamp': '2024-01-01T00:00:00',
                    'success': True,
                    'final_state': 'end',
                    'duration': 0.5
                },
                {
                    'execution_id': 'exec2',
                    'timestamp': '2024-01-01T00:01:00',
                    'success': False,
                    'final_state': 'error',
                    'duration': 0.3
                }
            ])
            mock_fsm.return_value = mock_instance
            
            result = runner.invoke(cli, [
                'history', 'list',
                '--limit', '10'
            ])
            
            assert result.exit_code == 0
            assert 'exec1' in result.output
            assert 'exec2' in result.output
    
    def test_history_show(self, runner):
        """Test showing execution details."""
        with patch('dataknobs_fsm.api.advanced.AdvancedFSM') as mock_fsm:
            mock_instance = Mock()
            mock_instance.get_execution = Mock(return_value={
                'execution_id': 'exec1',
                'timestamp': '2024-01-01T00:00:00',
                'success': True,
                'final_state': 'end',
                'duration': 0.5,
                'state_history': ['start', 'middle', 'end'],
                'data': {'input': 'test', 'output': 'result'}
            })
            mock_fsm.return_value = mock_instance
            
            result = runner.invoke(cli, [
                'history', 'show',
                'exec1'
            ])
            
            assert result.exit_code == 0
            assert 'exec1' in result.output
            assert 'end' in result.output
    
    def test_history_clear(self, runner):
        """Test clearing execution history."""
        with patch('dataknobs_fsm.api.advanced.AdvancedFSM') as mock_fsm:
            mock_instance = Mock()
            mock_instance.clear_history = Mock()
            mock_fsm.return_value = mock_instance
            
            with patch('click.confirm', return_value=True):
                result = runner.invoke(cli, [
                    'history', 'clear',
                    '--before', '2024-01-01'
                ])
                
                assert result.exit_code == 0
                assert 'cleared' in result.output.lower()


class TestPatternCommands:
    """Test pattern command group."""
    
    def test_pattern_etl(self, runner):
        """Test ETL pattern execution."""
        with patch('dataknobs_fsm.patterns.etl.create_etl_pipeline') as mock_create:
            mock_etl = Mock()
            mock_etl.run = AsyncMock(return_value={
                'extracted': 100,
                'transformed': 95,
                'loaded': 95,
                'errors': 5
            })
            mock_create.return_value = mock_etl
            
            result = runner.invoke(cli, [
                'pattern', 'etl',
                '--source', 'postgresql://localhost/source',
                '--target', 'postgresql://localhost/target',
                '--mode', 'full'
            ])
            
            assert result.exit_code == 0
            assert 'Extracted: 100' in result.output
            assert 'Loaded: 95' in result.output
    
    def test_pattern_file_process(self, runner):
        """Test file processing pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / 'input'
            output_dir = Path(tmpdir) / 'output'
            input_dir.mkdir()
            output_dir.mkdir()
            
            # Create test file
            test_file = input_dir / 'test.txt'
            test_file.write_text('test content')
            
            with patch('dataknobs_fsm.patterns.file_processing.create_file_processor') as mock_create:
                mock_processor = Mock()
                mock_processor.process = AsyncMock(return_value=[
                    {'success': True, 'file': str(test_file)}
                ])
                mock_processor.get_metrics = Mock(return_value={
                    'files_processed': 1,
                    'files_failed': 0,
                    'total_files': 1
                })
                mock_create.return_value = mock_processor
                
                result = runner.invoke(cli, [
                    'pattern', 'file-process',
                    '--input', str(input_dir),
                    '--output', str(output_dir),
                    '--pattern', '*.txt',
                    '--mode', 'batch'
                ])
                
                assert result.exit_code == 0
                assert 'Processed: 1' in result.output


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def test_invalid_config_file(self, runner):
        """Test handling of invalid config file."""
        result = runner.invoke(cli, [
            'config', 'validate',
            'nonexistent.yaml'
        ])
        
        assert result.exit_code != 0
        assert 'Error' in result.output or 'not found' in result.output.lower()
    
    def test_invalid_json_data(self, runner, temp_config_file):
        """Test handling of invalid JSON data."""
        result = runner.invoke(cli, [
            'run', 'execute',
            temp_config_file,
            '--data', 'invalid json'
        ])
        
        assert result.exit_code != 0
        assert 'Invalid JSON' in result.output or 'Error' in result.output
    
    def test_missing_required_args(self, runner):
        """Test handling of missing required arguments."""
        result = runner.invoke(cli, ['config', 'create'])
        
        assert result.exit_code != 0
        assert 'Missing' in result.output or 'required' in result.output.lower()


class TestCLIIntegration:
    """Integration tests for CLI."""
    
    def test_full_workflow(self, runner):
        """Test complete workflow from config creation to execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            
            # Create config
            with patch('builtins.open', mock_open()) as mock_file:
                result = runner.invoke(cli, [
                    'config', 'create',
                    '--name', 'test_workflow',
                    '--template', 'simple',
                    '--output', str(config_path)
                ])
                assert result.exit_code == 0
            
            # Validate config
            with patch('dataknobs_fsm.api.simple.SimpleFSM') as mock_fsm:
                mock_instance = Mock()
                mock_instance.validate.return_value = []
                mock_fsm.return_value = mock_instance
                
                result = runner.invoke(cli, [
                    'config', 'validate',
                    str(config_path)
                ])
                assert result.exit_code == 0
            
            # Execute FSM
            with patch('dataknobs_fsm.api.simple.SimpleFSM') as mock_fsm:
                mock_instance = Mock()
                mock_instance.execute = AsyncMock(return_value={
                    'success': True,
                    'final_state': 'complete'
                })
                mock_fsm.return_value = mock_instance
                
                result = runner.invoke(cli, [
                    'run', 'execute',
                    str(config_path),
                    '--data', '{}'
                ])
                assert result.exit_code == 0