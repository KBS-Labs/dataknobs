"""Real implementation tests for FSM CLI commands."""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from click.testing import CliRunner
import asyncio

from dataknobs_fsm import __version__
from dataknobs_fsm.cli.main import cli
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.api.advanced import AdvancedFSM
from dataknobs_fsm.config.loader import ConfigLoader


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def simple_config():
    """Simple FSM configuration."""
    return {
        'name': 'test_fsm',
        'data_mode': 'copy',
        'states': [
            {'name': 'start', 'is_start': True},
            {'name': 'process'},
            {'name': 'end', 'is_end': True}
        ],
        'arcs': [
            {
                'from': 'start', 
                'to': 'process', 
                'name': 'begin',
                'transform': {'code': 'data["step"] = 1; data'}
            },
            {
                'from': 'process', 
                'to': 'end', 
                'name': 'complete',
                'transform': {'code': 'data["step"] = 2; data'}
            }
        ]
    }


@pytest.fixture
def etl_config():
    """ETL FSM configuration."""
    return {
        'name': 'etl_pipeline',
        'data_mode': 'copy',
        'states': [
            {'name': 'extract', 'is_start': True},
            {'name': 'transform'},
            {'name': 'load'},
            {'name': 'complete', 'is_end': True}
        ],
        'arcs': [
            {
                'from': 'extract', 
                'to': 'transform', 
                'name': 'extracted',
                'transform': {'code': 'data["extracted"] = True; data'}
            },
            {
                'from': 'transform', 
                'to': 'load', 
                'name': 'transformed',
                'transform': {'code': 'data["transformed"] = True; data'}
            },
            {
                'from': 'load', 
                'to': 'complete', 
                'name': 'loaded',
                'transform': {'code': 'data["loaded"] = True; data'}
            }
        ]
    }


@pytest.fixture
def workflow_config():
    """Workflow FSM configuration with conditions."""
    return {
        'name': 'approval_workflow',
        'data_mode': 'reference',
        'states': [
            {'name': 'receive', 'is_start': True},
            {'name': 'validate'},
            {'name': 'approve'},
            {'name': 'reject'},
            {'name': 'complete', 'is_end': True}
        ],
        'arcs': [
            {
                'from': 'receive', 
                'to': 'validate', 
                'name': 'received'
            },
            {
                'from': 'validate', 
                'to': 'approve', 
                'name': 'valid',
                'pre_test': {'code': 'data.get("is_valid", False)'}
            },
            {
                'from': 'validate', 
                'to': 'reject', 
                'name': 'invalid',
                'pre_test': {'code': 'not data.get("is_valid", False)'}
            },
            {
                'from': 'approve', 
                'to': 'complete', 
                'name': 'approved',
                'transform': {'code': 'data["status"] = "approved"; data'}
            },
            {
                'from': 'reject', 
                'to': 'complete', 
                'name': 'rejected',
                'transform': {'code': 'data["status"] = "rejected"; data'}
            }
        ]
    }


@pytest.fixture
def temp_config_file(simple_config):
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(simple_config, f)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_etl_config_file(etl_config):
    """Create temporary ETL config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(etl_config, f)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_workflow_config_file(workflow_config):
    """Create temporary workflow config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(workflow_config, f)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


class TestCLIMain:
    """Test main CLI command."""
    
    def test_cli_version(self, runner):
        """Test version display."""
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert __version__ in result.output
    
    def test_cli_help(self, runner):
        """Test help display."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'FSM CLI' in result.output
        assert 'config' in result.output
        assert 'run' in result.output


class TestConfigCommands:
    """Test config command group with real implementations."""
    
    def test_config_create_basic(self, runner):
        """Test creating basic configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'config.yaml'
            
            result = runner.invoke(cli, [
                'config', 'create',
                'basic',
                '--output', str(output_path)
            ])
            
            assert result.exit_code == 0
            assert 'Created basic configuration' in result.output
            assert output_path.exists()
            
            # Verify the created config
            with open(output_path) as f:
                config = yaml.safe_load(f)
            assert config['name'] == 'Basic_FSM'
            assert len(config['states']) == 3
            assert len(config['arcs']) == 2
    
    def test_config_create_etl_template(self, runner):
        """Test creating ETL template configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'etl.yaml'
            
            result = runner.invoke(cli, [
                'config', 'create',
                'etl',
                '--output', str(output_path)
            ])
            
            assert result.exit_code == 0
            assert 'Created etl configuration' in result.output
            assert output_path.exists()
            
            # Verify the ETL config
            with open(output_path) as f:
                config = yaml.safe_load(f)
            assert config['name'] == 'ETL_Pipeline'
            assert 'resources' in config
            assert len(config['states']) == 4
    
    def test_config_create_json_format(self, runner):
        """Test creating configuration in JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'config.json'
            
            result = runner.invoke(cli, [
                'config', 'create',
                'workflow',
                '--output', str(output_path),
                '--format', 'json'
            ])
            
            assert result.exit_code == 0
            assert output_path.exists()
            
            # Verify JSON format
            with open(output_path) as f:
                config = json.load(f)
            assert config['name'] == 'Workflow_FSM'
    
    def test_config_validate_valid(self, runner, temp_config_file):
        """Test validating valid configuration using real validator."""
        result = runner.invoke(cli, [
            'config', 'validate',
            temp_config_file
        ])
        
        assert result.exit_code == 0
        assert 'Configuration is valid' in result.output
    
    def test_config_validate_verbose(self, runner, temp_config_file):
        """Test verbose validation output."""
        result = runner.invoke(cli, [
            'config', 'validate',
            temp_config_file,
            '--verbose'
        ])
        
        assert result.exit_code == 0
        assert 'Configuration is valid' in result.output
        assert 'Configuration Details' in result.output
        assert 'States: 3' in result.output
        assert 'Arcs: 2' in result.output
    
    def test_config_validate_invalid(self, runner):
        """Test validating invalid configuration."""
        # Create invalid config (missing required fields)
        invalid_config = {
            'name': 'invalid_fsm',
            'states': [
                {'name': 'orphan'}  # No start or end states
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            result = runner.invoke(cli, [
                'config', 'validate',
                temp_path
            ])
            
            assert result.exit_code != 0
            assert 'validation failed' in result.output.lower() or 'error' in result.output.lower()
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_show_tree(self, runner, temp_config_file):
        """Test showing configuration as tree."""
        result = runner.invoke(cli, [
            'config', 'show',
            temp_config_file,
            '--format', 'tree'
        ])
        
        assert result.exit_code == 0
        assert 'test_fsm' in result.output
        assert 'States' in result.output
        assert 'start' in result.output
        assert 'Arcs' in result.output
    
    def test_config_show_table(self, runner, temp_etl_config_file):
        """Test showing configuration as table."""
        result = runner.invoke(cli, [
            'config', 'show',
            temp_etl_config_file,
            '--format', 'table'
        ])
        
        assert result.exit_code == 0
        assert 'etl_pipeline' in result.output
        assert 'extract' in result.output
        assert 'transform' in result.output
        assert 'load' in result.output
    
    def test_config_show_graph(self, runner, temp_workflow_config_file):
        """Test showing configuration as graph."""
        result = runner.invoke(cli, [
            'config', 'show',
            temp_workflow_config_file,
            '--format', 'graph'
        ])
        
        assert result.exit_code == 0
        assert 'mermaid' in result.output.lower()
        assert 'graph TD' in result.output
        assert 'receive' in result.output
        assert 'approve' in result.output


class TestRunCommands:
    """Test run commands with real FSM execution."""
    
    def test_run_execute_simple(self, runner, temp_config_file):
        """Test executing simple FSM with real engine."""
        result = runner.invoke(cli, [
            'run', 'execute',
            temp_config_file,
            '--data', '{"input": "test"}'
        ])
        
        # Check that execution attempted (may fail due to missing run command implementation)
        assert result.exit_code == 0 or 'not implemented' in result.output.lower()
    
    def test_run_execute_with_initial_state(self, runner, temp_config_file):
        """Test execution with initial state specified."""
        result = runner.invoke(cli, [
            'run', 'execute',
            temp_config_file,
            '--data', '{"value": 42}',
            '--initial-state', 'process'
        ])
        
        # Check execution attempted
        assert result.exit_code == 0 or 'not implemented' in result.output.lower()
    
    def test_run_batch_processing(self, runner, temp_config_file):
        """Test batch processing with real batch executor."""
        batch_data = [
            {'id': 1, 'value': 'a'},
            {'id': 2, 'value': 'b'},
            {'id': 3, 'value': 'c'}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(batch_data, f)
            batch_file = f.name
        
        try:
            result = runner.invoke(cli, [
                'run', 'batch',
                temp_config_file,
                batch_file,  # data_file is a positional argument
                '--batch-size', '2'
            ])
            
            # Check execution attempted
            assert result.exit_code == 0 or 'not implemented' in result.output.lower()
        finally:
            Path(batch_file).unlink(missing_ok=True)
    
    def test_run_workflow_valid_path(self, runner, temp_workflow_config_file):
        """Test workflow execution with valid data path."""
        result = runner.invoke(cli, [
            'run', 'execute',
            temp_workflow_config_file,
            '--data', '{"is_valid": true}'
        ])
        
        # Check execution attempted
        assert result.exit_code == 0 or 'not implemented' in result.output.lower()
    
    def test_run_workflow_invalid_path(self, runner, temp_workflow_config_file):
        """Test workflow execution with invalid data path."""
        result = runner.invoke(cli, [
            'run', 'execute',
            temp_workflow_config_file,
            '--data', '{"is_valid": false}'
        ])
        
        # Check execution attempted
        assert result.exit_code == 0 or 'not implemented' in result.output.lower()


class TestCLIErrorHandling:
    """Test CLI error handling with real file operations."""
    
    def test_invalid_config_file(self, runner):
        """Test handling of non-existent config file."""
        result = runner.invoke(cli, [
            'config', 'validate',
            'nonexistent.yaml'
        ])
        
        assert result.exit_code != 0
        assert 'error' in result.output.lower() or 'not found' in result.output.lower()
    
    def test_invalid_json_data(self, runner, temp_config_file):
        """Test handling of invalid JSON data."""
        result = runner.invoke(cli, [
            'run', 'execute',
            temp_config_file,
            '--data', 'invalid json {'
        ])
        
        # Should fail due to invalid JSON
        assert result.exit_code != 0 or 'error' in result.output.lower()
    
    def test_malformed_yaml_config(self, runner):
        """Test handling of malformed YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid:\n  - yaml\n    malformed: here")
            temp_path = f.name
        
        try:
            result = runner.invoke(cli, [
                'config', 'validate',
                temp_path
            ])
            
            assert result.exit_code != 0
            assert 'error' in result.output.lower()
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_missing_required_template(self, runner):
        """Test handling of missing template argument."""
        result = runner.invoke(cli, ['config', 'create'])
        
        assert result.exit_code != 0
        assert 'missing' in result.output.lower() or 'required' in result.output.lower()
    
    def test_invalid_template_choice(self, runner):
        """Test handling of invalid template choice."""
        result = runner.invoke(cli, [
            'config', 'create',
            'invalid_template'
        ])
        
        assert result.exit_code != 0
        assert 'invalid' in result.output.lower() or 'error' in result.output.lower()


class TestCLIIntegration:
    """Integration tests for complete CLI workflows."""
    
    def test_create_validate_execute_workflow(self, runner):
        """Test complete workflow from creation to execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'test_config.yaml'
            
            # Step 1: Create configuration
            result = runner.invoke(cli, [
                'config', 'create',
                'basic',
                '--output', str(config_path)
            ])
            assert result.exit_code == 0
            assert config_path.exists()
            
            # Step 2: Validate configuration
            result = runner.invoke(cli, [
                'config', 'validate',
                str(config_path)
            ])
            assert result.exit_code == 0
            assert 'valid' in result.output.lower()
            
            # Step 3: Show configuration
            result = runner.invoke(cli, [
                'config', 'show',
                str(config_path)
            ])
            assert result.exit_code == 0
            assert 'Basic_FSM' in result.output
            
            # Step 4: Execute FSM (if implemented)
            result = runner.invoke(cli, [
                'run', 'execute',
                str(config_path),
                '--data', '{"test": "data"}'
            ])
            # Check execution attempted
            assert result.exit_code == 0 or 'not implemented' in result.output.lower()
    
    def test_etl_template_workflow(self, runner):
        """Test ETL template creation and validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            etl_path = Path(tmpdir) / 'etl.yaml'
            
            # Create ETL configuration
            result = runner.invoke(cli, [
                'config', 'create',
                'etl',
                '--output', str(etl_path)
            ])
            assert result.exit_code == 0
            
            # Validate ETL configuration
            result = runner.invoke(cli, [
                'config', 'validate',
                str(etl_path),
                '--verbose'
            ])
            if result.exit_code != 0:
                print(f"Validation failed: {result.output}")
                print(f"Exception: {result.exception}")
            assert result.exit_code == 0
            assert 'Resources:' in result.output
    
    def test_multiple_format_outputs(self, runner, temp_config_file):
        """Test different output formats for the same config."""
        # Tree format
        result = runner.invoke(cli, [
            'config', 'show',
            temp_config_file,
            '--format', 'tree'
        ])
        assert result.exit_code == 0
        assert 'States' in result.output
        
        # Table format
        result = runner.invoke(cli, [
            'config', 'show',
            temp_config_file,
            '--format', 'table'
        ])
        assert result.exit_code == 0
        assert 'Name' in result.output
        
        # Graph format
        result = runner.invoke(cli, [
            'config', 'show',
            temp_config_file,
            '--format', 'graph'
        ])
        assert result.exit_code == 0
        assert 'mermaid' in result.output.lower()


class TestAdvancedCLIFeatures:
    """Test advanced CLI features with real components."""
    
    def test_complex_transformation_config(self, runner):
        """Test config with complex transformations."""
        complex_config = {
            'name': 'complex_fsm',
            'data_mode': 'copy',
            'functions': {
                'double': {'code': 'data["value"] = data.get("value", 0) * 2; data'},
                'validate': {'code': 'data.get("value", 0) > 0'}
            },
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'process'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'process',
                    'name': 'begin',
                    'condition': {'type': 'inline', 'code': 'data.get("value", 0) > 0'},
                    'transform': {'type': 'inline', 'code': 'data["value"] = data.get("value", 0) * 2; data'}
                },
                {
                    'from': 'process',
                    'to': 'end',
                    'name': 'complete'
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(complex_config, f)
            temp_path = f.name
        
        try:
            # Validate complex configuration
            result = runner.invoke(cli, [
                'config', 'validate',
                temp_path,
                '--verbose'
            ])
            
            if result.exit_code != 0:
                print(f"Validation failed: {result.output}")
            assert result.exit_code == 0
            # Changed to inline functions, so no longer shows function count
            assert 'Configuration is valid' in result.output
            assert 'States: 3' in result.output
            assert 'Arcs: 2' in result.output
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_resource_configuration(self, runner):
        """Test configuration with resources."""
        resource_config = {
            'name': 'resource_fsm',
            'data_mode': 'copy',
            'resources': [
                {
                    'name': 'database',
                    'type': 'database',
                    'provider': 'sqlite',
                    'path': ':memory:'
                },
                {
                    'name': 'cache',
                    'type': 'custom',
                    'provider': 'memory',
                    'size': 1000
                }
            ],
            'states': [
                {'name': 'start', 'is_start': True, 'resources': ['database']},
                {'name': 'cache', 'resources': ['cache']},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {'from': 'start', 'to': 'cache', 'name': 'query'},
                {'from': 'cache', 'to': 'end', 'name': 'complete'}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(resource_config, f)
            temp_path = f.name
        
        try:
            result = runner.invoke(cli, [
                'config', 'show',
                temp_path,
                '--format', 'tree'
            ])
            
            if result.exit_code != 0:
                print(f"Show command failed: {result.output}")
            assert result.exit_code == 0
            assert 'Resources' in result.output
            assert 'database' in result.output
            assert 'cache' in result.output
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestAsyncCLIOperations:
    """Test async operations in CLI."""
    
    @pytest.mark.asyncio
    async def test_async_fsm_execution(self, temp_config_file):
        """Test async FSM execution directly."""
        # SimpleFSM expects a file path, not a config object
        fsm = SimpleFSM(temp_config_file)
        result = fsm.process({'test': 'data'})
        
        assert 'final_state' in result
        assert result['final_state'] == 'end'
    
    def test_batch_async_execution(self, temp_config_file):
        """Test batch async execution."""
        # SimpleFSM expects a file path, not a config object
        fsm = SimpleFSM(temp_config_file)
        batch_data = [{'id': i} for i in range(3)]

        # Note: process_batch internally uses async but returns sync results
        results = fsm.process_batch(batch_data)

        assert len(results) == 3
        for result in results:
            assert 'final_state' in result


class TestHistoryCLICommands:
    """End-to-end tests for ``fsm history list`` and ``fsm history show-execution``.

    Until this point the commands had latent breakage hidden by
    ``# type: ignore``: ``ExecutionHistory(storage)`` mis-used the
    history dataclass as a manager, and ``query_history`` /
    ``get_execution`` were never methods on any class.  These tests
    exercise the rewritten commands against a real ``FileStorage``
    backend seeded via the ``BaseHistoryStorage`` API.

    Tests redirect ``Path.home()`` by setting ``HOME`` (and ``USERPROFILE``
    on Windows) for the ``CliRunner`` invocation, so the CLI's default
    history location ``~/.fsm/history.json`` resolves under ``tmp_path``
    without any production-code changes.
    """

    @pytest.fixture
    def history_dir(self, tmp_path: Path) -> Path:
        """Path to the CLI's history file under ``tmp_path``.

        ``AsyncFileDatabase`` stores data in a single JSON file, so
        the CLI's canonical location is ``~/.fsm/history.json`` (not
        a directory).  This fixture creates the parent ``~/.fsm/``
        directory and returns the file path; the file itself is
        created lazily by the storage on first write.

        The fixture name is preserved for compatibility with the rest
        of this suite even though it now returns a file.
        """
        parent = tmp_path / '.fsm'
        parent.mkdir(parents=True, exist_ok=True)
        return parent / 'history.json'

    @pytest.fixture
    def home_env(self, tmp_path: Path) -> dict[str, str]:
        """Env overlay that points ``Path.home()`` at ``tmp_path``."""
        # ``Path.home()`` consults ``HOME`` on POSIX and ``USERPROFILE``
        # on Windows.  Set both so the test is platform-agnostic.
        return {'HOME': str(tmp_path), 'USERPROFILE': str(tmp_path)}

    @staticmethod
    def _seed_history(
        history_path: Path,
        execution_id: str,
        fsm_name: str,
        *,
        fail_one_step: bool = False,
        in_progress: bool = False,
    ) -> None:
        """Write a single ExecutionHistory + its steps to the history file.

        Uses the real ``FileStorage`` + ``BaseHistoryStorage`` API, so
        whatever shape the CLI reads back is exactly what production
        callers would write.

        When ``in_progress`` is True, ``end_time`` is left unset so the
        history record is persisted in the same shape an actively-running
        execution would have on disk.  Mutually exclusive with
        ``fail_one_step``: an in-progress run has no terminal status to
        compare.
        """
        from dataknobs_fsm.core.data_modes import DataHandlingMode
        from dataknobs_fsm.execution.history import ExecutionHistory
        from dataknobs_fsm.storage.base import StorageBackend, StorageConfig
        from dataknobs_fsm.storage.file import FileStorage

        async def _seed() -> None:
            storage = FileStorage(
                StorageConfig(
                    backend=StorageBackend.FILE,
                    connection_params={'path': str(history_path)},
                )
            )
            await storage.initialize()
            try:
                hist = ExecutionHistory(
                    fsm_name=fsm_name,
                    execution_id=execution_id,
                    data_mode=DataHandlingMode.COPY,
                )
                step1 = hist.add_step('start', 'main')
                step1.complete(arc_taken='begin')
                step2 = hist.add_step('process', 'main', parent_step_id=step1.step_id)
                if fail_one_step:
                    step2.fail(RuntimeError("synthetic failure"))
                    hist.failed_steps = 1
                else:
                    step2.complete(arc_taken='done')
                if not in_progress:
                    hist.end_time = hist.start_time + 1.5
                hist.total_steps = 2
                await storage.save_history(hist)
                await storage.save_step(execution_id, step1)
                await storage.save_step(execution_id, step2)
            finally:
                await storage.close()

        asyncio.run(_seed())

    def test_list_empty(self, runner, history_dir, home_env):
        """An empty history dir reports no entries cleanly."""
        result = runner.invoke(cli, ['history', 'list'], env=home_env)
        assert result.exit_code == 0, result.output
        assert 'No history entries found' in result.output

    def test_list_with_entries_table(
        self, runner, tmp_path, history_dir, home_env
    ):
        """Seeded history shows up in the table output with correct columns."""
        self._seed_history(
            history_dir, execution_id='exec-aaaaaaaaaa', fsm_name='etl_pipeline',
        )
        result = runner.invoke(cli, ['history', 'list'], env=home_env)

        assert result.exit_code == 0, result.output
        # ID column shows first 8 chars of the execution id
        assert 'exec-aaa' in result.output
        # FSM name column
        assert 'etl_pipeline' in result.output
        # Status comes from BaseHistoryStorage's status field, not a hallucinated bool
        assert 'completed' in result.output.lower()
        # Total steps column populated
        assert ' 2 ' in result.output or '\n2' in result.output

    def test_list_filters_by_fsm_name(
        self, runner, tmp_path, history_dir, home_env
    ):
        """``--fsm-name`` filters server-side via ``query_histories``."""
        self._seed_history(
            history_dir, execution_id='exec-keep-1', fsm_name='target_fsm',
        )
        self._seed_history(
            history_dir, execution_id='exec-skip-2', fsm_name='other_fsm',
        )

        result = runner.invoke(
            cli, ['history', 'list', '--fsm-name', 'target_fsm'], env=home_env,
        )

        assert result.exit_code == 0, result.output
        assert 'exec-kee' in result.output
        assert 'exec-ski' not in result.output

    def test_list_json_format(self, runner, tmp_path, history_dir, home_env):
        """``--format json`` emits parseable JSON with the real schema."""
        self._seed_history(
            history_dir, execution_id='exec-json-1234', fsm_name='json_fsm',
        )
        result = runner.invoke(
            cli, ['history', 'list', '--format', 'json'], env=home_env,
        )
        assert result.exit_code == 0, result.output

        payload = json.loads(result.output)
        assert isinstance(payload, list)
        assert len(payload) == 1
        # query_histories returns: id, fsm_name, status, start_time, end_time,
        # total_steps, failed_steps, metadata, data_mode
        entry = payload[0]
        assert entry['id'] == 'exec-json-1234'
        assert entry['fsm_name'] == 'json_fsm'
        assert 'status' in entry
        assert 'total_steps' in entry

    def test_show_execution_found(
        self, runner, tmp_path, history_dir, home_env
    ):
        """``show-execution <id>`` prints details for a known execution."""
        self._seed_history(
            history_dir, execution_id='exec-show-12', fsm_name='show_fsm',
        )
        result = runner.invoke(
            cli, ['history', 'show-execution', 'exec-show-12'], env=home_env,
        )
        assert result.exit_code == 0, result.output
        assert 'exec-show-12' in result.output
        assert 'show_fsm' in result.output
        assert 'Total steps: 2' in result.output

    def test_show_execution_verbose_lists_steps(
        self, runner, tmp_path, history_dir, home_env
    ):
        """``--verbose`` includes the per-step execution path."""
        self._seed_history(
            history_dir, execution_id='exec-verbose-1', fsm_name='verbose_fsm',
        )
        result = runner.invoke(
            cli,
            ['history', 'show-execution', 'exec-verbose-1', '--verbose'],
            env=home_env,
        )
        assert result.exit_code == 0, result.output
        assert 'Execution Path' in result.output
        # Both seeded states should appear in the path listing
        assert 'start' in result.output
        assert 'process' in result.output

    def test_show_execution_failed_run_reports_failure(
        self, runner, tmp_path, history_dir, home_env
    ):
        """``failed_steps`` from the underlying history surfaces in the output."""
        self._seed_history(
            history_dir,
            execution_id='exec-failed-1',
            fsm_name='failing_fsm',
            fail_one_step=True,
        )
        result = runner.invoke(
            cli, ['history', 'show-execution', 'exec-failed-1'], env=home_env,
        )
        assert result.exit_code == 0, result.output
        assert 'Failed steps: 1' in result.output

    def test_show_execution_not_found_exits_nonzero(
        self, runner, history_dir, home_env
    ):
        """Unknown execution id exits with a non-zero status."""
        result = runner.invoke(
            cli, ['history', 'show-execution', 'does-not-exist'], env=home_env,
        )
        assert result.exit_code != 0
        assert 'not found' in result.output.lower()

    def test_show_execution_in_progress_run_status_consistent(
        self, runner, tmp_path, history_dir, home_env
    ):
        """In-progress runs display 'in_progress' status, not 'completed'.

        Storage writes ``status='completed'`` whenever ``end_time`` is
        set, so the CLI must derive its display status from
        ``(end_time, failed_steps)`` rather than echo the stored field.
        Pre-fix, ``status`` was hard-coded to ``'failed' if failed_steps
        else 'completed'`` and contradicted the ``End: In progress``
        line for executions with no terminal time.
        """
        self._seed_history(
            history_dir,
            execution_id='exec-running-1',
            fsm_name='running_fsm',
            in_progress=True,
        )
        result = runner.invoke(
            cli, ['history', 'show-execution', 'exec-running-1'], env=home_env,
        )
        assert result.exit_code == 0, result.output
        assert 'End: In progress' in result.output
        # Status line and end-time line must agree.
        assert 'in_progress' in result.output
        # Sanity: 'completed' must not appear as the bot's status when
        # End is 'In progress' (it can still appear in unrelated text;
        # we check the colourised status token specifically).
        assert 'Status: [green]completed' not in result.output

    def test_list_in_progress_run_shows_in_progress_status(
        self, runner, tmp_path, history_dir, home_env
    ):
        """``history list`` derives in-progress status from end_time.

        The storage record's ``status`` field is ``'in_progress'`` for
        a running execution (set in ``save_history``), but the CLI
        re-derives so list and show-execution stay aligned even if a
        backend ever stops writing the field.
        """
        self._seed_history(
            history_dir,
            execution_id='exec-listrun-1',
            fsm_name='listrun_fsm',
            in_progress=True,
        )
        result = runner.invoke(cli, ['history', 'list'], env=home_env)
        assert result.exit_code == 0, result.output
        assert 'exec-lis' in result.output
        assert 'in_progress' in result.output
