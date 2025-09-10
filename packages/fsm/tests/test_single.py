import pytest
from click.testing import CliRunner
from pathlib import Path
import tempfile
import yaml
from dataknobs_fsm.cli.main import cli

def test_config_validate():
    """Debug test for config validation."""
    runner = CliRunner()
    
    # Create config
    config = {
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
                'transform': {
                    'code': 'data["step"] = 1; data'
                }
            },
            {
                'from': 'process',
                'to': 'end',
                'name': 'finish',
                'transform': {
                    'code': 'data["result"] = data.get("step", 0) * 2; data'
                }
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_file = f.name
    
    try:
        result = runner.invoke(cli, [
            'config', 'validate',
            temp_file
        ])
        
        print(f"Exit code: {result.exit_code}")
        print(f"Output: {result.output}")
        if result.exception:
            print(f"Exception: {result.exception}")
            import traceback
            traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
    finally:
        Path(temp_file).unlink(missing_ok=True)

if __name__ == '__main__':
    test_config_validate()
