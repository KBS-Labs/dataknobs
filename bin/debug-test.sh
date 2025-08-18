#!/bin/bash
# Debug script to test module resolution

set -e

echo "=== Testing different pytest invocations ==="

echo -e "\n1. Testing with uv run pytest from root (like quality script):"
uv run pytest packages/config/tests/test_config.py::TestConfigBasics::test_empty_config -v 2>&1 | grep -A5 -B5 "ModuleNotFoundError\|PASSED\|FAILED"

echo -e "\n2. Testing with activated venv from root:"
source .venv/bin/activate
pytest packages/config/tests/test_config.py::TestConfigBasics::test_empty_config -v 2>&1 | grep -A5 -B5 "ModuleNotFoundError\|PASSED\|FAILED"
deactivate

echo -e "\n3. Testing from package directory:"
cd packages/config
uv run pytest tests/test_config.py::TestConfigBasics::test_empty_config -v 2>&1 | grep -A5 -B5 "ModuleNotFoundError\|PASSED\|FAILED"
cd ../..

echo -e "\n4. Checking Python path in each scenario:"
echo "uv run from root:"
uv run python -c "import sys; print('\n'.join(sys.path[:5]))"

echo -e "\nActivated venv from root:"
source .venv/bin/activate
python -c "import sys; print('\n'.join(sys.path[:5]))"
deactivate

echo -e "\n5. Checking if packages are importable:"
uv run python -c "import dataknobs_config; print('dataknobs_config imported OK')"
uv run python -c "import dataknobs_data; print('dataknobs_data imported OK')"