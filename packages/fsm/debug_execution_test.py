#!/usr/bin/env python3
"""Debug execution engine directly to verify the fix."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataknobs_fsm.execution.engine import ExecutionEngine

# Check the source code of the execute method to see if our fix is there
import inspect

print("ExecutionEngine._execute_transition source:")
source = inspect.getsource(ExecutionEngine._execute_transition)
print(source[1000:1500])  # Print a portion that should include our fix

# Look for the specific line
if "success = True  # If no exception was thrown" in source:
    print("\n✅ Fix is present in the code!")
else:
    print("\n❌ Fix is NOT present in the code!")