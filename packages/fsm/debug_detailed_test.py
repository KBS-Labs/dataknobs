#!/usr/bin/env python3
"""More detailed debugging to trace exactly where execution stops."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataknobs_fsm.execution.engine import ExecutionEngine

# Store the original method
original_execute_transition = ExecutionEngine._execute_transition

def debug_execute_transition(self, context, arc):
    """Debug wrapper with step-by-step tracing."""
    print(f"\n=== DEBUGGING _execute_transition step by step ===")
    print(f"Arc: {arc}")
    print(f"Context state: {context.current_state}")
    print(f"Context data: {context.data}")
    
    retry_count = 0
    print(f"1. Starting retry loop, max_retries: {self.max_retries}")
    
    while retry_count <= self.max_retries:
        print(f"2. Retry attempt {retry_count}")
        try:
            # Validate data before processing
            if context.data is None:
                print("3. Data is None, returning False")
                return False
                
            print("4. Data is not None, continuing")
            
            # Create arc execution
            if hasattr(self.fsm, 'function_registry'):
                if hasattr(self.fsm.function_registry, 'functions'):
                    func_reg = self.fsm.function_registry.functions
                else:
                    func_reg = {}
            else:
                func_reg = {}
                
            print(f"5. Function registry: {func_reg}")
                
            from dataknobs_fsm.core.arc import ArcExecution
            arc_exec = ArcExecution(
                arc,
                source_state=context.current_state or "",
                function_registry=func_reg
            )
            
            print("6. Created ArcExecution, about to execute")
            
            # Execute with resource context
            result = arc_exec.execute(context, context.data)
            success = True  # If no exception was thrown, the arc execution succeeded
            
            print(f"7. Arc execution result: {result}")
            print(f"8. Success set to: {success}")
            
            if success:
                print("9. In success block")
                # Update data with result
                if result is not None:
                    print(f"10. Updating context data from {context.data} to {result}")
                    context.data = result
                
                # Update state
                print(f"11. Setting state to: {arc.target_state}")
                context.set_state(arc.target_state)
                self._transition_count += 1
                
                # Execute state transforms when entering the new state
                print(f"12. About to execute state transforms for: {arc.target_state}")
                self._execute_state_transforms(context, arc.target_state)
                print("13. State transforms completed")
                
                # Fire post-transition hooks
                if self.enable_hooks:
                    print("14. Firing post-transition hooks")
                    for hook in self._post_transition_hooks:
                        hook(context, arc)
                
                print("15. About to return True")
                return True
            else:
                print("16. In else block (this shouldn't happen!)")
                return False
                
        except (TypeError, AttributeError, ValueError) as e:
            print(f"17. Data type error: {e}")
            self._error_count += 1
            return False
            
        except Exception as e:
            print(f"18. General exception: {e}")
            import traceback
            traceback.print_exc()
            self._error_count += 1
            retry_count += 1
            if retry_count <= self.max_retries:
                import time
                time.sleep(self.retry_delay * retry_count)
            else:
                return False
    
    print("19. Exited retry loop, returning False")
    return False

# Monkey patch
ExecutionEngine._execute_transition = debug_execute_transition

# Now run the test
from dataknobs_fsm.api.simple import SimpleFSM

config = {
    'name': 'debug_fsm',
    'main_network': 'main',
    'networks': [{
        'name': 'main',
        'states': [
            {
                'name': 'input',
                'is_start': True
            },
            {
                'name': 'multiply',
                'functions': {
                    'transform': 'lambda state: {"result": state.data.get("value", 1) * 2}'
                }
            },
            {
                'name': 'output',
                'is_end': True
            }
        ],
        'arcs': [
            {'from': 'input', 'to': 'multiply', 'name': 'process'},
            {'from': 'multiply', 'to': 'output', 'name': 'done'}
        ]
    }]
}

print("Creating FSM...")
fsm = SimpleFSM(config)

print("Calling fsm.process()...")
result = fsm.process({'value': 5})

print(f"\nFinal result: {result}")