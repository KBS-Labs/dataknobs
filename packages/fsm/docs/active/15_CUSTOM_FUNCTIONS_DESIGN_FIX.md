# FSM Custom Functions Design Fix

## Implementation Status: ✅ COMPLETED

This design document proposed solutions for custom function registration in the FSM framework. The recommended approach has been implemented successfully.

## Problem Statement

The current FSM implementation has a significant limitation in how custom functions can be used in configurations:

1. **No Easy Custom Function Registration**: Users cannot easily register their own Python functions to be referenced in FSM configurations
2. **Forced to Use Complex Lambdas**: Users must write complex lambda expressions as strings, which are:
   - Hard to read and maintain
   - Difficult to debug
   - Limited in functionality (can't import modules easily)
   - Prone to syntax errors
3. **Inline Code Limitations**: Inline code blocks have module import issues and are not reusable
4. **Module Import Requirement**: The "custom" function type requires functions to be in importable modules, not suitable for script-level functions

## Current Function Reference Types

```python
class FunctionReference(BaseModel):
    type: Literal["builtin", "custom", "inline"]
    name: str | None = None
    module: str | None = None  # Required for "custom" type
    code: str | None = None     # Required for "inline" type
```

### Problems with Each Type:
- **builtin**: Limited to pre-registered FSM library functions
- **custom**: Requires module path, can't use local functions
- **inline**: String-based code, hard to maintain

## Proposed Solutions

### Solution 1: Add Custom Functions Parameter to SimpleFSM

**Implementation:**
```python
# In simple.py
class SimpleFSM:
    def __init__(
        self,
        config: Union[str, Path, Dict[str, Any]],
        data_mode: DataHandlingMode | None = None,
        resources: Dict[str, Any] | None = None,
        custom_functions: Dict[str, Callable] | None = None  # NEW
    ):
        # ... existing code ...
        
        # Build FSM with custom functions
        builder = FSMBuilder()
        if custom_functions:
            for name, func in custom_functions.items():
                builder.register_function(name, func)
        self._fsm = builder.build(self._config)
```

**Usage Example:**
```python
def validate_email(state_data):
    email = state_data.get('email', '')
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return {"valid": bool(re.match(pattern, email))}

def normalize_data(state):
    data = state.data.copy()
    data['email'] = data.get('email', '').lower()
    return data

# Register functions when creating FSM
fsm = SimpleFSM(
    config,
    custom_functions={
        'validate_email': validate_email,
        'normalize_data': normalize_data
    }
)

# In config, reference by name:
config = {
    "states": [{
        "name": "validate",
        "functions": {
            "validate": "validate_email",  # References custom function
            "transform": "normalize_data"
        }
    }]
}
```

### Solution 2: Add a Function Registry Pattern

**Implementation:**
```python
# New file: function_registry.py
class FunctionRegistry:
    _instance = None
    _functions = {}
    
    @classmethod
    def register(cls, name: str, func: Callable):
        cls._functions[name] = func
    
    @classmethod
    def get(cls, name: str) -> Callable:
        return cls._functions.get(name)
    
    @classmethod
    def clear(cls):
        cls._functions.clear()

# In builder.py - check registry first
def _resolve_function(self, func_ref, expected_type):
    if func_ref.type == "custom":
        # Check global registry first
        func = FunctionRegistry.get(func_ref.name)
        if func:
            return func
        # Fall back to module import...
```

**Usage:**
```python
from dataknobs_fsm import FunctionRegistry

# Register functions before creating FSM
FunctionRegistry.register('validate_email', validate_email)
FunctionRegistry.register('normalize_data', normalize_data)

# Create FSM - functions are available
fsm = create_fsm(config)
```

### Solution 3: Support Function Objects Directly in Config

**Implementation:**
Allow passing actual function objects in the configuration:

```python
# In loader.py
def _transform_state_functions(self, config):
    # ... existing code ...
    
    # Handle function objects directly
    if callable(functions['transform']):
        # Store function in a temporary registry
        func_id = f"_custom_{id(functions['transform'])}"
        self._temp_functions[func_id] = functions['transform']
        state['transforms'] = [{
            'type': 'registered',
            'name': func_id
        }]
```

**Usage:**
```python
config = {
    "states": [{
        "name": "validate",
        "functions": {
            "transform": normalize_data  # Actual function object
        }
    }]
}
```

### Solution 4: Enhanced Inline Functions with Imports

**Implementation:**
Allow inline functions to specify imports:

```python
class FunctionReference(BaseModel):
    type: Literal["builtin", "custom", "inline", "registered"]
    name: str | None = None
    module: str | None = None
    code: str | None = None
    imports: List[str] | None = None  # NEW
```

**Usage:**
```python
config = {
    "functions": {
        "transform": {
            "type": "inline",
            "imports": ["import re", "from datetime import datetime"],
            "code": """
def transform(state):
    # Can use imported modules
    data = state.data.copy()
    data['email'] = re.sub(r'\\s+', '', data.get('email', ''))
    data['timestamp'] = datetime.now().isoformat()
    return data
"""
        }
    }
}
```

## Recommended Approach

**Combine Solutions 1 and 3** for maximum flexibility:

1. **Add `custom_functions` parameter to SimpleFSM** (Solution 1)
   - Easy to use for script-level development
   - Clear and pythonic
   - No global state issues

2. **Support function objects in config** (Solution 3)
   - Most intuitive for Python developers
   - No string parsing needed
   - Full IDE support (autocomplete, type hints)

3. **Keep existing types for backward compatibility**
   - `inline` for simple lambdas
   - `custom` for module imports
   - `builtin` for library functions

## Implementation Priority

1. **Phase 1**: Add `custom_functions` parameter to SimpleFSM
   - Minimal changes required
   - Immediate benefit for users
   - Backward compatible

2. **Phase 2**: Support function objects in config
   - More complex but more intuitive
   - May require config validation updates

3. **Phase 3**: Enhanced inline functions with imports
   - For advanced use cases
   - Lower priority

## Benefits

1. **Developer Experience**
   - Write functions in Python, not strings
   - Full IDE support (syntax highlighting, autocomplete)
   - Easier debugging and testing

2. **Code Quality**
   - Reusable functions
   - Testable components
   - Type hints support

3. **Maintainability**
   - Functions in version control
   - Clear separation of logic and configuration
   - Easier refactoring

## Example: Fixed Validator Implementation

With the proposed fix:

```python
# validators.py - Clean Python functions
def validate_email(state_data):
    email = state_data.get('email', '')
    if not email:
        return {"valid": False, "error": "Email required"}
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return {"valid": False, "error": "Invalid email format"}
    
    return {"valid": True}

def normalize_email(state):
    data = state.data.copy()
    data['email'] = data.get('email', '').lower().strip()
    return data

# main.py - Use functions in FSM
from validators import validate_email, normalize_email

config = {
    "name": "EmailValidator",
    "networks": [{
        "name": "main",
        "states": [{
            "name": "normalize",
            "functions": {
                "transform": "normalize_email"  # Reference by name
            }
        }, {
            "name": "validate",
            "functions": {
                "validate": "validate_email"
            }
        }]
    }]
}

# Create FSM with custom functions
fsm = SimpleFSM(
    config,
    custom_functions={
        'validate_email': validate_email,
        'normalize_email': normalize_email
    }
)

# Or with create_fsm helper
fsm = create_fsm(
    config,
    custom_functions={
        'validate_email': validate_email,
        'normalize_email': normalize_email
    }
)
```

## Implementation Details

### What Was Implemented

1. **Custom Functions Parameter in SimpleFSM** ✅
   - Added `custom_functions` parameter to `SimpleFSM.__init__()`
   - Functions are registered with the FSMBuilder before building the FSM
   - Clean, pythonic API for users

2. **New "registered" Function Type** ✅
   - Added `"registered"` to the FunctionReference type literals
   - Allows referencing functions by name without module path
   - Schema validation ensures registered functions have a name

3. **Function Registration in FSMBuilder** ✅
   - Functions are properly registered and resolved
   - Fixed issue where wrapped functions weren't matched in registry
   - Preserved function names through wrapping process

4. **Proper Error Handling** ✅
   - Added `FunctionError` exception for deterministic function failures
   - Distinguished between recoverable and non-recoverable errors
   - State transform failures mark state as failed but continue execution
   - Arc transform failures prevent transitions as designed

### Usage Example

```python
# Define custom functions
def validate_email(state):
    data = state.data.copy()
    # validation logic
    return data

def normalize_phone(state):
    data = state.data.copy()
    # normalization logic
    return data

# Register with FSM
fsm = SimpleFSM(
    config,
    custom_functions={
        'validate_email': validate_email,
        'normalize_phone': normalize_phone
    }
)

# Reference in configuration
config = {
    "states": [{
        "name": "validate",
        "functions": {
            "transform": {
                "type": "registered",
                "name": "validate_email"
            }
        }
    }]
}
```

### Test Coverage

Comprehensive tests were added in `test_custom_functions.py`:
- Custom function registration
- Function resolution and execution
- State transform error handling
- Arc transform error handling
- Validation pipelines with routing

## Conclusion

The implementation successfully addresses all the identified problems:

1. ✅ **Easy Custom Function Registration** - Users can register Python functions with SimpleFSM
2. ✅ **No More Complex Lambdas** - Functions are written as normal Python code
3. ✅ **Full IDE Support** - Syntax highlighting, autocomplete, and debugging work normally
4. ✅ **Reusable and Testable** - Functions can be tested independently and reused
5. ✅ **Backward Compatible** - Existing inline and custom types still work

The FSM framework is now more Pythonic, maintainable, and developer-friendly while maintaining full backward compatibility.