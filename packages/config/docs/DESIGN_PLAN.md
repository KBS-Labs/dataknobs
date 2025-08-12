# Config Package Design Plan

## Overview
A modular, reusable configuration system for composable settings with environment variable overrides, file loading, and optional object construction helpers.

## Core Architecture

### 1. Data Structure
```python
# Internal structure
{
    "type_1": [
        {"name": "obj1", "param1": "value1", ...},  # Atomic config 1
        {"name": "obj2", "param2": "value2", ...},  # Atomic config 2
    ],
    "type_2": [
        {"name": "obj3", "param3": "value3", ...},  # Atomic config 3
    ],
    "settings": {  # Global settings (special type)
        "config_root": "/path/to/config",
        "global_root": "/path/to/global",
        "type_1.global_root": "/path/to/type1",
        "path_resolution_attributes": ["path", "type_1.config_path"],
        "default_timeout": 30,  # Global default
        "type_1.default_retry": 3,  # Type-specific default
    }
}
```

### 2. Core Components

#### Config Class
- **Storage**: Internal dictionary of lists of atomic configurations
- **Loading**: Support for YAML, JSON files and dictionaries
- **Merging**: Combine multiple configurations with precedence rules
- **Access**: Type-based and name/index-based getters/setters

#### String Reference System (xref)
- **Format**: `xref:<type>[<name_or_index>]`
- **Examples**:
  - `xref:foo[bar]` - Named reference
  - `xref:foo[0]` - Index reference
  - `xref:foo[-1]` - Last item reference
  - `xref:foo` - First/only item reference
- **Resolution**: Decode references to actual configuration objects

#### Environment Variable Overrides
- **Format**: Modified xref format for bash compatibility
  - `xref:foo[bar].param` → `DATAKNOBS_FOO__BAR__PARAM`
  - `xref:foo[0].param` → `DATAKNOBS_FOO__0__PARAM`
- **Application**: Override atomic config values at load time

#### Path Resolution
- **Trigger**: Attributes listed in `path_resolution_attributes`
- **Base**: Relative to `config_root` or type-specific roots
- **Storage**: Convert to absolute paths on load

#### File References
- **Format**: `@path/to/config.yaml`
- **Resolution**: Load referenced file as atomic configuration
- **Base Path**: Relative to `config_root` setting

### 3. Optional Object Construction
- **Direct Construction**: Use `class` attribute to instantiate objects
- **Factory Pattern**: Use `factory` attribute for factory-based construction
- **Caching**: Cache constructed objects by reference

## API Design

### Core Methods
```python
class Config:
    # Construction
    def __init__(self, *sources, **kwargs)
    def from_file(cls, path: str) -> Config
    def from_dict(cls, data: dict) -> Config
    
    # Access
    def get_types(self) -> List[str]
    def get_count(self, type_name: str) -> int
    def get_names(self, type_name: str) -> List[str]
    def get(self, type_name: str, name_or_index: Union[str, int] = 0) -> dict
    def set(self, type_name: str, name_or_index: Union[str, int], config: dict)
    
    # References
    def resolve_reference(self, ref: str) -> dict
    def build_reference(self, type_name: str, name_or_index: Union[str, int]) -> str
    
    # Merging
    def merge(self, other: Config, precedence: str = "first")
    
    # Object Construction (optional)
    def build_object(self, ref: str, cache: bool = True) -> Any
```

## Implementation Phases

### Phase 1: Core Structure
1. Basic Config class with dictionary storage
2. Type/name-based access methods
3. Basic validation

### Phase 2: File Loading
1. YAML/JSON file parsing
2. Dictionary initialization
3. File reference resolution (@-prefixed)

### Phase 3: Reference System
1. String reference parsing (xref:)
2. Reference resolution
3. Cross-reference support in configs

### Phase 4: Environment Variables
1. Environment variable naming scheme
2. Override application logic
3. Type conversion for overrides

### Phase 5: Settings & Defaults
1. Global settings management
2. Type-specific defaults
3. Path resolution logic
4. Precedence rules

### Phase 6: Object Construction
1. Class instantiation support
2. Factory pattern support
3. Object caching mechanism

## Testing Strategy
- Unit tests for each component
- Integration tests for file loading
- Environment variable override tests
- Path resolution tests
- Object construction tests

## Dependencies
- PyYAML for YAML support
- Standard library json for JSON support
- No other external dependencies for core functionality

## File Structure
```
packages/config/
├── DESIGN_PLAN.md (this file)
├── PROGRESS_CHECKLIST.md
├── README.md
├── pyproject.toml
├── src/
│   └── dataknobs_config/
│       ├── __init__.py
│       ├── config.py
│       ├── references.py
│       ├── environment.py
│       ├── builders.py
│       └── exceptions.py
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_references.py
    ├── test_environment.py
    ├── test_builders.py
    └── fixtures/
        ├── test_config.yaml
        └── test_config.json
```