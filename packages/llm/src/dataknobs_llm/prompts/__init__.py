"""Advanced prompt engineering library for dataknobs_llm.

This package provides a comprehensive prompt management system with:

- **Resource Adapters**: Plug in any data source (dicts, databases, vector stores)
- **Template Rendering**: CONDITIONAL strategy with {{variables}} and ((conditionals))
- **Validation System**: Configurable ERROR/WARN/IGNORE levels
- **RAG Integration**: Explicit placement with {{RAG_CONTENT}} placeholders
- **Prompt Libraries**: Filesystem, config, and composite implementations
- **Builder Pattern**: PromptBuilder (sync) and AsyncPromptBuilder (async)

Quick Start:

    from dataknobs_llm.prompts import (
        PromptBuilder,
        DictResourceAdapter,
        FileSystemPromptLibrary,
        ValidationLevel
    )

    # Create a library
    library = FileSystemPromptLibrary(prompt_dir=Path("prompts/"))

    # Create a builder with adapters
    builder = PromptBuilder(
        library=library,
        adapters={'config': DictResourceAdapter(config_dict)}
    )

    # Render a prompt
    result = builder.render_user_prompt(
        'analyze_code',
        index=0,
        params={'language': 'python', 'code': code}
    )

For more information, see the documentation in the design document.
"""

# Core types and validation
from .base import (
    ValidationLevel,
    ValidationConfig,
    PromptTemplate,
    RAGConfig,
    MessageIndex,
    RenderResult,
    AbstractPromptLibrary,
    BasePromptLibrary,
)

# Resource adapters
from .adapters import (
    ResourceAdapter,
    AsyncResourceAdapter,
    ResourceAdapterBase,
    BaseSearchLogic,
    DictResourceAdapter,
    AsyncDictResourceAdapter,
    DataknobsBackendAdapter,
    AsyncDataknobsBackendAdapter,
)

# Template rendering
from .rendering import (
    TemplateRenderer,
    render_template,
    render_template_strict,
)

# Prompt library implementations
from .implementations import (
    FileSystemPromptLibrary,
    ConfigPromptLibrary,
    CompositePromptLibrary,
)

# Prompt builders
from .builders import (
    PromptBuilder,
    AsyncPromptBuilder,
)

# Template composition
from .utils import (
    TemplateComposer,
)

# Version info
__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",

    # Validation
    "ValidationLevel",
    "ValidationConfig",

    # Types
    "PromptTemplate",
    "RAGConfig",
    "MessageIndex",
    "RenderResult",

    # Base classes
    "AbstractPromptLibrary",
    "BasePromptLibrary",

    # Adapters
    "ResourceAdapter",
    "AsyncResourceAdapter",
    "ResourceAdapterBase",
    "BaseSearchLogic",
    "DictResourceAdapter",
    "AsyncDictResourceAdapter",
    "DataknobsBackendAdapter",
    "AsyncDataknobsBackendAdapter",

    # Rendering
    "TemplateRenderer",
    "render_template",
    "render_template_strict",

    # Library implementations
    "FileSystemPromptLibrary",
    "ConfigPromptLibrary",
    "CompositePromptLibrary",

    # Builders
    "PromptBuilder",
    "AsyncPromptBuilder",

    # Template composition
    "TemplateComposer",
]
