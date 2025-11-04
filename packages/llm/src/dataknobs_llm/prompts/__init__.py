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
        params={'language': 'python', 'code': code}
    )

For more information, see the documentation in the design document.
"""

# Core types and validation
from .base import (
    ValidationLevel,
    ValidationConfig,
    PromptTemplateDict,
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
    InMemoryAdapter,
    InMemoryAsyncAdapter,
)

# Template rendering
from .rendering import (
    TemplateRenderer,
    TemplateSyntaxError,
    render_template,
    render_template_strict,
)

# Prompt library implementations
from .implementations import (
    FileSystemPromptLibrary,
    ConfigPromptLibrary,
    CompositePromptLibrary,
    VersionedPromptLibrary,
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

# Versioning and A/B testing
from .versioning import (
    VersionManager,
    ABTestManager,
    MetricsCollector,
    PromptVersion,
    PromptExperiment,
    PromptVariant,
    PromptMetrics,
    VersioningError,
    VersionStatus,
    MetricEvent,
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
    "PromptTemplateDict",
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
    "InMemoryAdapter",
    "InMemoryAsyncAdapter",

    # Rendering
    "TemplateRenderer",
    "TemplateSyntaxError",
    "render_template",
    "render_template_strict",

    # Library implementations
    "FileSystemPromptLibrary",
    "ConfigPromptLibrary",
    "CompositePromptLibrary",
    "VersionedPromptLibrary",

    # Builders
    "PromptBuilder",
    "AsyncPromptBuilder",

    # Template composition
    "TemplateComposer",

    # Versioning and A/B testing
    "VersionManager",
    "ABTestManager",
    "MetricsCollector",
    "PromptVersion",
    "PromptExperiment",
    "PromptVariant",
    "PromptMetrics",
    "VersioningError",
    "VersionStatus",
    "MetricEvent",
]
