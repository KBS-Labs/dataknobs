"""Deterministic content generators for structured output production.

This package provides:
- **Generator**: Abstract base class defining the generator interface
- **GeneratorContext**: Dependencies available during generation
- **GeneratorOutput**: Result with content, provenance, and validation
- **GeneratorDefinition**: Serializable snapshot of a registered generator
- **TemplateGenerator**: Jinja2 template-based generator (YAML/JSON output)
- **GeneratorRegistry**: Database-backed registry with artifact integration
"""

from .base import (
    Generator,
    GeneratorContext,
    GeneratorDefinition,
    GeneratorOutput,
)
from .registry import GeneratorRegistry
from .template_generator import TemplateGenerator

__all__ = [
    "Generator",
    "GeneratorContext",
    "GeneratorDefinition",
    "GeneratorOutput",
    "GeneratorRegistry",
    "TemplateGenerator",
]
