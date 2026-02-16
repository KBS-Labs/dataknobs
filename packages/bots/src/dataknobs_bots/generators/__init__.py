"""Deterministic content generators for structured output production.

This package provides:
- **Generator**: Abstract base class defining the generator interface
- **GeneratorContext**: Dependencies available during generation
- **GeneratorOutput**: Result with content, provenance, and validation
- **TemplateGenerator**: Jinja2 template-based generator (YAML/JSON output)
- **GeneratorRegistry**: Database-backed registry with artifact integration
"""

from .base import Generator, GeneratorContext, GeneratorOutput
from .registry import GeneratorRegistry
from .template_generator import TemplateGenerator

__all__ = [
    "Generator",
    "GeneratorContext",
    "GeneratorOutput",
    "GeneratorRegistry",
    "TemplateGenerator",
]
