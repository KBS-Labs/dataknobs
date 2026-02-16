"""Template-based content generator using Jinja2.

Renders Jinja2 templates with input parameters to produce structured
content in YAML or JSON format. Output is validated against JSON Schema
and tracked via provenance.

Example:
    >>> gen = TemplateGenerator(
    ...     generator_id="greeting",
    ...     version="1.0.0",
    ...     template="greeting: Hello {{ name }}!",
    ...     parameter_schema={"type": "object", "required": ["name"],
    ...         "properties": {"name": {"type": "string"}}},
    ...     output_schema={"type": "object",
    ...         "properties": {"greeting": {"type": "string"}}},
    ... )
    >>> output = await gen.generate({"name": "World"})
    >>> output.content
    {'greeting': 'Hello World!'}
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_bots.artifacts.provenance import ToolInvocation, create_provenance

from .base import Generator, GeneratorContext, GeneratorOutput

logger = logging.getLogger(__name__)


class TemplateGenerator(Generator):
    """Generator that renders Jinja2 templates to produce structured content.

    Templates are rendered with input parameters, then parsed as YAML or
    JSON to produce the output dictionary. Provenance records the template
    version and parameters used.

    Args:
        generator_id: Unique generator identifier.
        version: Semantic version string.
        template: Jinja2 template string.
        parameter_schema: JSON Schema for input parameters.
        output_schema: JSON Schema for output content.
        output_format: Output format ("yaml" or "json"). Defaults to "yaml".
    """

    def __init__(
        self,
        generator_id: str,
        version: str,
        template: str,
        parameter_schema: dict[str, Any],
        output_schema: dict[str, Any],
        output_format: str = "yaml",
    ) -> None:
        self._id = generator_id
        self._version = version
        self._template = template
        self._parameter_schema = parameter_schema
        self._output_schema = output_schema
        self._output_format = output_format

    @property
    def id(self) -> str:
        """Unique generator identifier."""
        return self._id

    @property
    def version(self) -> str:
        """Semantic version string."""
        return self._version

    @property
    def parameter_schema(self) -> dict[str, Any]:
        """JSON Schema for input parameters."""
        return self._parameter_schema

    @property
    def output_schema(self) -> dict[str, Any]:
        """JSON Schema for output content."""
        return self._output_schema

    async def generate(
        self,
        parameters: dict[str, Any],
        context: GeneratorContext | None = None,
    ) -> GeneratorOutput:
        """Generate content by rendering the template with parameters.

        Steps:
        1. Validate parameters against parameter_schema
        2. Render Jinja2 template
        3. Parse rendered output (YAML or JSON)
        4. Validate output against output_schema
        5. Build provenance record
        6. Return GeneratorOutput

        Args:
            parameters: Input parameters for template rendering.
            context: Optional dependencies (unused by template generator).

        Returns:
            GeneratorOutput with rendered content and provenance.

        Raises:
            ValueError: If parameter validation fails.
        """
        param_errors = await self.validate_parameters(parameters)
        if param_errors:
            raise ValueError(
                f"Parameter validation failed: {'; '.join(param_errors)}"
            )

        rendered = self._render_template(parameters)
        content = self._parse_output(rendered)

        output_errors = await self.validate_output(content)

        provenance = create_provenance(
            created_by=f"system:generator:{self._id}",
            creation_method="generator",
            creation_context={
                "generator_id": self._id,
                "generator_version": self._version,
                "parameters": parameters,
                "output_format": self._output_format,
            },
            tool_chain=[
                ToolInvocation(
                    tool_name=f"generator:{self._id}",
                    tool_version=self._version,
                    parameters=parameters,
                ),
            ],
        )

        return GeneratorOutput(
            content=content,
            provenance=provenance,
            validation_errors=output_errors,
            metadata={
                "generator_id": self._id,
                "generator_version": self._version,
                "output_format": self._output_format,
            },
        )

    def _render_template(self, parameters: dict[str, Any]) -> str:
        """Render the Jinja2 template with the given parameters.

        Raises:
            ValueError: If the template references undefined variables.
        """
        import jinja2

        try:
            template = jinja2.Template(
                self._template, undefined=jinja2.StrictUndefined
            )
            return template.render(**parameters)
        except jinja2.UndefinedError as e:
            raise ValueError(f"Template rendering failed: {e}") from e

    def _parse_output(self, rendered: str) -> dict[str, Any]:
        """Parse rendered template output as YAML or JSON.

        Raises:
            ValueError: If the rendered output cannot be parsed or is not a dict.
        """
        if self._output_format == "json":
            return self._parse_json(rendered)
        return self._parse_yaml(rendered)

    def _parse_yaml(self, text: str) -> dict[str, Any]:
        """Parse text as YAML, returning a dictionary."""
        import yaml

        try:
            result = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse template output as YAML: {e}") from e

        if not isinstance(result, dict):
            raise ValueError(
                f"Template output must be a YAML mapping, got {type(result).__name__}"
            )
        return result

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse text as JSON, returning a dictionary."""
        import json

        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse template output as JSON: {e}") from e

        if not isinstance(result, dict):
            raise ValueError(
                f"Template output must be a JSON object, got {type(result).__name__}"
            )
        return result

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TemplateGenerator:
        """Create a TemplateGenerator from a configuration dictionary.

        Config keys:
        - ``id`` (required): Generator identifier.
        - ``version`` (required): Semantic version.
        - ``template`` (required): Jinja2 template string.
        - ``parameter_schema`` (required): JSON Schema for parameters.
        - ``output_schema`` (required): JSON Schema for output.
        - ``output_format`` (optional): "yaml" or "json", defaults to "yaml".

        Args:
            config: Configuration dictionary.

        Returns:
            A configured TemplateGenerator.

        Raises:
            ValueError: If required config keys are missing.
        """
        required_keys = ["id", "version", "template", "parameter_schema", "output_schema"]
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ValueError(
                f"TemplateGenerator config missing required keys: {missing}"
            )

        return cls(
            generator_id=config["id"],
            version=config["version"],
            template=config["template"],
            parameter_schema=config["parameter_schema"],
            output_schema=config["output_schema"],
            output_format=config.get("output_format", "yaml"),
        )
