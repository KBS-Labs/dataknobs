"""Generator registry for managing generators with database-backed metadata.

The registry stores generator metadata (id, version, schemas) in an
AsyncDatabase while keeping generator instances in memory (code and
templates cannot be serialized). Optionally integrates with ArtifactRegistry
to store generated output as artifacts.

Example:
    >>> from dataknobs_data.backends.memory import AsyncMemoryDatabase
    >>> db = AsyncMemoryDatabase()
    >>> registry = GeneratorRegistry(db)
    >>> await registry.register(my_generator)
    >>> output = await registry.generate("my_gen", {"param": "value"})
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_data import AsyncDatabase, Record

from .base import Generator, GeneratorContext, GeneratorOutput
from .template_generator import TemplateGenerator

logger = logging.getLogger(__name__)


class GeneratorRegistry:
    """Registry for managing generator instances with database-backed metadata.

    Generator definitions (id, version, schemas) are stored as Records
    in the database. Generator instances are held in memory since they
    contain code or templates that cannot be serialized.

    Args:
        db: Async database backend for metadata storage.
        artifact_registry: Optional artifact registry for storing generated
            output as artifacts.
    """

    def __init__(
        self,
        db: AsyncDatabase,
        artifact_registry: Any | None = None,
    ) -> None:
        self._db = db
        self._artifact_registry = artifact_registry
        self._generators: dict[str, Generator] = {}

    async def register(self, generator: Generator) -> str:
        """Register a generator instance.

        Stores the generator's metadata in the database and keeps the
        instance in memory.

        Args:
            generator: The generator to register.

        Returns:
            The generator's ID.
        """
        metadata = {
            "generator_id": generator.id,
            "version": generator.version,
            "parameter_schema": generator.parameter_schema,
            "output_schema": generator.output_schema,
        }
        await self._db.upsert(
            f"gen:{generator.id}",
            Record(metadata),
        )
        self._generators[generator.id] = generator

        logger.info(
            "Registered generator '%s' v%s",
            generator.id,
            generator.version,
        )

        return generator.id

    async def get(self, generator_id: str) -> Generator | None:
        """Retrieve a registered generator by ID.

        Args:
            generator_id: The generator identifier.

        Returns:
            The generator instance, or None if not found.
        """
        return self._generators.get(generator_id)

    async def list_all(self) -> list[str]:
        """List all registered generator IDs.

        Returns:
            List of generator IDs.
        """
        return list(self._generators.keys())

    async def generate(
        self,
        generator_id: str,
        parameters: dict[str, Any],
        context: GeneratorContext | None = None,
    ) -> GeneratorOutput:
        """Generate content using a registered generator.

        Steps:
        1. Look up the generator by ID
        2. Validate parameters
        3. Run generation
        4. Validate output
        5. If artifact_registry is available, create an artifact
        6. Return the output

        Args:
            generator_id: The generator to use.
            parameters: Input parameters for the generator.
            context: Optional dependencies for generation.

        Returns:
            The generated output.

        Raises:
            ValueError: If the generator is not found or parameter validation fails.
        """
        generator = self._generators.get(generator_id)
        if generator is None:
            raise ValueError(f"Generator '{generator_id}' not found")

        param_errors = await generator.validate_parameters(parameters)
        if param_errors:
            raise ValueError(
                f"Parameter validation failed for generator '{generator_id}': "
                f"{'; '.join(param_errors)}"
            )

        output = await generator.generate(parameters, context)

        output_errors = await generator.validate_output(output.content)
        output.validation_errors.extend(output_errors)

        if self._artifact_registry is not None and not output.validation_errors:
            await self._create_artifact(generator, output)

        return output

    async def _create_artifact(
        self,
        generator: Generator,
        output: GeneratorOutput,
    ) -> None:
        """Create an artifact from generator output.

        Args:
            generator: The generator that produced the output.
            output: The generated output.
        """
        artifact = await self._artifact_registry.create(
            artifact_type=output.metadata.get("artifact_type", "generated_content"),
            name=f"Generated by {generator.id} v{generator.version}",
            content=output.content,
            provenance=output.provenance,
        )
        output.metadata["artifact_id"] = artifact.id

        logger.info(
            "Created artifact '%s' from generator '%s'",
            artifact.id,
            generator.id,
        )

    @classmethod
    async def from_config(
        cls,
        config: dict[str, Any],
        db: AsyncDatabase,
        artifact_registry: Any | None = None,
    ) -> GeneratorRegistry:
        """Create a GeneratorRegistry from configuration.

        The config should have a ``"generators"`` key containing a list of
        generator configurations. Currently supports ``"template"`` type.

        Args:
            config: Configuration dictionary.
            db: Async database backend.
            artifact_registry: Optional artifact registry.

        Returns:
            A configured GeneratorRegistry with generators loaded.
        """
        registry = cls(db=db, artifact_registry=artifact_registry)

        for gen_config in config.get("generators", []):
            gen_type = gen_config.get("type", "template")

            if gen_type == "template":
                generator = TemplateGenerator.from_config(gen_config)
                await registry.register(generator)
            else:
                logger.warning(
                    "Unknown generator type '%s' in config, skipping",
                    gen_type,
                )

        return registry
