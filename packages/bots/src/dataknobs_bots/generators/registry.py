"""Generator registry for managing generators with database-backed metadata.

The registry stores a :class:`GeneratorDefinition` snapshot of each
generator (id, version, schemas) in an :class:`AsyncDatabase` while
keeping the live ``Generator`` instance in memory (code and templates
cannot be serialized).  Optionally integrates with
:class:`ArtifactRegistry` to store generated output as artifacts.

Internally composes :class:`AsyncKeyedRecordStore[GeneratorDefinition]`
so every ``Record`` is built in one place (the store's serializer) and
the ``metadata`` channel is preserved by construction.  The serializer
signature ``(GeneratorDefinition) -> (data, metadata)`` makes the
metadata column part of the function's type — a future change to the
definition cannot silently drop the metadata channel without a
type-visible diff.  This also closes the historical "shadow bug" at
this site, where a local variable named ``metadata`` was passed
positionally to ``Record(metadata)`` and silently routed into the data
column.  All ``Record`` construction now flows through the keyed
store's serializer, so the variable-name collision cannot recur.

Example:
    >>> from dataknobs_data.backends.memory import AsyncMemoryDatabase
    >>> db = AsyncMemoryDatabase()
    >>> registry = GeneratorRegistry(db)
    >>> await registry.register(my_generator, metadata={"tenant_id": "acme"})
    >>> output = await registry.generate("my_gen", {"param": "value"})
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from dataknobs_data import AsyncDatabase, AsyncKeyedRecordStore, SortSpec

from .base import Generator, GeneratorContext, GeneratorDefinition, GeneratorOutput
from .template_generator import TemplateGenerator

if TYPE_CHECKING:
    from dataknobs_data import Record

logger = logging.getLogger(__name__)


def _definition_to_columns(
    defn: GeneratorDefinition,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a :class:`GeneratorDefinition` into ``(data, metadata)``.

    The two-channel return type is load-bearing: the metadata column is
    part of this function's *signature*, so a future change to the
    definition cannot accidentally drop the metadata channel without a
    type-visible diff at this site.  This is the structural fix for the
    historical shadow bug at this site.
    """
    data: dict[str, Any] = {
        "generator_id": defn.generator_id,
        "version": defn.version,
        "parameter_schema": dict(defn.parameter_schema),
        "output_schema": dict(defn.output_schema),
    }
    return data, dict(defn.metadata)


def _definition_from_record(record: Record) -> GeneratorDefinition:
    """Reconstruct a :class:`GeneratorDefinition` from a stored ``Record``."""
    data = record.data or {}
    return GeneratorDefinition(
        generator_id=data.get("generator_id", ""),
        version=data.get("version", ""),
        parameter_schema=dict(data.get("parameter_schema") or {}),
        output_schema=dict(data.get("output_schema") or {}),
        metadata=dict(record.metadata or {}),
    )


class GeneratorRegistry:
    """Registry for managing generator instances with database-backed metadata.

    A :class:`GeneratorDefinition` snapshot of each registered generator
    (id, version, schemas) is stored as a ``Record`` in the database via
    an internal :class:`AsyncKeyedRecordStore[GeneratorDefinition]`.
    Live ``Generator`` instances themselves are held in memory because
    they contain code or templates that cannot be serialized.

    Args:
        db: Async database backend for definition storage.
        artifact_registry: Optional artifact registry for storing generated
            output as artifacts.
    """

    def __init__(
        self,
        db: AsyncDatabase,
        artifact_registry: Any | None = None,
    ) -> None:
        self._store: AsyncKeyedRecordStore[GeneratorDefinition] = AsyncKeyedRecordStore[
            GeneratorDefinition
        ](
            db,
            serializer=_definition_to_columns,
            deserializer=_definition_from_record,
        )
        self._artifact_registry = artifact_registry
        self._generators: dict[str, Generator] = {}

    async def register(
        self,
        generator: Generator,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register a generator instance.

        Stores a :class:`GeneratorDefinition` snapshot in the database
        and keeps the live ``Generator`` instance in memory (its code
        and templates cannot be serialized).

        Args:
            generator: The generator to register.
            metadata: Cross-cutting context routed to the underlying
                record's ``metadata`` column so it is independently
                filterable via ``filter_metadata`` on
                :meth:`list_definitions` without scanning every row.

        Returns:
            The generator's ID.
        """
        defn = GeneratorDefinition.from_generator(generator, metadata=metadata)
        await self._store.put(f"gen:{generator.id}", defn)
        self._generators[generator.id] = generator

        logger.info(
            "Registered generator '%s' v%s",
            generator.id,
            generator.version,
        )

        return generator.id

    async def get(self, generator_id: str) -> Generator | None:
        """Retrieve a live registered generator instance by ID.

        Args:
            generator_id: The generator identifier.

        Returns:
            The live generator instance, or None if not found.
        """
        return self._generators.get(generator_id)

    async def get_definition(self, generator_id: str) -> GeneratorDefinition | None:
        """Read the persisted :class:`GeneratorDefinition` for a generator.

        Returns the definition snapshot from the backing database (which
        includes the metadata column), not the live ``Generator``
        instance.  Use :meth:`get` to retrieve the live instance.

        Args:
            generator_id: The generator identifier.

        Returns:
            The persisted definition, or ``None`` if not registered.
        """
        return await self._store.get(f"gen:{generator_id}")

    async def list_all(self) -> list[str]:
        """List all registered generator IDs.

        Returns:
            List of generator IDs.
        """
        return list(self._generators.keys())

    async def list_definitions(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[GeneratorDefinition]:
        """List persisted generator definitions, optionally filtered.

        Args:
            filter_metadata: Equality filter over the ``metadata``
                column.  Entries are routed via the ``metadata.X``
                field-path convention so SQL/JSONB backends push the
                filter into the indexable column.
            sort: Optional multi-key sort specification, pushed down to
                the database query so SQL backends can use indexes when
                available.
            limit: Optional row limit, pushed down to the database.
                Unlike the dual-write registries (artifacts, rubrics),
                ``GeneratorRegistry`` writes a single row per generator
                id (``gen:{id}``) — no latest-pointer / snapshot
                divergence — so the database's row count IS the
                definition count and pagination is safe to push down.
            offset: Optional row offset, pushed down to the database.
                Safe for the same reason as ``limit``.

        Returns:
            List of :class:`GeneratorDefinition` snapshots whose
            metadata matches every supplied entry.  An empty mapping is
            equivalent to ``None`` (no filter).
        """
        return await self._store.list(
            filter_metadata=filter_metadata,
            sort=sort,
            limit=limit,
            offset=offset,
        )

    async def count_definitions(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count persisted generator definitions matching the filter.

        Mirrors :meth:`list_definitions` parameter-for-parameter (minus
        ``sort``/``limit``/``offset``, which don't affect the count)
        and is equivalent to ``len(await self.list_definitions(...))``.

        Cost note:
            ``GeneratorRegistry`` uses single-write keying
            (``gen:{id}``) with no latest-pointer / snapshot divergence,
            so the database's row count IS the definition count.
            Routes through :meth:`AsyncKeyedRecordStore.count` which
            calls ``AsyncDatabase.count(query)``, letting backends with
            pushdown counts (``SELECT COUNT(*) WHERE ...``) avoid
            materializing rows.

        Args:
            filter_metadata: Optional equality filter over the
                ``metadata`` column.

        Returns:
            Number of registered generator definitions matching the
            filter.
        """
        return await self._store.count(filter_metadata=filter_metadata)

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
