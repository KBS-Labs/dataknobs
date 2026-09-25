# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Database schema definitions for field structures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dataknobs_common.exceptions import ValidationError

from .fields import FieldType

#: The keys a field declaration takes, in either spelling. ``dimensions`` and
#: ``source_field`` are the vector shorthands, and ``enum`` the allowed-values
#: one that :class:`~dataknobs_data.sources.database.DatabaseSource` reads; all
#: three fold into ``metadata``. A door
#: that uses less of a declaration narrows this set (``keys=`` on
#: :func:`read_field_declarations`), so that a key it would discard is refused
#: rather than loaded.
FIELD_KEYS: frozenset[str] = frozenset(
    {"name", "type", "required", "default", "metadata", "dimensions", "source_field", "enum"}
)

#: The keys that fold into a field's ``metadata`` under their own name, where an
#: explicit ``metadata`` entry wins.
_METADATA_SHORTHANDS: tuple[str, ...] = ("dimensions", "source_field", "enum")

#: The keys a schema declaration takes at its top level.
SCHEMA_KEYS: frozenset[str] = frozenset({"fields", "metadata"})


@dataclass
class FieldSchema:
    """Schema definition for a field without actual data.

    Defines the structure and constraints for a field in a database schema.
    Used for validation, type checking, and backend schema generation.

    Attributes:
        name: Field name
        type: Field data type
        metadata: Additional field metadata
        required: Whether the field is required
        default: Default value if field is missing

    Example:
        ```python
        from dataknobs_data.schema import FieldSchema
        from dataknobs_data.fields import FieldType

        # Simple field schema
        name_schema = FieldSchema(name="name", type=FieldType.STRING, required=True)

        # Vector field schema with metadata
        embedding_schema = FieldSchema(
            name="embedding",
            type=FieldType.VECTOR,
            metadata={"dimensions": 384, "source_field": "content"},
            required=False
        )

        # Check if vector field
        is_vector = embedding_schema.is_vector_field()  # True
        dims = embedding_schema.get_dimensions()  # 384
        ```
    """

    name: str
    type: FieldType
    metadata: dict[str, Any] = field(default_factory=dict)
    required: bool = False
    default: Any = None

    def is_vector_field(self) -> bool:
        """Check if this is a vector field.

        Returns:
            True if the field type is VECTOR or SPARSE_VECTOR

        Example:
            ```python
            vector_schema = FieldSchema(name="embedding", type=FieldType.VECTOR)
            print(vector_schema.is_vector_field())  # True

            text_schema = FieldSchema(name="content", type=FieldType.TEXT)
            print(text_schema.is_vector_field())  # False
            ```
        """
        return self.type in (FieldType.VECTOR, FieldType.SPARSE_VECTOR)

    def get_dimensions(self) -> int | None:
        """Get vector dimensions if this is a vector field."""
        if self.is_vector_field():
            return self.metadata.get("dimensions")
        return None

    def get_source_field(self) -> str | None:
        """Get source field if this is a derived vector field."""
        if self.is_vector_field():
            return self.metadata.get("source_field")
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type.value,
            "metadata": self.metadata,
            "required": self.required,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldSchema:
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            type=FieldType(data["type"]),
            metadata=data.get("metadata", {}),
            required=data.get("required", False),
            default=data.get("default"),
        )


@dataclass
class DatabaseSchema:
    """Schema definition for a database.

    Defines the structure of a database by specifying field schemas. Used for
    validation, type checking, and ensuring consistent data structure across records.

    Attributes:
        fields: Dictionary mapping field names to FieldSchema objects
        metadata: Optional schema-level metadata

    Example:
        ```python
        from dataknobs_data.schema import DatabaseSchema, FieldSchema
        from dataknobs_data.fields import FieldType

        # Create schema using .create() method
        schema = DatabaseSchema.create(
            name=FieldType.STRING,
            age=FieldType.INTEGER,
            email=FieldType.STRING
        )

        # With vector field and metadata
        schema = DatabaseSchema.create(
            content=FieldType.TEXT,
            embedding=(FieldType.VECTOR, {
                "dimensions": 384,
                "source_field": "content"
            })
        )

        # Add fields after creation
        schema.add_field(FieldSchema(
            name="created_at",
            type=FieldType.DATETIME,
            required=True
        ))

        # Get field schemas
        content_schema = schema.get_field("content")
        all_field_names = schema.get_field_names()
        ```
    """

    fields: dict[str, FieldSchema] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls, **field_definitions: FieldType | tuple[FieldType, dict[str, Any]]
    ) -> DatabaseSchema:
        """Create a schema from keyword arguments.

        Args:
            **field_definitions: Field definitions where each key is a field name and
                each value is either a FieldType or a tuple of (FieldType, options_dict)

        Returns:
            A new DatabaseSchema instance

        The options in a tuple are a field declaration without its ``type``
        (which is the tuple's first element), read by
        :func:`read_field_declarations`: the same keys, the same rules, and an
        explicit ``metadata`` entry winning over the ``dimensions`` /
        ``source_field`` shorthand. The caller's ``metadata`` is copied, not
        written into.

        Raises:
            ValueError: When a definition is neither a ``FieldType`` nor a
                ``(FieldType, options)`` tuple.
            ValidationError: When a tuple's options are ones a field
                declaration refuses -- an unknown key, a ``type``, or a
                ``required`` that is not a boolean.

        Example:
            ```python
            # Simple field types
            schema = DatabaseSchema.create(
                name=FieldType.STRING,
                age=FieldType.INTEGER
            )

            # With field options
            schema = DatabaseSchema.create(
                content=FieldType.TEXT,
                embedding=(FieldType.VECTOR, {"dimensions": 384, "source_field": "content"}),
                score=(FieldType.FLOAT, {"required": True, "default": 0.0})
            )
            ```
        """
        declarations: dict[str, FieldType | Mapping[str, Any]] = {}
        for name, definition in field_definitions.items():
            if isinstance(definition, FieldType):
                declarations[name] = definition
            elif (
                isinstance(definition, tuple)
                and len(definition) == 2
                and isinstance(definition[0], FieldType)
                and isinstance(definition[1], Mapping)
            ):
                field_type, options = definition
                if "type" in options:
                    raise ValidationError(
                        f"field {name!r} declares a `type` among its options; a "
                        f"`(FieldType, options)` tuple's type is its first element",
                        context={"field": name},
                    )
                declarations[name] = {**options, "type": field_type}
            else:
                raise ValueError(f"Invalid field definition for {name}: {definition}")
        return cls(fields=read_field_declarations(declarations))

    def add_field(self, field_schema: FieldSchema) -> DatabaseSchema:
        """Add a field to the schema.

        Returns self for chaining.
        """
        self.fields[field_schema.name] = field_schema
        return self

    def add_text_field(self, name: str, required: bool = False) -> DatabaseSchema:
        """Add a text field to the schema."""
        return self.add_field(FieldSchema(name=name, type=FieldType.TEXT, required=required))

    def add_vector_field(
        self, name: str, dimensions: int, source_field: str | None = None, required: bool = False
    ) -> DatabaseSchema:
        """Add a vector field to the schema."""
        return self.add_field(
            FieldSchema(
                name=name,
                type=FieldType.VECTOR,
                metadata={"dimensions": dimensions, "source_field": source_field},
                required=required,
            )
        )

    def remove_field(self, name: str) -> bool:
        """Remove a field from the schema."""
        if name in self.fields:
            del self.fields[name]
            return True
        return False

    def get_vector_fields(self) -> dict[str, FieldSchema]:
        """Get all vector fields in the schema."""
        return {name: field for name, field in self.fields.items() if field.is_vector_field()}

    def get_source_fields(self) -> dict[str, list[str]]:
        """Get mapping of source fields to their dependent vector fields."""
        source_map: dict[str, list[str]] = {}
        for name, field_obj in self.fields.items():
            if field_obj.is_vector_field():
                source = field_obj.get_source_field()
                if source:
                    if source not in source_map:
                        source_map[source] = []
                    source_map[source].append(name)
        return source_map

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "fields": {name: f.to_dict() for name, f in self.fields.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        origin: str | None = None,
        context: Mapping[str, Any] | None = None,
        keys: frozenset[str] = FIELD_KEYS,
    ) -> DatabaseSchema:
        """Create from dictionary representation.

        A schema declaration takes ``fields:`` and ``metadata:`` and nothing
        else. ``fields:`` is read by :func:`read_field_declarations`, in either
        of its two spellings:

        Examples:
            # A mapping: a type name, or a field mapping
            {"fields": {"content": "text", "score": {"type": "float", "required": True}}}

            # A list of rows, which is how an ontology binding writes it
            {"fields": [{"name": "content", "type": "text"}, {"name": "score", "type": "float"}]}

            # Vector fields
            {"fields": {"embedding": {"type": "vector", "dimensions": 384}}}

        An empty mapping is an empty schema, and ``fields: null`` or
        ``metadata: null`` is that key left out. A declaration that says something
        this reader cannot read is refused rather than read as no schema: a
        column written at the top level instead of under ``fields:`` would
        otherwise declare nothing, and every check keyed on the schema would
        then pass over nothing.

        Args:
            data: The declaration.
            origin: Where it came from (``"source 'courses'"``), prefixed to
                every refusal, the top-level ones included.
            context: Carried into every refusal's ``context``.
            keys: The keys a field takes through this door, as for
                :func:`read_field_declarations`.

        Raises:
            ValidationError: When ``data`` is not a mapping, carries a key other
                than ``fields`` and ``metadata``, or declares a field
                :func:`read_field_declarations` refuses.
        """
        prefix = f"{origin}: " if origin else ""
        base: dict[str, Any] = dict(context or {})
        if not isinstance(data, Mapping):
            raise ValidationError(
                f"{prefix}a schema declaration is a mapping, got {type(data).__name__}",
                context={**base, "got": type(data).__name__},
            )
        unknown = sorted(str(key) for key in data if key not in SCHEMA_KEYS)
        if unknown:
            raise ValidationError(
                f"{prefix}a schema declaration takes `fields:` and `metadata:` and nothing "
                f"else, and this one declares {unknown}. Columns go under `fields:` -- "
                f"`{{fields: {{<column>: <type>}}}}` or "
                f"`{{fields: [{{name: <column>, type: <type>}}]}}`",
                context={**base, "keys": unknown},
            )
        # An explicit `null` is the key left out, as it is for a field's keys.
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValidationError(
                f"{prefix}a schema's `metadata:` is a mapping, got {type(metadata).__name__}",
                context={**base, "got": type(metadata).__name__},
            )
        return cls(
            fields=read_field_declarations(
                data.get("fields") or {}, origin=origin, context=context, keys=keys
            ),
            metadata=dict(metadata),
        )


def read_field_declarations(
    declared: Any,
    *,
    origin: str | None = None,
    context: Mapping[str, Any] | None = None,
    keys: frozenset[str] = FIELD_KEYS,
) -> dict[str, FieldSchema]:
    """Read field declarations, in either spelling, or refuse them by name.

    The one reader behind every door a declaration comes through --
    :meth:`DatabaseSchema.from_dict`, a database config's ``schema:``, and an
    ontology binding's ``schema:`` rows -- so the two spellings cannot be read
    two ways.

    - **A mapping** of ``{<column>: <type name> | <field mapping>}``.
    - **A sequence of rows**, each a field mapping that carries its ``name``.

    A field mapping takes the keys in ``keys`` (by default :data:`FIELD_KEYS`).
    ``type`` defaults to ``string`` and is a type name in any case (``String``
    is ``string``); ``required`` is a boolean; ``enum`` is a list of the values
    a field allows, whichever of ``enum:`` and ``metadata.enum`` it is written
    as; ``dimensions``, ``source_field`` and ``enum`` fold into
    ``metadata``, where an explicit ``metadata`` entry wins. A mapping entry may repeat its ``name`` (which is what
    :meth:`DatabaseSchema.to_dict` writes) and must then agree with its key. A
    key a field takes, given an explicit ``null``, reads as that key left out
    -- YAML's ``type:`` with no value is ``string``, as ``type`` left out is. A
    field given as ``null`` is not a key left out, and is refused.

    Args:
        declared: The declarations: a mapping or a sequence of rows.
        origin: Where the declaration came from (``"binding 'catalog'"``),
            prefixed to every message, so a caller that knows says so without
            catching and re-raising.
        context: Carried into every refusal's ``context`` beside what the
            refusal adds, in the caller's own keys (an ontology binding's
            ``source_id``).
        keys: The keys a field takes through this door: :data:`FIELD_KEYS`,
            or a subset of it for a door that reads less of a declaration (an
            ontology binding reads a column's name and type), so that a key it
            would load and discard is refused instead.

    Returns:
        The declared fields, by name, in declaration order.

    Raises:
        ValidationError: When ``declared`` is neither a mapping nor a sequence
            of rows, or a field is unnamed, repeated, of an unknown type, or
            carries a key a field does not take.
        ValueError: When ``keys`` names a key this reader does not read, which
            is a caller's error rather than a declaration's.
    """
    unreadable = sorted(keys - FIELD_KEYS)
    if unreadable:
        raise ValueError(
            f"read_field_declarations reads {sorted(FIELD_KEYS)}; keys={unreadable} "
            f"would be admitted and then discarded"
        )
    prefix = f"{origin}: " if origin else ""
    base: dict[str, Any] = dict(context or {})

    entries: list[tuple[str | None, Any]]
    if isinstance(declared, Mapping):
        for name in declared:
            # Checked before anything reads the key as a name: a `str()` here
            # would turn `{5: integer}` into a field called "5", which the row
            # spelling refuses.
            if not isinstance(name, str) or not name:
                raise ValidationError(
                    f"{prefix}a field's name is a non-empty string, got {name!r}",
                    context={**base, "name": name},
                )
        entries = list(declared.items())
    elif isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
        entries = [(None, row) for row in declared]
    else:
        raise ValidationError(
            f"{prefix}fields are declared as a mapping of `{{<column>: <type>}}` or a list "
            f"of `{{name: <column>, type: <type>}}` rows, got {type(declared).__name__}",
            context={**base, "got": type(declared).__name__},
        )

    fields: dict[str, FieldSchema] = {}
    for key, value in entries:
        # A bare type name is the mapping spelling's shorthand for `{type: <name>}`.
        declaration = (
            {"type": value} if key is not None and isinstance(value, (str, FieldType)) else value
        )
        if not isinstance(declaration, Mapping):
            label = f"field {key!r}" if key is not None else f"row {value!r}"
            raise ValidationError(
                f"{prefix}{label} is not a field declaration: a field is a type name "
                f"(`string`) or a mapping (`{{name: <column>, type: <type>}}`)",
                context={**base, "field": key},
            )
        declaration = _without_nulls(declaration, keys)
        name = _declared_name(key, declaration, prefix=prefix, context=base)
        if name in fields:
            raise ValidationError(
                f"{prefix}field {name!r} is declared twice", context={**base, "field": name}
            )
        fields[name] = _field_schema(name, declaration, keys=keys, prefix=prefix, context=base)
    return fields


def _without_nulls(declaration: Mapping[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    """The declaration with each taken key given as ``null`` read as left out.

    A key outside ``keys`` keeps its ``null``, so a misspelt ``requird:`` with
    no value is still refused as the unknown key it is.
    """
    return {key: value for key, value in declaration.items() if not (value is None and key in keys)}


def _declared_name(
    key: str | None, value: Mapping[str, Any], *, prefix: str, context: dict[str, Any]
) -> str:
    """A field's name: the mapping key, or the row's ``name``, and never both disagreeing."""
    if key is None:
        if "name" not in value:
            raise ValidationError(
                f"{prefix}every field row names its column -- "
                f"`{{name: <column>, type: <type>}}` -- and this one is {dict(value)!r}",
                context={**context, "row": dict(value)},
            )
        name = value["name"]
    else:
        name = value.get("name", key)
        if name != key:
            raise ValidationError(
                f"{prefix}field {key!r} names itself {name!r}; a mapping entry's `name` "
                f"is its key or absent",
                context={**context, "field": key, "name": name},
            )
    if not isinstance(name, str) or not name:
        raise ValidationError(
            f"{prefix}a field's name is a non-empty string, got {name!r}",
            context={**context, "name": name},
        )
    return name


def _field_schema(
    name: str,
    value: Mapping[str, Any],
    *,
    keys: frozenset[str],
    prefix: str,
    context: dict[str, Any],
) -> FieldSchema:
    """One field mapping, whose name is already read, as a :class:`FieldSchema`."""
    field_context = {**context, "field": name}
    unknown = sorted(str(key) for key in value if key not in keys)
    if unknown:
        raise ValidationError(
            f"{prefix}field {name!r} declares {unknown}, which a field does not take "
            f"here. A field here takes {sorted(keys)}",
            context={**field_context, "keys": unknown},
        )

    declared_type = value.get("type", FieldType.STRING)
    try:
        # Every type's value is its member name lowercased, so folding the case
        # can only ever find the member the name spells.
        field_type = (
            declared_type
            if isinstance(declared_type, FieldType)
            else FieldType(str(declared_type).lower())
        )
    except ValueError as exc:
        raise ValidationError(
            f"{prefix}field {name!r} declares type {declared_type!r}, which is not a "
            f"field type. Field types: {sorted(member.value for member in FieldType)}",
            context={**field_context, "type": declared_type},
        ) from exc

    required = value.get("required", False)
    if not isinstance(required, bool):
        # Checked for being a boolean rather than for truthiness: `required: "no"`
        # is truthy, so the value that most obviously means *off* turned it on.
        raise ValidationError(
            f"{prefix}field {name!r} declares `required: {required!r}`; it is `true` or `false`",
            context={**field_context, "required": required},
        )

    declared_metadata = value.get("metadata", {})
    if not isinstance(declared_metadata, Mapping):
        raise ValidationError(
            f"{prefix}field {name!r} declares `metadata:` as "
            f"{type(declared_metadata).__name__}; it is a mapping",
            context=field_context,
        )
    metadata: dict[str, Any] = {}
    for shorthand in _METADATA_SHORTHANDS:
        if shorthand in value:
            metadata[shorthand] = value[shorthand]
    metadata.update(declared_metadata)

    # Checked after the merge, because the value checked has to be the one that
    # wins: an explicit `metadata.enum` takes precedence over the shorthand.
    enum = metadata.get("enum")
    if enum is not None and not isinstance(enum, (list, tuple)):
        # A string is iterable, so it would reach a filter schema as one
        # allowed value per letter.
        spelled = "metadata.enum" if "enum" in declared_metadata else "enum"
        raise ValidationError(
            f"{prefix}field {name!r} declares `{spelled}: {enum!r}`; it is a list of the "
            f"values the field allows",
            context={**field_context, "enum": enum},
        )

    return FieldSchema(
        name=name,
        type=field_type,
        metadata=metadata,
        required=required,
        default=value.get("default"),
    )
