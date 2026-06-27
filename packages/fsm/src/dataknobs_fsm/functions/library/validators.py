"""Built-in validator functions for FSM.

This module provides commonly used validation functions that can be
referenced in FSM configurations.
"""

import inspect
import re
from collections.abc import Callable, Mapping
from typing import Any, Dict, List, Union

from pydantic import BaseModel, ValidationError

from dataknobs_fsm.functions.base import IValidationFunction, ValidationError as FSMValidationError


# Map a friendly schema ``type`` token to the Python type used for the
# ``isinstance`` check in :func:`build_record_validator`. Shared by every
# pattern that accepts a friendly dict validation schema (file-processing, ETL).
_VALIDATION_TYPE_MAP: Dict[str, type] = {
    "str": str, "string": str, "int": int, "integer": int,
    "float": float, "number": float, "bool": bool, "boolean": bool,
    "list": list, "array": list, "dict": dict, "object": dict,
}


def _friendly_schema_predicate(
    schema: Mapping[str, Any],
) -> Callable[..., bool]:
    """Build a ``(record, context) -> bool`` gate from a friendly dict schema.

    A record is valid iff it satisfies every field constraint. Supported
    per-field constraints: ``required``, ``type`` (mapped to a Python type for
    an ``isinstance`` check via :data:`_VALIDATION_TYPE_MAP`), ``min`` / ``max``
    (inclusive numeric bounds), and ``pattern`` (regex). A field whose
    constraint is the literal ``True`` is treated as simply required.

    Presence and value are independent: ``required`` (or the literal ``True``)
    governs whether an *absent* field rejects, while ``type`` / ``min`` /
    ``max`` / ``pattern`` apply only when the field is *present*. So
    ``{"score": {"min": 0}}`` means "if present, score must be >= 0" and an
    absent ``score`` passes; combine with ``"required": True`` to also demand
    presence. A present value that cannot satisfy a numeric bound (e.g. a
    string against ``min``) rejects the record rather than raising.
    """

    def validate_check(data: Dict[str, Any], context: Any = None) -> bool:
        for field_name, constraints in schema.items():
            if constraints is True:
                if field_name not in data:
                    return False
                continue
            if not isinstance(constraints, dict):
                continue
            # Presence is governed solely by ``required`` (or the literal
            # ``True`` shorthand above); the value constraints below apply only
            # when the field is present. So ``{"score": {"min": 0}}`` means
            # "if score is present it must be >= 0" — an absent optional field
            # passes, mark it ``required`` to also demand presence.
            if field_name not in data:
                if constraints.get("required"):
                    return False
                continue
            value = data[field_name]
            if "type" in constraints:
                expected = _VALIDATION_TYPE_MAP.get(constraints["type"])
                if expected is not None and not isinstance(value, expected):
                    return False
            # ``min`` / ``max`` are inclusive numeric bounds. A *present* value
            # that is not a real number cannot satisfy a numeric bound, so it is
            # invalid (the gate rejects it) — never a ``TypeError`` from
            # comparing e.g. ``"abc" >= 18``.
            if "min" in constraints and (
                not isinstance(value, (int, float))
                or value < constraints["min"]
            ):
                return False
            if "max" in constraints and (
                not isinstance(value, (int, float))
                or value > constraints["max"]
            ):
                return False
            if "pattern" in constraints and not re.match(
                constraints["pattern"], str(value)
            ):
                return False
        return True

    return validate_check


def build_gate_arcs(
    *,
    from_state: str,
    condition_name: str,
    pass_to: str,
    reject_to: str,
    pass_name: str,
    reject_name: str,
    priority: int = 10,
    resources: Mapping[str, str] | None = None,
) -> List[Dict[str, Any]]:
    """Build the two-arc shape that turns a state into a record gate.

    A higher-priority conditional ``pass`` arc routes records the registered
    ``condition_name`` accepts onward to ``pass_to``; an unconditional
    fall-through diverts the rest to ``reject_to``. The engine sorts available
    arcs by priority (higher first), so a passing record is routed
    deterministically without depending on arc declaration order.

    Shared by the file-processing and ETL gate patterns so the gate *shape*
    cannot drift between them — only the terminal names (``invalid`` /
    ``filtered_out``) and downstream accounting differ, and those are the
    parameters.

    ``resources`` (optional) is a ``{role: name}`` binding declared on the
    ``pass`` arc, so a resource-backed condition can resolve a reference
    resource from its :class:`FunctionContext`
    (``resource_for_role`` / ``require_resource``).

    Args:
        from_state: The gate state both arcs leave.
        condition_name: Registered name of the ``pass`` arc condition.
        pass_to: Target state for records the condition accepts.
        reject_to: Target (non-emitting terminal) for rejected records.
        pass_name / reject_name: Arc names for the pass / fall-through arcs.
        priority: Priority of the conditional pass arc (must beat the
            unconditional fall-through; default 10).
        resources: Optional ``{role: name}`` resource binding for the pass arc.

    Returns:
        A two-element list ``[pass_arc, reject_arc]`` of arc-config dicts.
    """
    pass_arc: Dict[str, Any] = {
        "from": from_state,
        "to": pass_to,
        "name": pass_name,
        "condition": {"type": "registered", "name": condition_name},
        "priority": priority,
    }
    if resources:
        pass_arc["resources"] = dict(resources)
    reject_arc: Dict[str, Any] = {
        "from": from_state,
        "to": reject_to,
        "name": reject_name,
    }
    return [pass_arc, reject_arc]


def _validation_function_predicate(
    validator: IValidationFunction,
) -> Callable[..., bool]:
    """Adapt an :class:`IValidationFunction` into a ``-> bool`` gate.

    The shipped library validators return ``True`` on success and raise
    :class:`FSMValidationError` on failure (the rich error message carries the
    per-field reason). A gate wants a boolean, so a raised validation error
    becomes ``False`` — the arc condition de-selects the ``valid`` arc and the
    record is diverted to the reject terminal.
    """

    def validate_check(data: Dict[str, Any], context: Any = None) -> bool:
        try:
            return bool(validator.validate(data))
        except FSMValidationError:
            return False

    return validate_check


def _callable_predicate(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Arity- and async-normalize a user predicate to the engine's call shape.

    The FSM engine always invokes an arc condition as ``fn(record, context)``.
    A consumer's predicate may be written as ``record -> bool`` or
    ``(record, context) -> bool``, and may be sync or async (an async predicate
    is what a resource-reading gate uses). The returned callable always accepts
    ``(record, context)`` and forwards the right number of arguments; it is a
    coroutine function iff ``fn`` is, so the engine's ``iscoroutinefunction``
    check routes it correctly.

    The context parameter must be positional (or have a default). Arity
    detection counts positional parameters only, so a predicate that declares
    ``context`` as a *required keyword-only* argument
    (``def fn(record, *, context): ...``) is called with the record alone and
    raises ``TypeError`` at evaluation time — write ``(record, context)`` or
    ``(record, context=None)`` instead.
    """
    try:
        params = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        ]
        wants_context = any(
            p.kind is inspect.Parameter.VAR_POSITIONAL for p in params
        ) or len(params) >= 2
    except (TypeError, ValueError):
        # Builtins / C-callables with no introspectable signature: be permissive
        # and pass only the record (the common ``record -> bool`` shape).
        wants_context = False

    if inspect.iscoroutinefunction(fn):
        async def async_check(data: Dict[str, Any], context: Any = None) -> bool:
            return bool(await (fn(data, context) if wants_context else fn(data)))

        return async_check

    def sync_check(data: Dict[str, Any], context: Any = None) -> bool:
        return bool(fn(data, context) if wants_context else fn(data))

    return sync_check


def build_record_validator(
    spec: Mapping[str, Any] | IValidationFunction | Callable[..., Any],
) -> Callable[..., Any]:
    """Normalize any supported validation spec to a record gate.

    Returns a callable the FSM engine can register as a ``validate`` arc
    condition: it is invoked as ``gate(record, context)`` and returns truthy
    when the record passes. Three spec forms are accepted so a consumer can
    pick the right tool — or roll their own:

    - a **friendly dict schema** (the config-authored, serializable default):
      ``{field: {required, type, min, max, pattern}}`` (a constraint of literal
      ``True`` means "present"). Shared with the file-processing pattern.
    - any library **:class:`IValidationFunction`** instance (the shipped
      validators, or a consumer subclass): its ``validate()`` raise contract is
      adapted to a boolean gate.
    - a plain **callable** predicate ``record -> bool`` or
      ``(record, context) -> bool`` (sync or async): used directly,
      arity-normalized to the engine's call shape.

    Args:
        spec: The validation specification (mapping / validator / callable).

    Returns:
        A ``(record, context) -> bool`` gate (a coroutine function when the
        supplied callable is async).

    Raises:
        TypeError: If ``spec`` is none of the supported forms.
    """
    if isinstance(spec, IValidationFunction):
        return _validation_function_predicate(spec)
    if isinstance(spec, Mapping):
        return _friendly_schema_predicate(spec)
    if callable(spec):
        return _callable_predicate(spec)
    raise TypeError(
        "validation spec must be a mapping (friendly schema), an "
        "IValidationFunction, or a callable predicate; got "
        f"{type(spec).__name__}"
    )


class RequiredFieldsValidator(IValidationFunction):
    """Validate that required fields are present in data."""

    def __init__(self, fields: List[str], allow_none: bool = False):
        """Initialize the validator.
        
        Args:
            fields: List of required field names.
            allow_none: Whether to allow None values for required fields.
        """
        self.fields = fields
        self.allow_none = allow_none

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that all required fields are present.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        if not isinstance(data, dict):
            raise FSMValidationError(f"Expected dict, got {type(data).__name__}")
        
        missing_fields = []
        none_fields = []
        
        for field in self.fields:
            if field not in data:
                missing_fields.append(field)
            elif not self.allow_none and data[field] is None:
                none_fields.append(field)
        
        if missing_fields:
            raise FSMValidationError(
                f"Missing required fields: {', '.join(missing_fields)}"
            )
        
        if none_fields:
            raise FSMValidationError(
                f"Fields cannot be None: {', '.join(none_fields)}"
            )

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        return {
            "required_fields": self.fields,
            "allow_none": self.allow_none
        }


class SchemaValidator(IValidationFunction):
    """Validate data against a Pydantic schema."""

    def __init__(self, schema: Union[type[BaseModel], Dict[str, Any]]):
        """Initialize the validator.
        
        Args:
            schema: Pydantic model class or schema dictionary.
        """
        if isinstance(schema, dict):
            # Create a dynamic Pydantic model from dictionary
            from pydantic import create_model
            self.schema = create_model("DynamicSchema", **schema)
        else:
            self.schema = schema

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate data against the schema.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        try:
            self.schema(**data)
            return True
        except ValidationError as e:
            errors = []
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                errors.append(f"{field_path}: {error['msg']}")
            
            raise FSMValidationError(
                f"Schema validation failed: {'; '.join(errors)}"
            ) from e
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        if hasattr(self.schema, 'model_json_schema'):
            return self.schema.model_json_schema()
        elif hasattr(self.schema, '__annotations__'):
            return dict(self.schema.__annotations__)
        else:
            return {"schema": str(self.schema)}


class RangeValidator(IValidationFunction):
    """Validate that numeric values are within specified ranges."""

    def __init__(
        self,
        field_ranges: Dict[str, Dict[str, Union[int, float]]],
    ):
        """Initialize the validator.
        
        Args:
            field_ranges: Dictionary mapping field names to range specifications.
                         Each range can have 'min', 'max', or both.
        """
        self.field_ranges = field_ranges

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that values are within specified ranges.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        errors = []
        
        for field, range_spec in self.field_ranges.items():
            if field not in data:
                continue
            
            value = data[field]
            if not isinstance(value, (int, float)):
                errors.append(f"{field}: Expected numeric value, got {type(value).__name__}")
                continue
            
            if "min" in range_spec and value < range_spec["min"]:
                errors.append(f"{field}: Value {value} is below minimum {range_spec['min']}")
            
            if "max" in range_spec and value > range_spec["max"]:
                errors.append(f"{field}: Value {value} is above maximum {range_spec['max']}")
        
        if errors:
            raise FSMValidationError("; ".join(errors))

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        return {
            "type": "range",
            "field_ranges": self.field_ranges
        }


class PatternValidator(IValidationFunction):
    """Validate that string values match specified patterns."""

    def __init__(
        self,
        field_patterns: Dict[str, str],
        flags: int = 0,
    ):
        """Initialize the validator.
        
        Args:
            field_patterns: Dictionary mapping field names to regex patterns.
            flags: Regex flags to apply (e.g., re.IGNORECASE).
        """
        self.field_patterns = {}
        for field, pattern in field_patterns.items():
            self.field_patterns[field] = re.compile(pattern, flags)

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that values match specified patterns.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        errors = []
        
        for field, pattern in self.field_patterns.items():
            if field not in data:
                continue
            
            value = data[field]
            if not isinstance(value, str):
                errors.append(f"{field}: Expected string value, got {type(value).__name__}")
                continue
            
            if not pattern.match(value):
                errors.append(f"{field}: Value '{value}' does not match pattern")

        if errors:
            raise FSMValidationError("; ".join(errors))

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        return {
            "type": "pattern",
            "field_patterns": {field: pattern.pattern for field, pattern in self.field_patterns.items()}
        }


class TypeValidator(IValidationFunction):
    """Validate that fields have expected types."""

    def __init__(
        self,
        field_types: Dict[str, Union[type, List[type]]],
        strict: bool = False,
    ):
        """Initialize the validator.
        
        Args:
            field_types: Dictionary mapping field names to expected types.
            strict: If True, reject extra fields not in field_types.
        """
        self.field_types = field_types
        self.strict = strict

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that fields have expected types.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        errors = []
        
        # Check field types
        for field, expected_type in self.field_types.items():
            if field not in data:
                continue
            
            value = data[field]
            if isinstance(expected_type, list):
                # Multiple allowed types
                if not any(isinstance(value, t) for t in expected_type):
                    type_names = ", ".join(t.__name__ for t in expected_type)
                    errors.append(
                        f"{field}: Expected one of [{type_names}], "
                        f"got {type(value).__name__}"
                    )
            else:
                # Single expected type
                if not isinstance(value, expected_type):
                    errors.append(
                        f"{field}: Expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
        
        # Check for extra fields if strict mode
        if self.strict:
            extra_fields = set(data.keys()) - set(self.field_types.keys())
            if extra_fields:
                errors.append(f"Unexpected fields: {', '.join(extra_fields)}")

        if errors:
            raise FSMValidationError("; ".join(errors))

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        field_type_names = {}
        for field, ftype in self.field_types.items():
            if isinstance(ftype, list):
                field_type_names[field] = [t.__name__ for t in ftype]
            else:
                field_type_names[field] = ftype.__name__
        return {
            "type": "type_check",
            "field_types": field_type_names,
            "strict": self.strict
        }


class LengthValidator(IValidationFunction):
    """Validate that collections have expected lengths."""

    def __init__(
        self,
        field_lengths: Dict[str, Dict[str, int]],
    ):
        """Initialize the validator.
        
        Args:
            field_lengths: Dictionary mapping field names to length specifications.
                          Each spec can have 'min', 'max', or 'exact'.
        """
        self.field_lengths = field_lengths

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that collections have expected lengths.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        errors = []
        
        for field, length_spec in self.field_lengths.items():
            if field not in data:
                continue
            
            value = data[field]
            if not hasattr(value, "__len__"):
                errors.append(f"{field}: Value does not have a length")
                continue
            
            length = len(value)
            
            if "exact" in length_spec and length != length_spec["exact"]:
                errors.append(
                    f"{field}: Length {length} does not match expected {length_spec['exact']}"
                )
            
            if "min" in length_spec and length < length_spec["min"]:
                errors.append(f"{field}: Length {length} is below minimum {length_spec['min']}")
            
            if "max" in length_spec and length > length_spec["max"]:
                errors.append(f"{field}: Length {length} is above maximum {length_spec['max']}")

        if errors:
            raise FSMValidationError("; ".join(errors))

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        return {
            "type": "length",
            "field_lengths": self.field_lengths
        }


class UniqueValidator(IValidationFunction):
    """Validate that values in collections are unique."""

    def __init__(
        self,
        fields: List[str],
        key: str | None = None,
    ):
        """Initialize the validator.
        
        Args:
            fields: List of field names to check for uniqueness.
            key: Optional key to extract from collection items for uniqueness check.
        """
        self.fields = fields
        self.key = key

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that values are unique.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        errors = []
        
        for field in self.fields:
            if field not in data:
                continue
            
            value = data[field]
            if not isinstance(value, (list, tuple, set)):
                errors.append(f"{field}: Expected collection, got {type(value).__name__}")
                continue
            
            if self.key:
                # Extract values using key
                try:
                    values = [item[self.key] if isinstance(item, dict) else getattr(item, self.key)
                             for item in value]
                except (KeyError, AttributeError) as e:
                    errors.append(f"{field}: Cannot extract key '{self.key}': {e}")
                    continue
            else:
                values = list(value)
            
            # Check for duplicates
            seen = set()
            duplicates = set()
            for v in values:
                if v in seen:
                    duplicates.add(str(v))
                seen.add(v)
            
            if duplicates:
                errors.append(f"{field}: Duplicate values found: {', '.join(duplicates)}")

        if errors:
            raise FSMValidationError("; ".join(errors))

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        return {
            "type": "unique",
            "fields": self.fields,
            "key": self.key
        }


class DependencyValidator(IValidationFunction):
    """Validate field dependencies (if field A exists, field B must also exist)."""

    def __init__(
        self,
        dependencies: Dict[str, Union[str, List[str]]],
    ):
        """Initialize the validator.
        
        Args:
            dependencies: Dictionary mapping field names to their dependencies.
        """
        self.dependencies = dependencies

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate field dependencies.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        errors = []
        
        for field, deps in self.dependencies.items():
            if field not in data:
                continue
            
            deps_list = deps if isinstance(deps, list) else [deps]
            
            missing_deps = [dep for dep in deps_list if dep not in data]
            
            if missing_deps:
                errors.append(
                    f"Field '{field}' requires: {', '.join(missing_deps)}"
                )

        if errors:
            raise FSMValidationError("; ".join(errors))

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        return {
            "type": "dependency",
            "dependencies": self.dependencies
        }


class CompositeValidator(IValidationFunction):
    """Compose multiple validators into a single validator."""

    def __init__(
        self,
        validators: List[IValidationFunction],
        stop_on_first_error: bool = False,
    ):
        """Initialize the composite validator.
        
        Args:
            validators: List of validators to apply.
            stop_on_first_error: If True, stop at first validation error.
        """
        self.validators = validators
        self.stop_on_first_error = stop_on_first_error

    def validate(self, data: Dict[str, Any]) -> bool:
        """Apply all validators to the data.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if all validators pass.
            
        Raises:
            FSMValidationError: If any validation fails.
        """
        errors = []
        
        for validator in self.validators:
            try:
                validator.validate(data)
            except FSMValidationError as e:
                if self.stop_on_first_error:
                    raise
                errors.append(str(e))
        
        if errors:
            raise FSMValidationError("; ".join(errors))
        
        return True


# Convenience functions for creating validators
def required_fields(*fields: str, allow_none: bool = False) -> RequiredFieldsValidator:
    """Create a RequiredFieldsValidator."""
    return RequiredFieldsValidator(list(fields), allow_none)


def schema(model: Union[type[BaseModel], Dict[str, Any]]) -> SchemaValidator:
    """Create a SchemaValidator."""
    return SchemaValidator(model)


def range_check(**field_ranges: Dict[str, Union[int, float]]) -> RangeValidator:
    """Create a RangeValidator."""
    return RangeValidator(field_ranges)


def pattern(**field_patterns: str) -> PatternValidator:
    """Create a PatternValidator."""
    return PatternValidator(field_patterns)


def type_check(**field_types: Union[type, List[type]]) -> TypeValidator:
    """Create a TypeValidator."""
    return TypeValidator(field_types)


def length(**field_lengths: Dict[str, int]) -> LengthValidator:
    """Create a LengthValidator."""
    return LengthValidator(field_lengths)


def unique(*fields: str, key: str | None = None) -> UniqueValidator:
    """Create a UniqueValidator."""
    return UniqueValidator(list(fields), key)


def depends_on(**dependencies: Union[str, List[str]]) -> DependencyValidator:
    """Create a DependencyValidator."""
    return DependencyValidator(dependencies)
