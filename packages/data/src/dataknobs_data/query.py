from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union


class Operator(Enum):
    """Query operators for filtering."""

    EQ = "="  # Equal
    NEQ = "!="  # Not equal
    GT = ">"  # Greater than
    GTE = ">="  # Greater than or equal
    LT = "<"  # Less than
    LTE = "<="  # Less than or equal
    IN = "in"  # In list
    NOT_IN = "not_in"  # Not in list
    LIKE = "like"  # String pattern matching (SQL LIKE)
    REGEX = "regex"  # Regular expression matching
    EXISTS = "exists"  # Field exists
    NOT_EXISTS = "not_exists"  # Field does not exist
    BETWEEN = "between"  # Value between two bounds (inclusive)
    NOT_BETWEEN = "not_between"  # Value not between two bounds


class SortOrder(Enum):
    """Sort order for query results."""

    ASC = "asc"
    DESC = "desc"


@dataclass
class Filter:
    """Represents a filter condition."""

    field: str
    operator: Operator
    value: Any = None

    def matches(self, record_value: Any) -> bool:
        """Check if a record value matches this filter.
        
        Supports type-aware comparisons for ranges and special handling
        for datetime/date objects.
        """
        if self.operator == Operator.EXISTS:
            return record_value is not None
        elif self.operator == Operator.NOT_EXISTS:
            return record_value is None
        elif record_value is None:
            return False

        if self.operator == Operator.EQ:
            return record_value == self.value
        elif self.operator == Operator.NEQ:
            return record_value != self.value
        elif self.operator == Operator.GT:
            return self._compare_values(record_value, self.value, lambda a, b: a > b)
        elif self.operator == Operator.GTE:
            return self._compare_values(record_value, self.value, lambda a, b: a >= b)
        elif self.operator == Operator.LT:
            return self._compare_values(record_value, self.value, lambda a, b: a < b)
        elif self.operator == Operator.LTE:
            return self._compare_values(record_value, self.value, lambda a, b: a <= b)
        elif self.operator == Operator.IN:
            return record_value in self.value
        elif self.operator == Operator.NOT_IN:
            return record_value not in self.value
        elif self.operator == Operator.BETWEEN:
            if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                return False
            lower, upper = self.value
            return self._compare_values(record_value, lower, lambda a, b: a >= b) and \
                   self._compare_values(record_value, upper, lambda a, b: a <= b)
        elif self.operator == Operator.NOT_BETWEEN:
            if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                return True
            lower, upper = self.value
            return not (self._compare_values(record_value, lower, lambda a, b: a >= b) and \
                       self._compare_values(record_value, upper, lambda a, b: a <= b))
        elif self.operator == Operator.LIKE:
            if not isinstance(record_value, str):
                return False
            import re

            pattern = self.value.replace("%", ".*").replace("_", ".")
            return bool(re.match(f"^{pattern}$", record_value))
        elif self.operator == Operator.REGEX:
            if not isinstance(record_value, str):
                return False
            import re

            return bool(re.search(self.value, record_value))

        return False

    def _compare_values(self, a: Any, b: Any, comparator) -> bool:
        """Compare two values with type awareness.
        
        Handles special cases:
        - Datetime strings are parsed for comparison
        - Mixed numeric types are converted appropriately
        - String comparisons are case-sensitive
        """
        from datetime import date, datetime

        # Handle datetime/date comparisons
        if isinstance(a, str) and isinstance(b, (datetime, date)):
            try:
                a = datetime.fromisoformat(a.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                return False
        elif isinstance(b, str) and isinstance(a, (datetime, date)):
            try:
                b = datetime.fromisoformat(b.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                return False
        elif isinstance(a, str) and isinstance(b, str):
            # Check if both look like dates
            if "T" in a or "-" in a:
                try:
                    a = datetime.fromisoformat(a.replace("Z", "+00:00"))
                    b = datetime.fromisoformat(b.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass  # Keep as strings

        # Handle numeric comparisons
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return comparator(a, b)

        # Try direct comparison
        try:
            return comparator(a, b)
        except TypeError:
            # Types not comparable
            return False

    def to_dict(self) -> dict[str, Any]:
        """Convert filter to dictionary representation."""
        return {"field": self.field, "operator": self.operator.value, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Filter":
        """Create filter from dictionary representation."""
        return cls(
            field=data["field"], operator=Operator(data["operator"]), value=data.get("value")
        )


@dataclass
class SortSpec:
    """Represents a sort specification."""

    field: str
    order: SortOrder = SortOrder.ASC

    def to_dict(self) -> dict[str, str]:
        """Convert sort spec to dictionary representation."""
        return {"field": self.field, "order": self.order.value}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "SortSpec":
        """Create sort spec from dictionary representation."""
        return cls(field=data["field"], order=SortOrder(data.get("order", "asc")))


@dataclass
class Query:
    """Represents a database query with filters, sorting, and pagination."""

    filters: list[Filter] = field(default_factory=list)
    sort_specs: list[SortSpec] = field(default_factory=list)
    limit_value: int | None = None
    offset_value: int | None = None
    fields: list[str] | None = None  # Field projection

    @property
    def sort_property(self) -> list[SortSpec]:
        """Get sort specifications (backward compatibility)."""
        return self.sort_specs

    @property
    def limit_property(self) -> int | None:
        """Get limit value (backward compatibility)."""
        return self.limit_value

    @property
    def offset_property(self) -> int | None:
        """Get offset value (backward compatibility)."""
        return self.offset_value

    def filter(self, field: str, operator: str | Operator, value: Any = None) -> "Query":
        """Add a filter to the query (fluent interface).

        Args:
            field: The field name to filter on
            operator: The operator (string or Operator enum)
            value: The value to compare against

        Returns:
            Self for method chaining
        """
        if isinstance(operator, str):
            op_map = {
                "=": Operator.EQ,
                "==": Operator.EQ,
                "!=": Operator.NEQ,
                ">": Operator.GT,
                ">=": Operator.GTE,
                "<": Operator.LT,
                "<=": Operator.LTE,
                "in": Operator.IN,
                "IN": Operator.IN,
                "not_in": Operator.NOT_IN,
                "NOT IN": Operator.NOT_IN,
                "like": Operator.LIKE,
                "LIKE": Operator.LIKE,
                "regex": Operator.REGEX,
                "exists": Operator.EXISTS,
                "not_exists": Operator.NOT_EXISTS,
                "between": Operator.BETWEEN,
                "BETWEEN": Operator.BETWEEN,
                "not_between": Operator.NOT_BETWEEN,
                "NOT BETWEEN": Operator.NOT_BETWEEN,
            }
            operator = op_map.get(operator, Operator.EQ)

        self.filters.append(Filter(field=field, operator=operator, value=value))
        return self

    def sort_by(self, field: str, order: str | SortOrder = "asc") -> "Query":
        """Add a sort specification to the query (fluent interface).

        Args:
            field: The field name to sort by
            order: The sort order ("asc", "desc", or SortOrder enum)

        Returns:
            Self for method chaining
        """
        if isinstance(order, str):
            order = SortOrder.ASC if order.lower() == "asc" else SortOrder.DESC

        self.sort_specs.append(SortSpec(field=field, order=order))
        return self

    def sort(self, field: str, order: str | SortOrder = "asc") -> "Query":
        """Add sorting (fluent interface)."""
        return self.sort_by(field, order)

    def set_limit(self, limit: int) -> "Query":
        """Set the result limit (fluent interface).

        Args:
            limit: Maximum number of results

        Returns:
            Self for method chaining
        """
        self.limit_value = limit
        return self

    def limit(self, value: int) -> "Query":
        """Set limit (fluent interface)."""
        return self.set_limit(value)

    def set_offset(self, offset: int) -> "Query":
        """Set the result offset (fluent interface).

        Args:
            offset: Number of results to skip

        Returns:
            Self for method chaining
        """
        self.offset_value = offset
        return self

    def offset(self, value: int) -> "Query":
        """Set offset (fluent interface)."""
        return self.set_offset(value)

    def select(self, *fields: str) -> "Query":
        """Set field projection (fluent interface).

        Args:
            fields: Field names to include in results

        Returns:
            Self for method chaining
        """
        self.fields = list(fields) if fields else None
        return self

    def clear_filters(self) -> "Query":
        """Clear all filters (fluent interface)."""
        self.filters = []
        return self

    def clear_sort(self) -> "Query":
        """Clear all sort specifications (fluent interface)."""
        self.sort_specs = []
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert query to dictionary representation."""
        result = {
            "filters": [f.to_dict() for f in self.filters],
            "sort": [s.to_dict() for s in self.sort_specs],
        }
        if self.limit_value is not None:
            result["limit"] = self.limit_value
        if self.offset_value is not None:
            result["offset"] = self.offset_value
        if self.fields is not None:
            result["fields"] = self.fields
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Query":
        """Create query from dictionary representation."""
        query = cls()

        for filter_data in data.get("filters", []):
            query.filters.append(Filter.from_dict(filter_data))

        for sort_data in data.get("sort", []):
            query.sort_specs.append(SortSpec.from_dict(sort_data))

        query.limit_value = data.get("limit")
        query.offset_value = data.get("offset")
        query.fields = data.get("fields")

        return query

    def copy(self) -> "Query":
        """Create a copy of the query."""
        import copy

        return Query(
            filters=copy.deepcopy(self.filters),
            sort_specs=copy.deepcopy(self.sort_specs),
            limit_value=self.limit_value,
            offset_value=self.offset_value,
            fields=self.fields.copy() if self.fields else None,
        )

    def or_(self, *filters: Union[Filter, "Query"]) -> "ComplexQuery":
        """Create a ComplexQuery with OR logic.
        
        The current query's filters become an AND group, combined with OR conditions.
        Example: Query with filters [A, B] calling or_(C, D) creates: (A AND B) AND (C OR D)
        
        Args:
            filters: Filter objects or Query objects to OR together
            
        Returns:
            ComplexQuery with OR logic
        """
        from .query_logic import ComplexQuery, FilterCondition, LogicCondition, LogicOperator

        # Build OR conditions from the arguments
        or_conditions = []
        for item in filters:
            if isinstance(item, Filter):
                or_conditions.append(FilterCondition(item))
            elif isinstance(item, Query):
                if len(item.filters) == 1:
                    or_conditions.append(FilterCondition(item.filters[0]))
                elif item.filters:
                    and_cond = LogicCondition(operator=LogicOperator.AND)
                    for f in item.filters:
                        and_cond.conditions.append(FilterCondition(f))
                    or_conditions.append(and_cond)

        # Create the OR condition group
        or_group = None
        if or_conditions:
            if len(or_conditions) == 1:
                or_group = or_conditions[0]
            else:
                or_group = LogicCondition(
                    operator=LogicOperator.OR,
                    conditions=or_conditions
                )

        # Combine with existing filters (if any) using AND
        if self.filters:
            # Create AND condition for existing filters
            if len(self.filters) == 1:
                existing = FilterCondition(self.filters[0])
            else:
                existing = LogicCondition(operator=LogicOperator.AND)
                for f in self.filters:
                    existing.conditions.append(FilterCondition(f))

            # Combine existing AND new OR group with AND
            if or_group:
                root_condition = LogicCondition(
                    operator=LogicOperator.AND,
                    conditions=[existing, or_group]
                )
            else:
                root_condition = existing
        else:
            # No existing filters, just use OR group
            root_condition = or_group

        return ComplexQuery(
            condition=root_condition,
            sort_specs=self.sort_specs.copy(),
            limit_value=self.limit_value,
            offset_value=self.offset_value,
            fields=self.fields.copy() if self.fields else None
        )

    def and_(self, *filters: Union[Filter, "Query"]) -> "Query":
        """Add more filters with AND logic (convenience method).
        
        Args:
            filters: Filter objects or Query objects to AND together
            
        Returns:
            Self for chaining
        """
        for item in filters:
            if isinstance(item, Filter):
                self.filters.append(item)
            elif isinstance(item, Query):
                self.filters.extend(item.filters)
        return self

    def not_(self, filter: Filter) -> "ComplexQuery":
        """Create a ComplexQuery with NOT logic.
        
        Args:
            filter: Filter to negate
            
        Returns:
            ComplexQuery with NOT logic
        """
        from .query_logic import ComplexQuery, FilterCondition, LogicCondition, LogicOperator

        # Current filters as AND
        conditions = []
        if self.filters:
            if len(self.filters) == 1:
                conditions.append(FilterCondition(self.filters[0]))
            else:
                and_cond = LogicCondition(operator=LogicOperator.AND)
                for f in self.filters:
                    and_cond.conditions.append(FilterCondition(f))
                conditions.append(and_cond)

        # Add NOT condition
        not_cond = LogicCondition(
            operator=LogicOperator.NOT,
            conditions=[FilterCondition(filter)]
        )
        conditions.append(not_cond)

        # Create root condition
        if len(conditions) == 1:
            root_condition = conditions[0]
        else:
            root_condition = LogicCondition(
                operator=LogicOperator.AND,
                conditions=conditions
            )

        return ComplexQuery(
            condition=root_condition,
            sort_specs=self.sort_specs.copy(),
            limit_value=self.limit_value,
            offset_value=self.offset_value,
            fields=self.fields.copy() if self.fields else None
        )
