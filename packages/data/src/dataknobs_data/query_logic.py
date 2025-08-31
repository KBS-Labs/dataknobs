"""Boolean logic support for complex queries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from .query import Filter, Operator, VectorQuery

if TYPE_CHECKING:
    import numpy as np

    from .query import Query
    from .vector.types import DistanceMetric


class LogicOperator(Enum):
    """Logical operators for combining conditions."""
    AND = "and"
    OR = "or"
    NOT = "not"


class Condition(ABC):
    """Abstract base class for query conditions."""

    @abstractmethod
    def matches(self, record: Any) -> bool:
        """Check if a record matches this condition."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert condition to dictionary representation."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Condition:
        """Create condition from dictionary representation."""
        pass


@dataclass
class FilterCondition(Condition):
    """A single filter condition."""
    filter: Filter

    def matches(self, record: Any) -> bool:
        """Check if a record matches this filter."""
        from .records import Record

        if isinstance(record, Record):
            value = record.get_value(self.filter.field)
        elif isinstance(record, dict):
            # Support nested field access for dicts
            value = record
            for part in self.filter.field.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
        else:
            value = getattr(record, self.filter.field, None)

        return self.filter.matches(value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "filter",
            "filter": self.filter.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilterCondition:
        """Create from dictionary representation."""
        return cls(filter=Filter.from_dict(data["filter"]))


@dataclass
class LogicCondition(Condition):
    """A logical combination of conditions."""
    operator: LogicOperator
    conditions: list[Condition] = field(default_factory=list)

    def matches(self, record: Any) -> bool:
        """Check if a record matches this logical condition."""
        if self.operator == LogicOperator.AND:
            # All conditions must match
            return all(cond.matches(record) for cond in self.conditions)
        elif self.operator == LogicOperator.OR:
            # At least one condition must match
            return any(cond.matches(record) for cond in self.conditions)
        elif self.operator == LogicOperator.NOT:
            # No conditions should match (or negate single condition)
            if len(self.conditions) == 1:
                return not self.conditions[0].matches(record)
            else:
                # NOT with multiple conditions = none should match
                return not any(cond.matches(record) for cond in self.conditions)
        else:
            # This should never be reached as all operators are handled above
            raise ValueError(f"Unknown logical operator: {self.operator}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "logic",
            "operator": self.operator.value,
            "conditions": [cond.to_dict() for cond in self.conditions]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogicCondition:
        """Create from dictionary representation."""
        conditions: list[Condition] = []
        for cond_data in data.get("conditions", []):
            if cond_data["type"] == "filter":
                conditions.append(FilterCondition.from_dict(cond_data))
            elif cond_data["type"] == "logic":
                conditions.append(LogicCondition.from_dict(cond_data))

        return cls(
            operator=LogicOperator(data["operator"]),
            conditions=conditions
        )


def condition_from_dict(data: dict[str, Any]) -> Condition:
    """Factory function to create condition from dictionary."""
    if data["type"] == "filter":
        return FilterCondition.from_dict(data)
    elif data["type"] == "logic":
        return LogicCondition.from_dict(data)
    else:
        raise ValueError(f"Unknown condition type: {data['type']}")


class QueryBuilder:
    """Builder for complex queries with boolean logic."""

    def __init__(self):
        """Initialize empty query builder."""
        self.root_condition = None
        self.sort_specs = []
        self.limit_value = None
        self.offset_value = None
        self.fields = None
        self.vector_query = None

    def where(self, field: str, operator: str | Operator, value: Any = None) -> QueryBuilder:
        """Add a filter condition (defaults to AND with existing conditions)."""
        op = Operator(operator) if isinstance(operator, str) else operator
        filter_cond = FilterCondition(Filter(field, op, value))

        if self.root_condition is None:
            self.root_condition = filter_cond
        elif isinstance(self.root_condition, LogicCondition) and self.root_condition.operator == LogicOperator.AND:
            self.root_condition.conditions.append(filter_cond)
        else:
            # Wrap existing condition in AND
            self.root_condition = LogicCondition(
                operator=LogicOperator.AND,
                conditions=[self.root_condition, filter_cond]
            )

        return self

    def and_(self, *conditions: QueryBuilder | Filter | Condition) -> QueryBuilder:
        """Add AND conditions."""
        logic_cond = LogicCondition(operator=LogicOperator.AND)

        for cond in conditions:
            if isinstance(cond, QueryBuilder):
                if cond.root_condition:
                    logic_cond.conditions.append(cond.root_condition)
            elif isinstance(cond, Filter):
                logic_cond.conditions.append(FilterCondition(cond))
            elif isinstance(cond, Condition):
                logic_cond.conditions.append(cond)

        if self.root_condition is None:
            self.root_condition = logic_cond
        elif isinstance(self.root_condition, LogicCondition) and self.root_condition.operator == LogicOperator.AND:
            self.root_condition.conditions.extend(logic_cond.conditions)
        else:
            self.root_condition = LogicCondition(
                operator=LogicOperator.AND,
                conditions=[self.root_condition, logic_cond]
            )

        return self

    def or_(self, *conditions: QueryBuilder | Filter | Condition) -> QueryBuilder:
        """Add OR conditions."""
        logic_cond = LogicCondition(operator=LogicOperator.OR)

        for cond in conditions:
            if isinstance(cond, QueryBuilder):
                if cond.root_condition:
                    logic_cond.conditions.append(cond.root_condition)
            elif isinstance(cond, Filter):
                logic_cond.conditions.append(FilterCondition(cond))
            elif isinstance(cond, Condition):
                logic_cond.conditions.append(cond)

        if self.root_condition is None:
            self.root_condition = logic_cond
        else:
            # Always wrap in OR at top level
            if isinstance(self.root_condition, LogicCondition) and self.root_condition.operator == LogicOperator.OR:
                self.root_condition.conditions.extend(logic_cond.conditions)
            else:
                self.root_condition = LogicCondition(
                    operator=LogicOperator.OR,
                    conditions=[self.root_condition] + logic_cond.conditions
                )

        return self

    def not_(self, condition: QueryBuilder | Filter | Condition) -> QueryBuilder:
        """Add NOT condition."""
        if isinstance(condition, QueryBuilder):
            not_cond = LogicCondition(
                operator=LogicOperator.NOT,
                conditions=[condition.root_condition] if condition.root_condition else []
            )
        elif isinstance(condition, Filter):
            not_cond = LogicCondition(
                operator=LogicOperator.NOT,
                conditions=[FilterCondition(condition)]
            )
        else:
            not_cond = LogicCondition(
                operator=LogicOperator.NOT,
                conditions=[condition]
            )

        if self.root_condition is None:
            self.root_condition = not_cond
        elif isinstance(self.root_condition, LogicCondition) and self.root_condition.operator == LogicOperator.AND:
            self.root_condition.conditions.append(not_cond)
        else:
            self.root_condition = LogicCondition(
                operator=LogicOperator.AND,
                conditions=[self.root_condition, not_cond]
            )

        return self

    def sort_by(self, field: str, order: str = "asc") -> QueryBuilder:
        """Add sort specification."""
        from .query import SortOrder, SortSpec

        sort_order = SortOrder.ASC if order.lower() == "asc" else SortOrder.DESC
        self.sort_specs.append(SortSpec(field=field, order=sort_order))
        return self

    def limit(self, value: int) -> QueryBuilder:
        """Set result limit."""
        self.limit_value = value
        return self

    def offset(self, value: int) -> QueryBuilder:
        """Set result offset."""
        self.offset_value = value
        return self

    def select(self, *fields: str) -> QueryBuilder:
        """Set field projection."""
        self.fields = list(fields) if fields else None
        return self

    def similar_to(
        self,
        vector: np.ndarray | list[float],
        field: str = "embedding",
        k: int = 10,
        metric: DistanceMetric | str = "cosine",
        include_source: bool = True,
        score_threshold: float | None = None,
    ) -> QueryBuilder:
        """Add vector similarity search."""
        self.vector_query = VectorQuery(
            vector=vector,
            field_name=field,
            k=k,
            metric=metric,
            include_source=include_source,
            score_threshold=score_threshold,
        )
        # If limit is not set, use k as the limit
        if self.limit_value is None:
            self.limit_value = k
        return self

    def build(self) -> ComplexQuery:
        """Build the final query."""
        return ComplexQuery(
            condition=self.root_condition,
            sort_specs=self.sort_specs,
            limit_value=self.limit_value,
            offset_value=self.offset_value,
            fields=self.fields,
            vector_query=self.vector_query
        )


@dataclass
class ComplexQuery:
    """A query with complex boolean logic support."""

    # All fields have defaults to avoid ordering issues
    condition: Condition | None = None
    sort_specs: list = field(default_factory=list)
    limit_value: int | None = None
    offset_value: int | None = None
    fields: list[str] | None = None
    vector_query: VectorQuery | None = None  # Vector similarity search

    @classmethod
    def AND(cls, queries: list[Query]) -> ComplexQuery:
        """Create a complex query with AND logic."""
        from .query import Query

        conditions: list[Condition] = []
        for q in queries:
            if isinstance(q, Query):
                # Convert Query filters to conditions
                for f in q.filters:
                    conditions.append(FilterCondition(filter=f))

        return cls(
            condition=LogicCondition(operator=LogicOperator.AND, conditions=conditions)
        )

    @classmethod
    def OR(cls, queries: list[Query]) -> ComplexQuery:
        """Create a complex query with OR logic."""
        from .query import Query

        conditions: list[Condition] = []
        for q in queries:
            if isinstance(q, Query):
                # Convert Query filters to conditions
                for f in q.filters:
                    conditions.append(FilterCondition(filter=f))

        return cls(
            condition=LogicCondition(operator=LogicOperator.OR, conditions=conditions)
        )

    def matches(self, record: Any) -> bool:
        """Check if a record matches this query."""
        if self.condition is None:
            return True
        return self.condition.matches(record)

    def to_simple_query(self) -> Query:
        """Convert to simple Query if possible (AND filters only)."""
        from .query import Query

        filters = []

        # Try to extract simple filters if all are AND conditions
        if self.condition is None:
            pass
        elif isinstance(self.condition, FilterCondition):
            filters.append(self.condition.filter)
        elif isinstance(self.condition, LogicCondition) and self.condition.operator == LogicOperator.AND:
            # Check if all sub-conditions are simple filters
            all_filters = True
            for cond in self.condition.conditions:
                if isinstance(cond, FilterCondition):
                    filters.append(cond.filter)
                else:
                    all_filters = False
                    break

            if not all_filters:
                # Can't convert complex logic to simple query
                raise ValueError("Cannot convert complex boolean logic to simple Query")
        else:
            raise ValueError("Cannot convert complex boolean logic to simple Query")

        return Query(
            filters=filters,
            sort_specs=self.sort_specs,
            limit_value=self.limit_value,
            offset_value=self.offset_value,
            fields=self.fields,
            vector_query=self.vector_query
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}

        if self.condition:
            result["condition"] = self.condition.to_dict()

        if self.sort_specs:
            result["sort"] = [s.to_dict() for s in self.sort_specs]

        if self.limit_value is not None:
            result["limit"] = self.limit_value

        if self.offset_value is not None:
            result["offset"] = self.offset_value

        if self.fields is not None:
            result["fields"] = self.fields

        if self.vector_query is not None:
            result["vector_query"] = self.vector_query.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComplexQuery:
        """Create from dictionary representation."""
        from .query import SortSpec

        condition = None
        if "condition" in data:
            condition = condition_from_dict(data["condition"])

        sort_specs = []
        for sort_data in data.get("sort", []):
            sort_specs.append(SortSpec.from_dict(sort_data))

        vector_query = None
        if "vector_query" in data:
            vector_query = VectorQuery.from_dict(data["vector_query"])

        return cls(
            condition=condition,
            sort_specs=sort_specs,
            limit_value=data.get("limit"),
            offset_value=data.get("offset"),
            fields=data.get("fields"),
            vector_query=vector_query
        )
