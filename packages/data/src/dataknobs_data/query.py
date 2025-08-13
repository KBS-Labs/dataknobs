from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class Operator(Enum):
    """Query operators for filtering."""
    EQ = "="           # Equal
    NEQ = "!="         # Not equal
    GT = ">"           # Greater than
    GTE = ">="         # Greater than or equal
    LT = "<"           # Less than
    LTE = "<="         # Less than or equal
    IN = "in"          # In list
    NOT_IN = "not_in"  # Not in list
    LIKE = "like"      # String pattern matching (SQL LIKE)
    REGEX = "regex"    # Regular expression matching
    EXISTS = "exists"  # Field exists
    NOT_EXISTS = "not_exists"  # Field does not exist


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
        """Check if a record value matches this filter."""
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
            return record_value > self.value
        elif self.operator == Operator.GTE:
            return record_value >= self.value
        elif self.operator == Operator.LT:
            return record_value < self.value
        elif self.operator == Operator.LTE:
            return record_value <= self.value
        elif self.operator == Operator.IN:
            return record_value in self.value
        elif self.operator == Operator.NOT_IN:
            return record_value not in self.value
        elif self.operator == Operator.LIKE:
            if not isinstance(record_value, str):
                return False
            import re
            pattern = self.value.replace('%', '.*').replace('_', '.')
            return bool(re.match(f"^{pattern}$", record_value))
        elif self.operator == Operator.REGEX:
            if not isinstance(record_value, str):
                return False
            import re
            return bool(re.search(self.value, record_value))
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary representation."""
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Filter':
        """Create filter from dictionary representation."""
        return cls(
            field=data["field"],
            operator=Operator(data["operator"]),
            value=data.get("value")
        )


@dataclass
class SortSpec:
    """Represents a sort specification."""
    field: str
    order: SortOrder = SortOrder.ASC
    
    def to_dict(self) -> Dict[str, str]:
        """Convert sort spec to dictionary representation."""
        return {
            "field": self.field,
            "order": self.order.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'SortSpec':
        """Create sort spec from dictionary representation."""
        return cls(
            field=data["field"],
            order=SortOrder(data.get("order", "asc"))
        )


@dataclass
class Query:
    """Represents a database query with filters, sorting, and pagination."""
    
    filters: List[Filter] = field(default_factory=list)
    sort: List[SortSpec] = field(default_factory=list)
    limit: Optional[int] = None
    offset: Optional[int] = None
    fields: Optional[List[str]] = None  # Field projection
    
    def filter(self, field: str, operator: Union[str, Operator], value: Any = None) -> 'Query':
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
                "not_in": Operator.NOT_IN,
                "like": Operator.LIKE,
                "LIKE": Operator.LIKE,
                "regex": Operator.REGEX,
                "exists": Operator.EXISTS,
                "not_exists": Operator.NOT_EXISTS,
            }
            operator = op_map.get(operator, Operator.EQ)
        
        self.filters.append(Filter(field=field, operator=operator, value=value))
        return self
    
    def sort_by(self, field: str, order: Union[str, SortOrder] = "asc") -> 'Query':
        """Add a sort specification to the query (fluent interface).
        
        Args:
            field: The field name to sort by
            order: The sort order ("asc", "desc", or SortOrder enum)
            
        Returns:
            Self for method chaining
        """
        if isinstance(order, str):
            order = SortOrder.ASC if order.lower() == "asc" else SortOrder.DESC
        
        self.sort.append(SortSpec(field=field, order=order))
        return self
    
    def set_limit(self, limit: int) -> 'Query':
        """Set the result limit (fluent interface).
        
        Args:
            limit: Maximum number of results
            
        Returns:
            Self for method chaining
        """
        self.limit = limit
        return self
    
    def set_offset(self, offset: int) -> 'Query':
        """Set the result offset (fluent interface).
        
        Args:
            offset: Number of results to skip
            
        Returns:
            Self for method chaining
        """
        self.offset = offset
        return self
    
    def select(self, *fields: str) -> 'Query':
        """Set field projection (fluent interface).
        
        Args:
            fields: Field names to include in results
            
        Returns:
            Self for method chaining
        """
        self.fields = list(fields) if fields else None
        return self
    
    def clear_filters(self) -> 'Query':
        """Clear all filters (fluent interface)."""
        self.filters = []
        return self
    
    def clear_sort(self) -> 'Query':
        """Clear all sort specifications (fluent interface)."""
        self.sort = []
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary representation."""
        result = {
            "filters": [f.to_dict() for f in self.filters],
            "sort": [s.to_dict() for s in self.sort],
        }
        if self.limit is not None:
            result["limit"] = self.limit
        if self.offset is not None:
            result["offset"] = self.offset
        if self.fields is not None:
            result["fields"] = self.fields
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Query':
        """Create query from dictionary representation."""
        query = cls()
        
        for filter_data in data.get("filters", []):
            query.filters.append(Filter.from_dict(filter_data))
        
        for sort_data in data.get("sort", []):
            query.sort.append(SortSpec.from_dict(sort_data))
        
        query.limit = data.get("limit")
        query.offset = data.get("offset")
        query.fields = data.get("fields")
        
        return query
    
    def copy(self) -> 'Query':
        """Create a copy of the query."""
        import copy
        return Query(
            filters=copy.deepcopy(self.filters),
            sort=copy.deepcopy(self.sort),
            limit=self.limit,
            offset=self.offset,
            fields=self.fields.copy() if self.fields else None
        )