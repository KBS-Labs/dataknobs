# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The column layout a SQL backend reads its table through, shared by every such backend.

A backend reads either the table this package creates (the JSON layout) or a
table somebody else owns, through its own typed columns (``layout: native``).
:class:`ColumnLayoutMixin` holds what that choice means for a backend whatever
its engine: the layout and the one query builder made at construction, the
refusal of every write on a native table, the capabilities it claims, and a
schema set afterwards. A backend names its dialect and its driver's placeholder
style, and answers how its engine is asked whether a table is there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from dataknobs_common.capabilities import Capability, CapabilityLike
from dataknobs_common.exceptions import OperationError

from ..operation_gate import GATED_OPERATIONS, OperationGateMixin
from .column_layout import ColumnLayout, JsonbLayout, read_layout_config
from .sql_base import SQLQueryBuilder, SQLTableManager

if TYPE_CHECKING:
    from ..schema import DatabaseSchema
    from .config import ColumnLayoutConfig

#: What a backend refuses when it reads a table through the native layout:
#: every gated operation -- each method that writes, creates or drops
#: something, and the vector reads, which search a JSON column a native table
#: does not have.
NATIVE_REFUSED: frozenset[str] = GATED_OPERATIONS


class ColumnLayoutMixin(OperationGateMixin):
    """The column layout a SQL backend reads its table through, shared by its twins.

    The layout and the one query builder are made at construction, from the
    configuration and the declared schema, so a native configuration the
    layout refuses fails before anything connects. A schema set afterwards
    makes a new layout, and one the layout refuses leaves the backend as it
    was.

    A backend mixes this in ahead of its database base, calls
    :meth:`_setup_layout` from ``_setup`` once ``table_name`` is set, and asks
    :meth:`_relation_exists_query` and :meth:`_missing_relation_error` when it
    connects.
    """

    #: The SQL dialect the builder renders.
    _DIALECT: ClassVar[str]
    #: The placeholder style of the twin's driver.
    _PARAM_STYLE: ClassVar[str]
    #: The configuration keys that say where the table is, as a refusal names them.
    _LOCATION_KEYS: ClassVar[str] = "the table name"

    schema: DatabaseSchema
    table_name: str
    layout: ColumnLayout
    query_builder: SQLQueryBuilder
    table_manager: SQLTableManager

    if TYPE_CHECKING:

        @property
        def config(self) -> ColumnLayoutConfig:
            """The consumer's typed configuration."""

    @property
    def native(self) -> bool:
        """Whether the table is read through its own columns (``layout: native``)."""
        return not isinstance(self.layout, JsonbLayout)

    def _namespace(self) -> str | None:
        """The SQL namespace the table is in, or ``None`` for the engine's default."""
        return None

    def _qualified_name(self) -> str:
        """The table's name as a refusal says it."""
        namespace = self._namespace()
        return f"{namespace}.{self.table_name}" if namespace else self.table_name

    def _read_layout(self, schema: DatabaseSchema) -> ColumnLayout:
        cfg = self.config
        context: dict[str, Any] = {"table": self.table_name}
        if self._namespace() is not None:
            context["schema_name"] = self._namespace()
        return read_layout_config(
            {"layout": cfg.layout, "id_column": cfg.id_column, "scope": cfg.scope},
            schema,
            origin=f"{type(self).__name__} table {self.table_name!r}",
            context=context,
        )

    def _use_layout(self, layout: ColumnLayout) -> None:
        self.layout = layout
        self.query_builder = SQLQueryBuilder(
            self.table_name,
            self._namespace(),
            dialect=self._DIALECT,
            param_style=self._PARAM_STYLE,
            layout=layout,
        )

    def _setup_layout(self) -> None:
        self._use_layout(self._read_layout(self.schema))

    def _relation_exists_query(self) -> tuple[str, Any]:
        """The statement and parameters asking whether the table is there, in the driver's style.

        The engine's table lookup by default. A backend whose lookup misses a
        relation a native table may be -- a view, a materialized view -- asks
        its own way under that layout.
        """
        return self.table_manager.get_table_exists_sql()

    def _missing_relation_error(self) -> RuntimeError:
        """The refusal for a table that is not there, said as the layout would say it."""
        qualified = self._qualified_name()
        if self.native:
            return RuntimeError(
                f"Table {qualified} does not exist. A table read through "
                f"`layout: native` belongs to someone else and is never created "
                f"here: check {self._LOCATION_KEYS}."
            )
        return RuntimeError(
            f"Table {qualified} does not exist and auto_create_table is disabled. "
            "Run your migrations before starting the application."
        )

    def _refuse_operation(self, operation: str) -> None:
        """Refuse every :data:`NATIVE_REFUSED` operation on a table read through the native layout."""
        if self.native and operation in NATIVE_REFUSED:
            raise OperationError(
                f"{type(self).__name__}.{operation} refused on {self.table_name!r}: a table read "
                f"through `layout: native` is read-only, and is read only through its declared "
                f"columns",
                context={"table": self.table_name, "method": operation},
            )
        super()._refuse_operation(operation)

    def instance_capabilities(self) -> frozenset[CapabilityLike]:
        """The class's capabilities, less conditional writes on a native table."""
        capabilities: frozenset[CapabilityLike] = super().instance_capabilities()  # type: ignore[misc]
        if self.native:
            return frozenset(c for c in capabilities if c != Capability.CONDITIONAL_WRITE)
        return capabilities

    def set_schema(self, schema: DatabaseSchema) -> None:
        """Set the declared schema, and read the table through the layout it makes.

        ``add_field_schema`` and ``with_schema`` come through here too, so a
        schema whose layout is refused leaves the backend as it was, whichever
        door it came through.
        """
        layout = self._read_layout(schema)
        super().set_schema(schema)  # type: ignore[misc]
        self._use_layout(layout)


class FileLayoutMixin(ColumnLayoutMixin):
    """The column layout of a table in a database file, shared by SQLite's and DuckDB's twins.

    A native table is in somebody else's file, which is opened read-only and is
    never made. A backend names what its engine needs, beyond a readable file,
    to open one read-only, and the refusal of a file that will not open says it.
    """

    _LOCATION_KEYS: ClassVar[str] = "the table name and `path`"
    #: What the engine needs, beyond a readable file, to open one read-only.
    _READ_ONLY_NEEDS: ClassVar[str] = ""

    db_path: str

    def _unopened_file_error(self, error: Exception) -> RuntimeError:
        """The refusal for a native table's file that cannot be read: most often, one not there."""
        needs = f" {self._READ_ONLY_NEEDS}" if self._READ_ONLY_NEEDS else ""
        return RuntimeError(
            f"Database file {self.db_path} cannot be opened read-only ({error}). A table "
            f"read through `layout: native` is in a file somebody else made, and no database "
            f"file is created here: check `path`.{needs}"
        )
