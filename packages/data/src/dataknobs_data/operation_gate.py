# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The question every body of a gated operation asks a backend instance before it runs.

The database bases and the vector and bulk-embed mixins hold one body for each
write, for what creates or drops a vector index, and for the vector search
surface, and a backend either inherits that body or defines its own. A backend instance that cannot perform one of them -- a table read in
place, which nothing may write -- refuses it in one place,
:meth:`OperationGateMixin._refuse_operation`, however its class came by the
body.

That holds because no body asks by hand. :meth:`OperationGateMixin.__init_subclass__`
wraps each :data:`GATED_OPERATIONS` body a class defines, when the class is
defined, so a backend written tomorrow is gated without knowing it is.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from dataknobs_common.callbacks import is_async_callable

GATED_OPERATIONS: frozenset[str] = frozenset(
    {
        # writes
        "create", "update", "delete", "upsert", "clear",
        "create_batch", "upsert_batch", "delete_batch", "update_batch",
        "stream_write", "bulk_embed_and_store", "update_vector", "delete_from_index",
        "add_vectors", "transaction", "begin_transaction",
        # what creates or drops
        "enable_vector_support", "create_vector_index", "drop_vector_index",
        # the vector search surface
        "vector_search", "hybrid_search", "get_vector_index_stats",
    }
)  # fmt: skip
"""The operations whose every body asks :meth:`OperationGateMixin._refuse_operation` first."""

#: Set on a wrapper this module made, so a body is never gated twice.
_GATED_MARK = "_operation_gated"


class OperationGateMixin:
    """Gates every :data:`GATED_OPERATIONS` body a subclass defines behind one hook."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap each gated body this class defines so it asks the gate first.

        Runs for every class below this one -- the database bases and the
        vector and bulk-embed mixins as much as each backend -- so a body is
        gated by the class that defines it, and an override that never calls
        ``super()`` is gated like any other.
        """
        super().__init_subclass__(**kwargs)
        _install_operation_gate(cls)

    def _refuse_operation(self, operation: str) -> None:
        """Raise to refuse ``operation`` on this instance; return to permit it.

        Asked before every body of a :data:`GATED_OPERATIONS` member runs:
        before a row is read, and before a caller's embedding function is
        spent. ``operation`` is the public method's name. A context manager
        such as ``transaction`` is refused when it is called, before it is
        entered.

        The default permits everything.

        Raises:
            OperationError: In an override, naming the operation and why this
                instance cannot perform it.
        """


def _install_operation_gate(cls: type) -> None:
    """Wrap each :data:`GATED_OPERATIONS` body in ``cls.__dict__`` with the gate.

    Skips abstract declarations and bodies already gated. A class inheriting a
    body resolves to the defining class's wrapper, so nothing is gated twice.

    Raises:
        TypeError: When a gated operation is defined as something other than a
            plain function, or as an async generator function -- whose body a
            wrapper cannot run ahead of without becoming one itself.
    """
    for name in GATED_OPERATIONS & cls.__dict__.keys():
        body = cls.__dict__[name]
        if getattr(body, "__isabstractmethod__", False) or getattr(body, _GATED_MARK, False):
            continue
        if not inspect.isfunction(body) or inspect.isasyncgenfunction(body):
            raise TypeError(
                f"{cls.__qualname__}.{name} is a gated operation and must be a plain "
                f"function or coroutine function, so its body can ask "
                f"_refuse_operation before it runs; got {body!r}"
            )
        setattr(cls, name, _gated(name, body))


def _gated(name: str, body: Callable[..., Any]) -> Callable[..., Any]:
    if is_async_callable(body):

        @functools.wraps(body)
        async def gated_async(self: OperationGateMixin, *args: Any, **kwargs: Any) -> Any:
            self._refuse_operation(name)
            return await body(self, *args, **kwargs)

        setattr(gated_async, _GATED_MARK, True)
        return gated_async

    @functools.wraps(body)
    def gated(self: OperationGateMixin, *args: Any, **kwargs: Any) -> Any:
        self._refuse_operation(name)
        return body(self, *args, **kwargs)

    setattr(gated, _GATED_MARK, True)
    return gated
