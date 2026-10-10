# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The question a shared operation body asks a backend instance before it runs.

The database bases and the vector and bulk-embed mixins hold one body for each
write, for what creates or drops a vector index, and for the vector search
surface, and a backend inherits it rather than restating it. So a backend
instance that cannot perform one of them -- a Postgres table read in place,
which nothing may write -- has no method of its own to refuse it from.
:meth:`OperationGateMixin._refuse_operation` is where it refuses instead.
"""

from __future__ import annotations


class OperationGateMixin:
    """Declares the hook every shared operation body calls before anything else."""

    def _refuse_operation(self, operation: str) -> None:
        """Raise to refuse ``operation`` on this instance; return to permit it.

        Called first by each shared body of a write, of what creates or drops
        a vector index, and of the vector search surface: before a row is read,
        and before a caller's embedding function is spent. ``operation`` is the
        public method's name. A backend that defines its own body for one of
        these calls this from it too, so an instance refuses by overriding one
        method however its class came by the rest.

        The default permits everything.

        Raises:
            OperationError: In an override, naming the operation and why this
                instance cannot perform it.
        """
