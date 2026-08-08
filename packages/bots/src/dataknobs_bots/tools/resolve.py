"""Re-export of the shared dotted-path resolvers.

The implementation lives in :mod:`dataknobs_common.imports`, where every
package can reach it. This module used to hold its own copy — a good general
resolver that was exported from no ``__init__``, filed under a name that reads
as tool infrastructure, and reached by all of its callers through a
function-local deep import. Being unfindable is what caused it to be rewritten
eight more times across this package; the fix was to move it somewhere
findable, not to make this copy better.

It stays as a re-export so existing deep imports keep working. New code should
import from :mod:`dataknobs_common` directly.
"""

from __future__ import annotations

from dataknobs_common.imports import resolve_callable, resolve_optional_callable

__all__ = ["resolve_callable", "resolve_optional_callable"]
