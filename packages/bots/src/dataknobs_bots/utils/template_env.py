"""Sandboxed Jinja2 environment factory.

All template rendering in the bots package MUST use this factory
to ensure user-supplied data cannot exploit server-side template
injection (SSTI).  ``SandboxedEnvironment`` blocks access to dangerous
attributes (``__class__``, ``__subclasses__``, ``__globals__``, etc.)
while preserving normal template functionality (variables, loops,
conditionals, filters, ``data.get()``).
"""

from __future__ import annotations

import jinja2
from jinja2.sandbox import SandboxedEnvironment


def create_template_env(*, strict: bool = False) -> SandboxedEnvironment:
    """Create a sandboxed Jinja2 environment.

    Args:
        strict: If ``True``, use :class:`~jinja2.StrictUndefined`
            (raises on missing variables).  If ``False`` (default),
            use :class:`~jinja2.Undefined` (renders missing variables
            as empty strings).

    Returns:
        A :class:`~jinja2.sandbox.SandboxedEnvironment` instance.
    """
    undefined = jinja2.StrictUndefined if strict else jinja2.Undefined
    return SandboxedEnvironment(undefined=undefined)
