"""Jinja-templating-side ScopeProjector reference implementation.

This module lives in the bots layer because ``jinja2`` is a bots
dependency; placing it here avoids forcing an optional jinja2 import onto
``dataknobs_common``.

Other ScopeProjector reference impls (Identity, ReadOnly, Whitelist,
Chained, Callable, Cached) live in ``dataknobs_common.scope`` and have no
external dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["JinjaInputsProjector"]


class JinjaInputsProjector:
    """Evaluates declarative Jinja-expression ``inputs:`` against a base
    context.

    Construct with a mapping of ``name -> jinja_expression`` strings plus
    a ``base_context`` mapping. :meth:`project` evaluates each expression
    against the base context and returns a fresh dict of
    ``name -> evaluated_value``.

    Use case: a wizard stage declaring::

        inputs:
          sanitized_input: "user_input | lower | trim"
          input_length: "user_input | length"

    constructs ``JinjaInputsProjector(stage_inputs, base_context=context)``
    and merges the returned mapping into the response-template scope. Each
    expression is evaluated against the same base context captured at
    construction, so declared inputs cannot reference one another.

    The Jinja environment may be shared with the wizard's response-template
    environment (passed via the ``env=`` constructor kwarg) so
    consumer-registered filters/globals — and the sandboxing the wizard
    applies — are honored. Default construction lazily creates a
    ``jinja2.Environment()`` with no user-defined filters; pass the
    wizard's sandboxed env when evaluating expressions over user data.

    Source-argument semantics: the ``source`` argument to :meth:`project`
    is accepted for Protocol conformance but ignored — the base context is
    captured at construction. Consumers wanting per-call base-context
    selection construct a fresh projector.
    """

    def __init__(
        self,
        inputs: Mapping[str, str],
        base_context: Mapping[str, Any],
        *,
        env: Any = None,  # jinja2.Environment; typed Any to keep import lazy
    ) -> None:
        self._inputs = dict(inputs)
        self._base_context = dict(base_context)
        self._env = env

    def project(self, source: Any) -> Mapping[str, Any]:
        # source ignored for Protocol conformance; behavior captured at
        # construction.
        env = self._env or self._default_env()
        result: dict[str, Any] = {}
        for name, expression in self._inputs.items():
            compiled = env.compile_expression(expression)
            result[name] = compiled(**self._base_context)
        return result

    @staticmethod
    def _default_env() -> Any:
        # Lazy import — jinja2 is a bots-layer dependency.
        from jinja2 import Environment

        return Environment(autoescape=False)
