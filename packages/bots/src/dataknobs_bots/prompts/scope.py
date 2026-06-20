"""Jinja-templating-side ScopeProjector reference implementation.

This module lives in the bots layer because ``jinja2`` is a bots
dependency; placing it here avoids forcing an optional jinja2 import onto
``dataknobs_common``.

Other ScopeProjector reference impls (Identity, ReadOnly, Whitelist,
Chained, Callable, Cached) live in ``dataknobs_common.scope`` and have no
external dependencies.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

__all__ = ["JinjaInputsProjector"]

logger = logging.getLogger(__name__)


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

    Security — sandboxed by default: when no ``env=`` is supplied, the
    default environment is a
    :class:`~jinja2.sandbox.SandboxedEnvironment` (via the bots-layer
    :func:`~dataknobs_bots.utils.template_env.create_template_env`). The
    expressions are evaluated over runtime data that may be user-influenced,
    so the safe environment is the default — attribute-traversal escapes
    (``x.__class__.__mro__`` …) raise ``SecurityError`` rather than leaking
    interpreter internals. Pass ``env=`` only to share a pre-built
    environment (e.g. the wizard's, so consumer-registered filters/globals
    are honored); a caller who genuinely needs an unsandboxed environment
    must construct and pass one explicitly.

    Error handling — ``strict``: by default (``strict=True``) a failing
    expression (sandbox violation, type error against the runtime data,
    undefined-in-strict-mode) propagates. Construct with ``strict=False``
    to degrade gracefully — a failing expression is logged at WARNING and
    that single input is omitted from the result, so one malformed
    expression cannot abort the whole projection (and, by extension, the
    whole render). The wizard renderer constructs with ``strict=False`` so
    a bad author expression skips rather than taking down the stage.

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
        strict: bool = True,
    ) -> None:
        self._inputs = dict(inputs)
        self._base_context = dict(base_context)
        self._env = env
        self._strict = strict

    def project(self, source: Any) -> Mapping[str, Any]:
        # source ignored for Protocol conformance; behavior captured at
        # construction.
        env = self._env or self._default_env()
        result: dict[str, Any] = {}
        for name, expression in self._inputs.items():
            compiled = env.compile_expression(expression)
            try:
                result[name] = compiled(**self._base_context)
            except Exception:
                # Expressions evaluate over arbitrary runtime data, so any
                # exception type is possible (jinja TemplateError /
                # SecurityError, plain TypeError / ValueError, …). Under
                # strict mode propagate; otherwise log and skip this single
                # input so one malformed expression cannot abort the rest.
                if self._strict:
                    raise
                logger.warning(
                    "Skipping declarative input %r: expression %r failed "
                    "to evaluate",
                    name,
                    expression,
                    exc_info=True,
                )
        return result

    @staticmethod
    def _default_env() -> Any:
        # Lazy import — jinja2 is a bots-layer dependency. Default to the
        # canonical sandboxed factory so the safe environment is the
        # default for a general-purpose, shared-infra reference impl.
        from dataknobs_bots.utils.template_env import create_template_env

        return create_template_env()
