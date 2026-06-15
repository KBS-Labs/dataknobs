"""Registry tests for ``intent_classifier_backends``.

Pins the public registry surface after the
:class:`~dataknobs_common.registry.PluginRegistry` consolidation:

* ``PluginRegistry`` + ``BackendRegistry`` Protocol conformance.
* Explicit-key dispatch through :func:`create_intent_classifier`.
* Not-found error shape (``ValueError`` class preserved; new text
  produced by the parametrized ``not_found_kind`` shape — the
  pre-consolidation ``"Unknown intent_classifier '<name>'
  (registered: …)"`` text becomes ``"Unknown intent_classifier:
  <name>. Available backends: …"``; substrings asserted by the
  legacy test in ``test_intent_classifiers.py`` continue to pass).
* Composite recursion preserved end-to-end.
* :func:`create_intent_classifier_async` returns the same instance
  type as the sync shim for an identical config dict; async with
  sync-recursive composite child works.
* Consumer-registered custom backend resolves through both
  :func:`create_intent_classifier` and the registry directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from dataknobs_common.registry import BackendRegistry, PluginRegistry
from dataknobs_llm.intent import (
    CompositeIntentClassifier,
    IntentClassifier,
    IntentMatchResult,
    IntentSpec,
    KeywordIntentClassifier,
    create_intent_classifier,
    create_intent_classifier_async,
    intent_classifier_backends,
)


class TestIntentClassifierBackendsPluginRegistry:
    """``intent_classifier_backends`` is the fourth domain consolidated
    onto :class:`PluginRegistry` (after events, locks, ratelimit).
    Conformance + error-shape pins ride alongside the events/locks/
    ratelimit suites' equivalents in ``packages/common/tests/``.
    """

    def test_intent_classifier_backends_is_plugin_registry(self) -> None:
        assert isinstance(intent_classifier_backends, PluginRegistry)

    def test_intent_classifier_backends_is_backend_registry(self) -> None:
        assert isinstance(intent_classifier_backends, BackendRegistry)

    def test_create_intent_classifier_explicit_key_dispatch(self) -> None:
        """``create_intent_classifier(name, config)`` passes ``key=name``
        through to :meth:`PluginRegistry.create` — the explicit-key
        mode (no ``config_key`` configured). Validates the
        pre-consolidation public signature is preserved."""
        clf = create_intent_classifier("keyword", {})
        assert isinstance(clf, KeywordIntentClassifier)

    def test_create_intent_classifier_unknown_name_error_shape(self) -> None:
        """The new error shape:
        ``"Unknown intent_classifier: <name>. Available backends:
        <sorted-keys>"`` — produced by the parametrized
        ``not_found_kind="intent_classifier"`` /
        ``not_found_exception=ValueError`` registry knobs."""
        with pytest.raises(ValueError) as excinfo:
            create_intent_classifier("does_not_exist", {})
        msg = str(excinfo.value)
        assert "Unknown intent_classifier: does_not_exist" in msg
        assert "Available backends:" in msg
        # Every built-in is enumerated in the sorted message.
        for name in ("composite", "keyword", "llm"):
            assert name in msg

    def test_create_intent_classifier_exception_class_preserved(self) -> None:
        """The not-found raise is a plain :class:`ValueError` (the
        historical contract), not the :class:`NotFoundError`
        :class:`PluginRegistry` defaults to. The
        ``not_found_exception=ValueError`` ctor knob is the seam keeping
        the pre-consolidation call-site catches valid."""
        with pytest.raises(ValueError):
            create_intent_classifier("does_not_exist", {})

    def test_create_intent_classifier_composite_recursion(self) -> None:
        """Composite recursion routes child specs through the
        consolidated :func:`create_intent_classifier` shim. The
        end-to-end composite-of-keyword path is preserved unchanged."""
        composite = create_intent_classifier(
            "composite",
            {"classifiers": [{"classifier": "keyword"}]},
        )
        assert isinstance(composite, CompositeIntentClassifier)

    def test_create_intent_classifier_none_config(self) -> None:
        """``config=None`` is normalized to ``{}`` for both key
        resolution and factory invocation — same as the
        pre-consolidation shim."""
        clf = create_intent_classifier("keyword", None)
        assert isinstance(clf, KeywordIntentClassifier)

    def test_create_intent_classifier_unwraps_factory_value_errors(
        self,
    ) -> None:
        """The composite factory raises ``ValueError`` for malformed
        child specs (missing ``classifier:`` discriminator, non-mapping
        child, unknown child name).
        :meth:`PluginRegistry.create` wraps any factory-raised
        non-``OperationError`` exception in ``OperationError``;
        :func:`create_intent_classifier` re-raises the original
        ``ValueError`` so the historical consumer contract is
        preserved.

        Pinned with all three composite child-validation paths because
        any regression of the unwrap would silently downgrade
        ``ValueError``-catching consumer code to a missed-catch fallthrough.
        """
        # Missing ``classifier:`` discriminator on a child spec.
        with pytest.raises(ValueError) as excinfo:
            create_intent_classifier(
                "composite",
                {"classifiers": [{"classifer": "keyword"}]},  # typo
            )
        assert "missing required 'classifier'" in str(excinfo.value)

        # Non-mapping child spec.
        with pytest.raises(ValueError) as excinfo:
            create_intent_classifier(
                "composite",
                {"classifiers": ["keyword"]},
            )
        assert "must be a mapping" in str(excinfo.value)

        # Unknown child ``classifier:`` name — the inner recursion
        # raises ``ValueError`` (from the not-found shape), gets
        # wrapped in ``OperationError`` by the outer ``create()``,
        # then unwrapped by the shim.
        with pytest.raises(ValueError) as excinfo:
            create_intent_classifier(
                "composite",
                {"classifiers": [{"classifier": "does_not_exist"}]},
            )
        assert "does_not_exist" in str(excinfo.value)

    def test_intent_classifier_custom_backend_via_registry(self) -> None:
        """An out-of-tree consumer registers a custom
        :class:`IntentClassifier` backend through the registry surface
        and dispatches it via
        :func:`create_intent_classifier(name, config)` — the
        consumer-extensibility capability the consolidation surfaces.

        ``validate_type=IntentClassifier`` (set on the registry) enforces
        that the factory's resolved instance structurally conforms to
        the :class:`IntentClassifier` Protocol; the
        :func:`isinstance(clf, IntentClassifier)` check inside the
        ``try`` block exercises the same Protocol contract from the
        consumer side.
        """

        class _StubClassifier:
            async def classify(
                self,
                message: str,
                intents: Sequence[IntentSpec],
                **_: Any,
            ) -> IntentMatchResult:
                return IntentMatchResult(
                    intent=None,
                    extracted=None,
                    rule_based=False,
                    raw_reply=message,
                )

        def _make_stub(config: dict[str, Any]) -> _StubClassifier:
            return _StubClassifier()

        intent_classifier_backends.register(
            "test-custom", _make_stub, allow_overwrite=True,
        )
        try:
            clf = create_intent_classifier("test-custom", {})
            assert isinstance(clf, _StubClassifier)
            assert isinstance(clf, IntentClassifier)
        finally:
            intent_classifier_backends.unregister("test-custom")


class TestIntentClassifierAsyncShim:
    """``create_intent_classifier_async`` is shipped for API symmetry
    and consumer-extensibility (an out-of-tree classifier's
    ``from_config_async`` would be detected and awaited). Every
    built-in classifier constructs synchronously; the async shim must
    return the same instance type as the sync shim for an identical
    config dict.
    """

    async def test_create_intent_classifier_async_returns_same_type_as_sync(
        self,
    ) -> None:
        sync_clf = create_intent_classifier("keyword", {})
        async_clf = await create_intent_classifier_async("keyword", {})
        assert type(sync_clf) is type(async_clf)
        assert isinstance(async_clf, KeywordIntentClassifier)

    async def test_create_intent_classifier_async_preserves_error_shape(
        self,
    ) -> None:
        """The async shim's not-found raise is the same shape as the
        sync shim — same exception class, same message wording (the
        symmetry guard for the ``create_async`` path through
        ``_resolve_factory``)."""
        with pytest.raises(ValueError) as excinfo:
            await create_intent_classifier_async("does_not_exist", {})
        msg = str(excinfo.value)
        assert "Unknown intent_classifier: does_not_exist" in msg
        assert "Available backends:" in msg

    async def test_create_intent_classifier_async_recursion_via_sync_factory(
        self,
    ) -> None:
        """The composite factory's recursion still uses the **sync**
        :func:`create_intent_classifier` inside
        :func:`_composite_factory`. An async outer call → sync inner
        recursion works because the sync factory's non-awaitable
        result passes through ``create_async`` unchanged (the
        ``inspect.isawaitable`` branch is skipped)."""
        composite = await create_intent_classifier_async(
            "composite",
            {"classifiers": [{"classifier": "keyword"}]},
        )
        assert isinstance(composite, CompositeIntentClassifier)

    async def test_create_intent_classifier_async_unwraps_factory_value_errors(
        self,
    ) -> None:
        """Async-side parity of the unwrap behaviour:
        :func:`create_intent_classifier_async` re-raises a
        ``ValueError`` from inside the factory rather than surfacing
        the :class:`PluginRegistry`-wrapped ``OperationError``."""
        with pytest.raises(ValueError) as excinfo:
            await create_intent_classifier_async(
                "composite",
                {"classifiers": [{"classifier": "does_not_exist"}]},
            )
        assert "does_not_exist" in str(excinfo.value)
