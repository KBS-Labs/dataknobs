"""aiohttp request-shape knowledge shared across the HTTP-only providers.

Two dataknobs providers speak their vendor API directly over ``aiohttp`` with
no vendor SDK — Ollama (:mod:`~dataknobs_llm.llm.providers.ollama`) and
HuggingFace (:mod:`~dataknobs_llm.llm.providers.huggingface`). They share the
same transport-error shape, so the transport-boundary behavior they both need
lives here — imported by both — rather than duplicated in each (where it would
inevitably drift), mirroring ``_claude_shared`` for the Claude family.

``aiohttp`` is imported lazily inside the helper so importing this module stays
transport-free (matching each provider's own lazy ``import aiohttp``).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def raise_for_status_with_body(response: Any, *, body: str | None = None) -> None:
    """Like ``ClientResponse.raise_for_status()``, but keep the response body.

    aiohttp's own :meth:`ClientResponse.raise_for_status` sets the raised
    :class:`aiohttp.ClientResponseError`'s ``message`` to the HTTP *reason
    phrase* only (``"Bad Request"``), discarding the response **body** — which
    is exactly where a vendor puts the actionable error wording (a
    context-window overflow phrase, a validation detail). Both aiohttp-based
    providers route their non-2xx responses through here so the body survives
    into ``str(exc)`` and reaches the shared status dispatch
    (:meth:`~dataknobs_llm.llm.base.LLMProvider._dataknobs_error_for_status`),
    whose markers can then classify e.g. a context-length overflow that the
    bare reason phrase would hide.

    The real ``ClientResponseError`` that ``raise_for_status()`` raises is
    reused (not reconstructed), so its identity is preserved for the caller's
    ``... from exc`` chaining — only its ``message`` is enriched with the body.

    Args:
        response: The aiohttp ``ClientResponse`` (or a faithful stand-in) to
            check. Its ``raise_for_status`` and (when ``body`` is not supplied)
            ``text`` are used.
        body: The already-read response body, when the caller read it for its
            own gating (e.g. Ollama's "does not support tools" check). Supplying
            it avoids a redundant re-read; when ``None`` the body is read here,
            but only on the error path (``status >= 400``) so a success response
            is never eagerly drained.
    """
    if response.status < 400:
        return
    import aiohttp

    if body is None:
        try:
            body = await response.text()
        except Exception as read_exc:  # pragma: no cover - body unavailable
            # Best-effort: fall back to the bare reason phrase. The raised
            # ClientResponseError still carries status/headers, so classification
            # is unaffected — only the marker enrichment is lost.
            logger.debug("Failed to read error response body: %s", read_exc)
            body = ""
    try:
        response.raise_for_status()
    except aiohttp.ClientResponseError as exc:
        detail = (body or "").strip()
        if detail:
            # Replace the bare reason phrase with the vendor body so a marker
            # in the body reaches _dataknobs_error_for_status.
            exc.message = detail
        raise


__all__ = ["raise_for_status_with_body"]
