"""Shared aiohttp error scaffolding for the ollama / huggingface S4 tests.

Both providers speak the HuggingFace/Ollama HTTP APIs over ``aiohttp`` (no
vendor SDK), so their vendor-error-translation tests need the same three
pieces: a builder for a **real** ``aiohttp.ClientResponseError`` (never a fake —
the real exception class is what ``raise_for_status()`` raises), a fake response
whose ``raise_for_status()`` raises a scripted error, and a fake session whose
``post()`` returns a scripted async context manager (or raises a connection
error / timeout on entry).

Underscore-prefixed so pytest does not collect it as a test module.
"""

from __future__ import annotations

from typing import Any

import aiohttp
from aiohttp import ClientResponseError, RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL


def make_client_response_error(
    status: int,
    message: str = "error",
    headers: dict[str, str] | None = None,
    url: str = "http://localhost/api",
) -> ClientResponseError:
    """Build a real ``aiohttp.ClientResponseError`` for *status*.

    This is exactly the error ``ClientResponse.raise_for_status()`` raises for a
    non-2xx response, so translating it in a test exercises the real code path.
    """
    request_info = RequestInfo(
        URL(url), "POST", CIMultiDictProxy(CIMultiDict()), URL(url)
    )
    return ClientResponseError(
        request_info,
        (),
        status=status,
        message=message,
        headers=CIMultiDict(headers or {}),
    )


class FakeResponse:
    """Stand-in for an ``aiohttp.ClientResponse``.

    ``raise_for_status()`` raises *raise_exc* when set (a real
    ``ClientResponseError``); ``text()`` / ``json()`` return the scripted body.
    """

    def __init__(
        self,
        status: int = 200,
        *,
        text: str = "",
        json_data: Any = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.status = status
        self._text = text
        self._json = json_data
        self._raise_exc = raise_exc

    async def text(self) -> str:
        return self._text

    async def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self._raise_exc is not None:
            raise self._raise_exc


class _PostCtx:
    """Async-context-manager stand-in for ``session.post(...)``.

    Raises *enter_exc* on entry (a connection error / timeout surfaces here in
    real aiohttp) when set; otherwise yields the scripted response.
    """

    def __init__(
        self,
        response: FakeResponse | None = None,
        enter_exc: Exception | None = None,
    ) -> None:
        self._response = response
        self._enter_exc = enter_exc

    async def __aenter__(self) -> FakeResponse:
        if self._enter_exc is not None:
            raise self._enter_exc
        assert self._response is not None
        return self._response

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeSession:
    """Minimal ``aiohttp.ClientSession`` stand-in with scripted ``post`` outcomes.

    Each ``post()`` pops the next ``_PostCtx``. Construct outcomes with
    :meth:`responding` (response, possibly raising on ``raise_for_status``) or
    :meth:`failing` (raises on context entry — connection error / timeout).
    """

    def __init__(self, outcomes: list[_PostCtx]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[str] = []

    def post(self, url: str, json: Any = None) -> _PostCtx:
        self.calls.append(url)
        return self._outcomes.pop(0) if self._outcomes else _PostCtx(
            FakeResponse(200, json_data={})
        )

    @staticmethod
    def responding(response: FakeResponse) -> _PostCtx:
        return _PostCtx(response=response)

    @staticmethod
    def failing(exc: Exception) -> _PostCtx:
        return _PostCtx(enter_exc=exc)


__all__ = [
    "aiohttp",
    "make_client_response_error",
    "FakeResponse",
    "FakeSession",
]
