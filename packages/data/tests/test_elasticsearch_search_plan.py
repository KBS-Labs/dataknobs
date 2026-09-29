"""The plan both Elasticsearch twins drive to read one search's page.

:func:`plan_search` does no I/O, so what it asks for can be pinned here by
answering its requests from a list of fake hits, in order. The live-node half,
over both twins, is ``integration/test_elasticsearch_search_paging.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data.backends.config import (
    AsyncElasticsearchDatabaseConfig,
    SyncElasticsearchDatabaseConfig,
)
from dataknobs_data.backends.elasticsearch_query import (
    DEFAULT_MAX_RESULT_WINDOW,
    arun_search_plan,
    client_search_kwargs,
    plan_search,
    run_search_plan,
)

_QUERY: dict[str, Any] = {"match_all": {}}
_SORT = [{"data.n": {"order": "asc"}}]


class _Index:
    """Answers a plan's requests from ``total`` hits in ``id`` order.

    It also plays the point in time: each paged response returns a new id,
    as Elasticsearch may, so a driver that keeps sending the first one is
    caught.
    """

    def __init__(self, total: int) -> None:
        self.hits = [{"_id": f"r{n:03d}", "sort": [n, n]} for n in range(total)]
        self.requests: list[dict[str, Any]] = []
        self.opened = 0
        self.closed: list[str] = []
        self._pit = 0

    def __call__(self, body: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
        self.requests.append(body)
        if "search_after" in body:
            start = body["search_after"][0] + 1
        else:
            start = body.get("from", 0)
        page = self.hits[start : start + body["size"]]
        if "pit" not in body:
            return page, None
        assert body["pit"]["id"] == f"pit-{self._pit}", "sent a stale point-in-time id"
        self._pit += 1
        return page, f"pit-{self._pit}"

    def open_pit(self) -> str:
        self.opened += 1
        return f"pit-{self._pit}"

    def close_pit(self, pit_id: str) -> None:
        self.closed.append(pit_id)

    def ids(self, start: int, stop: int) -> list[str]:
        return [hit["_id"] for hit in self.hits[start:stop]]


def _run(index: _Index, offset: int | None, limit: int | None, **kw: Any) -> list[str]:
    plan = plan_search(_QUERY, _SORT, offset, limit, **kw)
    hits = run_search_plan(plan, index, index.open_pit, index.close_pit)
    return [hit["_id"] for hit in hits]


class TestABoundedReadInsideTheWindowIsOneRequest:
    def test_limit_and_offset(self) -> None:
        index = _Index(50)
        assert _run(index, 5, 10) == index.ids(5, 15)
        assert index.requests == [{"query": _QUERY, "from": 5, "size": 10, "sort": _SORT}]
        assert index.opened == 0, "a read inside the window needs no point in time"

    def test_limit_zero_is_size_zero(self) -> None:
        index = _Index(50)
        assert _run(index, None, 0) == []
        assert index.requests == [{"query": _QUERY, "from": 0, "size": 0, "sort": _SORT}]

    def test_exactly_the_window(self) -> None:
        index = _Index(30)
        assert _run(index, 4, 6, window=10) == index.ids(4, 10)
        assert len(index.requests) == 1


class TestAnythingElsePagesWithSearchAfter:
    def test_no_limit_reads_every_hit_in_one_point_in_time(self) -> None:
        index = _Index(23)
        assert _run(index, None, None, page_size=5) == index.ids(0, 23)
        assert all("from" not in body for body in index.requests)
        assert [body.get("search_after") for body in index.requests[1:]] == [
            index.hits[n]["sort"] for n in (4, 9, 14, 19)
        ]
        assert index.opened == 1
        assert index.closed == ["pit-5"], "the newest id is the one closed"

    def test_offset_without_a_limit_skips_across_pages(self) -> None:
        index = _Index(23)
        assert _run(index, 12, None, page_size=5) == index.ids(12, 23)

    def test_limit_and_offset_past_the_window(self) -> None:
        index = _Index(40)
        assert _run(index, 8, 7, window=10, page_size=4) == index.ids(8, 15)
        # Reads stop once the limit is met: 15 hits need four pages of four.
        assert len(index.requests) == 4

    def test_the_default_window_is_elasticsearchs(self) -> None:
        inside, past = _Index(3), _Index(3)
        _run(inside, DEFAULT_MAX_RESULT_WINDOW - 1, 1)
        _run(past, DEFAULT_MAX_RESULT_WINDOW, 1)
        assert inside.requests[0]["from"] == DEFAULT_MAX_RESULT_WINDOW - 1
        assert "from" not in past.requests[0]

    def test_an_offset_past_the_end_is_empty(self) -> None:
        assert _run(_Index(7), 20, None, page_size=5) == []

    def test_an_empty_index(self) -> None:
        assert _run(_Index(0), None, None) == []


class TestTheSortIsMadeTotal:
    def _first_sort(self, sort: list[dict[str, Any]]) -> list[dict[str, Any]]:
        plan = plan_search(_QUERY, sort, None, None)
        sent: list[dict[str, Any]] = next(plan)["sort"]
        plan.close()
        return sent

    def test_relevance_then_shard_order_when_none_is_given(self) -> None:
        assert self._first_sort([]) == [
            {"_score": {"order": "desc"}},
            {"_shard_doc": {"order": "asc"}},
        ]

    def test_the_callers_sort_then_shard_order(self) -> None:
        assert self._first_sort(_SORT) == [*_SORT, {"_shard_doc": {"order": "asc"}}]


class TestThePointInTimeIsClosed:
    def test_when_a_request_fails(self) -> None:
        index = _Index(23)
        calls = 0

        def failing(body: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("the node went away")
            return index(body)

        plan = plan_search(_QUERY, _SORT, None, None, page_size=5)
        with pytest.raises(RuntimeError, match="went away"):
            run_search_plan(plan, failing, index.open_pit, index.close_pit)
        assert index.closed == ["pit-2"]


class TestTheClientSpelling:
    def test_a_request_in_a_point_in_time_names_no_index(self) -> None:
        body = {"query": _QUERY, "size": 5, "sort": _SORT, "pit": {"id": "p", "keep_alive": "1m"}}
        kwargs = client_search_kwargs(body, "records")
        assert "index" not in kwargs
        assert kwargs["pit"] == {"id": "p", "keep_alive": "1m"}

    def test_a_request_outside_one_names_the_index(self) -> None:
        kwargs = client_search_kwargs({"query": _QUERY, "size": 5, "from": 2}, "records")
        assert (kwargs["index"], kwargs["from_"], kwargs["size"]) == ("records", 2, 5)


async def test_the_async_driver_reads_what_the_sync_one_does() -> None:
    index = _Index(23)

    async def execute(body: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
        return index(body)

    async def open_pit() -> str:
        return index.open_pit()

    async def close_pit(pit_id: str) -> None:
        index.close_pit(pit_id)

    plan = plan_search(_QUERY, _SORT, 3, None, page_size=5)
    hits = await arun_search_plan(plan, execute, open_pit, close_pit)
    assert [hit["_id"] for hit in hits] == index.ids(3, 23)
    assert (index.opened, index.closed) == (1, ["pit-5"])


@pytest.mark.parametrize(
    "config_class", [SyncElasticsearchDatabaseConfig, AsyncElasticsearchDatabaseConfig]
)
class TestThePagingSettings:
    def test_defaults(self, config_class: Any) -> None:
        config = config_class.from_dict({})
        assert (config.max_result_window, config.search_page_size) == (10_000, 1_000)

    def test_a_string_is_read_as_a_number(self, config_class: Any) -> None:
        config = config_class.from_dict({"max_result_window": "50", "search_page_size": "5"})
        assert (config.max_result_window, config.search_page_size) == (50, 5)

    @pytest.mark.parametrize(
        "settings",
        [
            {"max_result_window": 0},
            {"search_page_size": -1},
            {"search_page_size": True},
            {"max_result_window": "many"},
            {"max_result_window": 10, "search_page_size": 11},
        ],
    )
    def test_a_bad_value_is_refused(self, config_class: Any, settings: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="Elasticsearch"):
            config_class.from_dict(settings)
