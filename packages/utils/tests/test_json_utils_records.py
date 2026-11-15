"""Behavioural tests for record grouping in ``dataknobs_utils.json_utils``.

Covers ``Path``, ``PathGroup``, ``ArrayElementAcceptStrategy``, ``PathSorter``,
``RecordPathBuilder``, the module function ``stream_record_paths`` and
``get_records_df``, through their public interfaces only.
"""

import io
from typing import Any

import pytest

from dataknobs_utils.json_utils import (
    ArrayElementAcceptStrategy,
    GroupAcceptStrategy,
    Path,
    PathGroup,
    PathSorter,
    RecordPathBuilder,
    get_records_df,
    stream_record_paths,
)

RESPECIFIED = "behaviour re-specified"


def _line(rec_id: int, line_num: int, jq_path: str, item: Any) -> str:
    return f"{rec_id}|{line_num}|{jq_path}|{item!r}"


def _records(json_data: str) -> list[str]:
    out = io.StringIO()
    stream_record_paths(json_data, out, _line)
    return out.getvalue().splitlines()


class _FixedAnswer(GroupAcceptStrategy):
    """A strategy that gives one scripted answer per call, in order."""

    def __init__(self, answers: list[str | None]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, bool]] = []

    def accept_path(self, path: Path, group: PathGroup, distribute: bool = False) -> str | None:
        self.calls.append((path.jq_path, distribute))
        return self.answers.pop(0)


# ---------------------------------------------------------------------------
# Path


class TestPath:
    def test_attributes(self) -> None:
        p = Path(".a[0].b", {"k": 1}, line_num=4)
        assert p.jq_path == ".a[0].b"
        assert p.item == {"k": 1}
        assert p.line_num == 4

    def test_line_num_defaults_to_untracked(self) -> None:
        assert Path(".a", 1).line_num == -1

    @pytest.mark.parametrize(
        ("jq_path", "elts"),
        [
            (".a[0].b", ["", "a[0]", "b"]),
            ("[0].a", ["[0]", "a"]),
            ("", [""]),
            ("[0][1]", ["[0][1]"]),
        ],
    )
    def test_path_elts_and_size(self, jq_path: str, elts: list[str]) -> None:
        p = Path(jq_path, None)
        assert p.path_elts == elts
        assert p.size == len(elts)

    def test_repr(self) -> None:
        assert repr(Path(".a", 1)) == ".a: 1"
        assert repr(Path(".a", 1, line_num=3)) == "3: .a: 1"
        assert repr(Path(".s", "x", line_num=0)) == "0: .s: x"

    def test_untracked_equality_is_by_path_and_item(self) -> None:
        assert Path(".a", 1) == Path(".a", 1)
        assert hash(Path(".a", 1)) == hash(Path(".a", 1))
        assert Path(".a", 1) != Path(".a", 2)
        assert Path(".a", 1) != Path(".b", 1)

    def test_tracked_equality_is_by_line_num_alone(self) -> None:
        assert Path(".a", 1, line_num=3) == Path(".b", 2, line_num=3)
        assert hash(Path(".a", 1, line_num=3)) == hash(Path(".b", 2, line_num=3))
        assert Path(".a", 1, line_num=3) != Path(".a", 1, line_num=4)

    def test_tracked_and_untracked_never_equal(self) -> None:
        assert Path(".a", 1) != Path(".a", 1, line_num=0)

    def test_comparison_with_non_path(self) -> None:
        assert Path(".a", 1).__eq__(3) is NotImplemented
        assert (Path(".a", 1) == 3) is False

    def test_ordering(self) -> None:
        # Both tracked: by line number, whatever the paths say.
        assert Path(".z", 0, line_num=1) < Path(".a", 0, line_num=2)
        assert not Path(".a", 0, line_num=2) < Path(".z", 0, line_num=1)
        # Either untracked: by jq_path.
        assert Path(".a", 0) < Path(".b", 0, line_num=0)
        assert not Path(".b", 0) < Path(".a", 0, line_num=5)
        assert sorted([Path(".c", 0), Path(".a", 0), Path(".b", 0)]) == [
            Path(".a", 0),
            Path(".b", 0),
            Path(".c", 0),
        ]


# ---------------------------------------------------------------------------
# PathGroup


class TestPathGroup:
    def test_empty_group(self) -> None:
        g = PathGroup(ArrayElementAcceptStrategy(0))
        assert g.main_paths is None
        assert g.distributed_paths is None
        assert g.num_main_paths == 0
        assert g.num_distributed_paths == 0
        assert g.size == 0
        assert g.paths == []
        assert g.as_dict() == {}

    def test_first_path_is_offered_as_main_candidate(self) -> None:
        strategy = _FixedAnswer(["main"])
        g = PathGroup(strategy, first_path=Path(".r[0]", 1, line_num=0))
        assert strategy.calls == [(".r[0]", False)]
        assert g.main_paths == {Path(".r[0]", 1, line_num=0)}
        assert g.distributed_paths is None

    def test_rejected_first_path_leaves_group_empty(self) -> None:
        g = PathGroup(_FixedAnswer([None]), first_path=Path(".r[0]", 1))
        assert g.size == 0
        assert g.main_paths is None

    def test_accept_routes_by_answer(self) -> None:
        g = PathGroup(_FixedAnswer(["distributed", "main", None, "other"]))
        assert g.accept(Path(".m", 1, line_num=0)) is True
        assert g.main_paths is None
        assert g.distributed_paths == {Path(".m", 1, line_num=0)}
        assert g.accept(Path(".r[0].a", 2, line_num=1)) is True
        assert g.accept(Path(".x", 3, line_num=2)) is False
        # Any non-None answer other than "main" is a distributed path.
        assert g.accept(Path(".o", 4, line_num=3)) is True
        assert g.num_main_paths == 1
        assert g.num_distributed_paths == 2
        assert g.size == 3

    def test_accept_passes_distribute_flag(self) -> None:
        strategy = _FixedAnswer(["distributed"])
        PathGroup(strategy).accept(Path(".m", 1), distribute=True)
        assert strategy.calls == [(".m", True)]

    def test_paths_are_sorted_and_track_accepts(self) -> None:
        g = PathGroup(_FixedAnswer(["main", "distributed", "main"]))
        g.accept(Path(".r[0].b", 2, line_num=2))
        assert g.paths == [Path(".r[0].b", 2, line_num=2)]
        g.accept(Path(".m", 0, line_num=0))
        g.accept(Path(".r[0].a", 1, line_num=1))
        assert [p.line_num for p in g.paths] == [0, 1, 2]

    def test_path_in_both_sets_listed_once(self) -> None:
        g = PathGroup(_FixedAnswer(["main", "distributed"]))
        g.accept(Path(".a", 1))
        g.accept(Path(".a", 1))
        assert g.paths == [Path(".a", 1)]
        assert g.size == 2

    def test_as_dict_rebuilds_the_object(self) -> None:
        g = PathGroup(_FixedAnswer(["distributed", "main", "main"]))
        g.accept(Path(".m", 1, line_num=0))
        g.accept(Path(".r[0].a", 2, line_num=1))
        g.accept(Path(".r[0].b", 3, line_num=2))
        assert g.as_dict() == {"m": 1, "r": [{"a": 2, "b": 3}]}

    def test_incorporate_offers_every_path_distributed(self) -> None:
        source = PathGroup(_FixedAnswer(["main", "distributed"]))
        source.accept(Path(".r[0].a", 1, line_num=1))
        source.accept(Path(".m", 0, line_num=0))
        strategy = _FixedAnswer([None, "distributed"])
        target = PathGroup(strategy)
        target.incorporate_paths(source)
        assert strategy.calls == [(".m", True), (".r[0].a", True)]
        assert target.distributed_paths == {Path(".r[0].a", 1, line_num=1)}


# ---------------------------------------------------------------------------
# ArrayElementAcceptStrategy


def _group_with_main(strategy: ArrayElementAcceptStrategy, jq_path: str) -> PathGroup:
    g = PathGroup(strategy, first_path=Path(jq_path, 0, line_num=0))
    assert g.num_main_paths == 1
    return g


class TestArrayElementAcceptStrategy:
    def test_defaults(self) -> None:
        s = ArrayElementAcceptStrategy()
        assert s.max_array_level == -1
        assert s.ref_path is None

    def test_shared_path_is_distributed(self) -> None:
        s = ArrayElementAcceptStrategy(0)
        g = PathGroup(s)
        assert s.accept_path(Path(".m", 1), g) == "distributed"
        assert s.accept_path(Path(".m", 1), g, distribute=True) == "distributed"

    @pytest.mark.xfail(strict=True, reason=f"{RESPECIFIED}: only shared context is distributed")
    def test_indexed_path_is_never_distributed(self) -> None:
        s = ArrayElementAcceptStrategy(0)
        g = _group_with_main(s, ".r[0].a")
        assert s.accept_path(Path(".r[1].a", 1), g, distribute=True) is None

    def test_first_indexed_path_becomes_reference(self) -> None:
        s = ArrayElementAcceptStrategy(0)
        g = PathGroup(s)
        p = Path(".r[0].a", 1, line_num=0)
        assert s.accept_path(p, g) == "main"
        assert s.ref_path == p

    def test_same_record_key_is_main(self) -> None:
        s = ArrayElementAcceptStrategy(0)
        g = _group_with_main(s, ".r[0].a")
        assert s.accept_path(Path(".r[0].b[3]", 1), g) == "main"
        assert s.accept_path(Path(".r[1].a", 1), g) is None
        assert s.accept_path(Path(".q[0].a", 1), g) is None

    @pytest.mark.xfail(strict=True, reason=f"{RESPECIFIED}: records are keyed by index")
    def test_top_level_array_elements_are_records(self) -> None:
        s = ArrayElementAcceptStrategy(0)
        g = _group_with_main(s, "[0].a")
        assert s.accept_path(Path("[0].b[1]", 1), g) == "main"
        assert s.accept_path(Path("[1].a", 1), g) is None

    @pytest.mark.xfail(strict=True, reason=f"{RESPECIFIED}: records are keyed by index")
    def test_scalar_elements_of_top_level_array_are_records(self) -> None:
        s = ArrayElementAcceptStrategy(0)
        g = _group_with_main(s, "[0]")
        assert s.accept_path(Path("[1]", 1), g) is None

    @pytest.mark.xfail(strict=True, reason=f"{RESPECIFIED}: records are keyed by index")
    def test_array_of_arrays_keyed_by_first_index(self) -> None:
        s = ArrayElementAcceptStrategy(0)
        g = _group_with_main(s, "[0][0]")
        assert s.accept_path(Path("[0][1]", 1), g) == "main"
        assert s.accept_path(Path("[1][0]", 1), g) is None

    def test_level_one_full_keys(self) -> None:
        s = ArrayElementAcceptStrategy(1)
        g = _group_with_main(s, ".x[0].y[0].z")
        assert s.accept_path(Path(".x[0].y[0].w", 1), g) == "main"
        assert s.accept_path(Path(".x[0].y[1].z", 1), g) is None
        assert s.accept_path(Path(".x[1].name", 1), g) is None

    @pytest.mark.xfail(strict=True, reason=f"{RESPECIFIED}: compared over the indices both have")
    def test_level_one_shorter_path_compared_over_common_indices(self) -> None:
        s = ArrayElementAcceptStrategy(1)
        g = _group_with_main(s, ".x[0].y[0].z")
        assert s.accept_path(Path(".x[0].name", 1), g) == "main"

    @pytest.mark.xfail(strict=True, reason=f"{RESPECIFIED}: a negative level does not split")
    def test_negative_level_does_not_split(self) -> None:
        s = ArrayElementAcceptStrategy(-1)
        g = _group_with_main(s, ".r[0].a")
        assert s.accept_path(Path(".r[1].a", 1), g) == "main"
        assert s.accept_path(Path(".q[5]", 1), g) == "main"


# ---------------------------------------------------------------------------
# PathSorter


def _keyed(i: int, line_num: int | None = None) -> Path:
    return Path(f".r[{i}].a", i, line_num=i if line_num is None else line_num)


class TestPathSorter:
    @pytest.mark.parametrize(
        ("given", "stored"), [(0, 0), (-4, -4), (1, 3), (2, 3), (3, 3), (7, 7)]
    )
    def test_effective_max_groups(self, given: int, stored: int) -> None:
        assert PathSorter(ArrayElementAcceptStrategy(0), max_groups=given).max_groups == stored

    def test_initial_state(self) -> None:
        s = PathSorter(ArrayElementAcceptStrategy(0))
        assert s.group_size == 0
        assert s.max_groups == 0
        assert s.groups is None
        assert s.num_groups == 0
        assert s.close_group() is None
        assert s.accept_path(_keyed(0)) is False
        assert s.all_groups_have_size(0) is False

    def test_add_path_groups_consecutive_records(self) -> None:
        s = PathSorter(ArrayElementAcceptStrategy(0))
        assert s.add_path(Path(".r[0].a", 1, line_num=0)) is None
        assert s.add_path(Path(".r[0].b", 2, line_num=1)) is None
        closed = s.add_path(Path(".r[1].a", 3, line_num=2))
        assert closed is not None
        assert [p.jq_path for p in closed.paths] == [".r[0].a", ".r[0].b"]
        assert s.num_groups == 2
        assert s.groups is not None
        assert [p.jq_path for p in s.groups[-1].paths] == [".r[1].a"]

    def test_shared_context_reaches_later_records(self) -> None:
        s = PathSorter(ArrayElementAcceptStrategy(0))
        s.add_path(Path(".meta", "m", line_num=0))
        s.add_path(Path(".r[0].a", 1, line_num=1))
        s.add_path(Path(".r[1].a", 2, line_num=2))
        last = s.close_group(check_size=False)
        assert last is not None
        assert Path(".meta", "m", line_num=0) in last.paths

    @pytest.mark.xfail(
        strict=True, reason=f"{RESPECIFIED}: a record holds no earlier record's paths"
    )
    def test_closed_record_holds_no_earlier_record(self) -> None:
        s = PathSorter(ArrayElementAcceptStrategy(0))
        for i in range(3):
            s.add_path(_keyed(i))
        closed = s.add_path(_keyed(3))
        assert closed is not None
        assert [p.jq_path for p in closed.paths] == [".r[2].a"]

    @pytest.mark.xfail(strict=True, reason=f"{RESPECIFIED}: holds at most max_groups groups")
    def test_holds_at_most_max_groups(self) -> None:
        s = PathSorter(ArrayElementAcceptStrategy(0), max_groups=3)
        counts = []
        for i in range(6):
            s.add_path(_keyed(i))
            counts.append(s.num_groups)
        assert counts == [1, 2, 3, 3, 3, 3]

    def test_unlimited_groups(self) -> None:
        s = PathSorter(ArrayElementAcceptStrategy(0))
        for i in range(6):
            s.add_path(_keyed(i))
        assert s.num_groups == 6

    def test_close_group_keeps_large_enough_group(self) -> None:
        s = PathSorter(ArrayElementAcceptStrategy(0), group_size=2)
        s.add_path(Path(".r[0].a", 1, line_num=0))
        s.add_path(Path(".r[0].b", 2, line_num=1))
        closed = s.close_group()
        assert closed is not None
        assert closed.size == 2
        assert s.num_groups == 1

    @pytest.mark.xfail(
        strict=True, reason=f"{RESPECIFIED}: an undersized group is dropped and None returned"
    )
    def test_close_group_drops_undersized_group(self) -> None:
        s = PathSorter(ArrayElementAcceptStrategy(0), group_size=2)
        s.add_path(Path(".r[0].a", 1, line_num=0))
        assert s.close_group() is None
        assert s.num_groups == 0

    @pytest.mark.xfail(strict=True, reason=f"{RESPECIFIED}: the group at idx is the one dropped")
    def test_close_group_drops_the_group_at_idx(self) -> None:
        s = PathSorter(ArrayElementAcceptStrategy(0))
        s.add_path(Path(".r[0].a", 1, line_num=0))
        s.add_path(Path(".r[1].a", 2, line_num=1))
        s.add_path(Path(".r[1].b", 3, line_num=2))
        assert s.groups is not None
        assert [g.size for g in s.groups] == [1, 2]
        newest = s.groups[-1]
        s.group_size = 2
        assert s.close_group(idx=0) is None
        assert s.groups == [newest]

    @pytest.mark.xfail(
        strict=True, reason=f"{RESPECIFIED}: an undersized group is dropped and None returned"
    )
    def test_add_path_returns_none_when_closed_group_dropped(self) -> None:
        s = PathSorter(ArrayElementAcceptStrategy(0), group_size=2)
        s.add_path(Path(".r[0].a", 1, line_num=0))
        assert s.add_path(Path(".r[1].a", 2, line_num=1)) is None

    def test_close_group_without_size_check(self) -> None:
        s = PathSorter(ArrayElementAcceptStrategy(0), group_size=5)
        s.add_path(Path(".r[0].a", 1, line_num=0))
        closed = s.close_group(check_size=False)
        assert closed is not None
        assert s.num_groups == 1

    def test_accept_path_tries_held_groups(self) -> None:
        s = PathSorter(ArrayElementAcceptStrategy(0))
        s.add_path(_keyed(0))
        assert s.accept_path(Path(".r[0].z", 9, line_num=9)) is True
        assert s.accept_path(Path(".q[0].z", 9, line_num=10)) is False

    def test_all_groups_have_size(self) -> None:
        s = PathSorter(ArrayElementAcceptStrategy(0))
        s.add_path(Path(".r[0].a", 1, line_num=0))
        assert s.all_groups_have_size(1) is True
        assert s.all_groups_have_size(2) is False
        s.groups = []
        assert s.all_groups_have_size(7) is True


# ---------------------------------------------------------------------------
# RecordPathBuilder and stream_record_paths


class TestStreamRecordPaths:
    def test_document_without_index_is_one_record(self) -> None:
        assert _records('{"a":1,"b":2}') == ["0|0|.a|1", "0|1|.b|2"]

    def test_empty_array(self) -> None:
        assert _records("[]") == ["0|0||[]"]

    @pytest.mark.xfail(strict=True, reason=f"{RESPECIFIED}: one record per top-level array element")
    def test_top_level_array_of_objects(self) -> None:
        assert _records('[{"a":1,"b":[2,3]},{"a":4}]') == [
            "0|0|[0].a|1",
            "0|1|[0].b[0]|2",
            "0|2|[0].b[1]|3",
            "1|3|[1].a|4",
        ]

    @pytest.mark.xfail(
        strict=True,
        reason=f"{RESPECIFIED}: a record holds its own values and earlier shared context",
    )
    def test_shared_context(self) -> None:
        assert _records('{"meta":"x","items":[{"a":1},{"a":2}],"tail":5}') == [
            "0|0|.meta|'x'",
            "0|1|.items[0].a|1",
            "1|0|.meta|'x'",
            "1|2|.items[1].a|2",
            "1|3|.tail|5",
        ]

    @pytest.mark.xfail(
        strict=True, reason=f"{RESPECIFIED}: a record holds no earlier record's paths"
    )
    def test_nested_arrays_stay_in_their_record(self) -> None:
        assert _records('{"x":[{"y":[1,2]},{"y":[3]}]}') == [
            "0|0|.x[0].y[0]|1",
            "0|1|.x[0].y[1]|2",
            "1|2|.x[1].y[0]|3",
        ]

    @pytest.mark.xfail(strict=True, reason=f"{RESPECIFIED}: one record per top-level array element")
    def test_top_level_array_of_scalars(self) -> None:
        assert _records("[1,2,3]") == ["0|0|[0]|1", "1|1|[1]|2", "2|2|[2]|3"]

    @pytest.mark.xfail(strict=True, reason=f"{RESPECIFIED}: records are keyed by index")
    def test_array_of_arrays(self) -> None:
        assert _records("[[1,2],[3]]") == ["0|0|[0][0]|1", "0|1|[0][1]|2", "1|2|[1][0]|3"]

    def test_module_function_matches_builder(self) -> None:
        doc = '{"meta":"x","items":[{"a":1},{"a":2}]}'
        out = io.StringIO()
        RecordPathBuilder(doc, out, _line).stream_record_paths()
        assert out.getvalue().splitlines() == _records(doc)

    def test_builder_attributes(self) -> None:
        out = io.StringIO()
        rpb = RecordPathBuilder('{"a":1}', out, _line, timeout=3)
        assert rpb.jdata == '{"a":1}'
        assert rpb.output_stream is out
        assert rpb.builder_fn is _line
        assert rpb.timeout == 3
        assert rpb.rec_id == 0
        assert rpb.inum == 0
        assert rpb.sorter is None

    def test_counters_after_streaming(self) -> None:
        out = io.StringIO()
        rpb = RecordPathBuilder('{"x":[{"y":[1,2]},{"y":[3]}]}', out, _line)
        rpb.stream_record_paths()
        assert rpb.inum == 3
        assert rpb.rec_id == 1
        assert isinstance(rpb.sorter, PathSorter)
        assert isinstance(rpb.sorter.accept_strategy, ArrayElementAcceptStrategy)
        assert rpb.sorter.accept_strategy.max_array_level == 0
        assert rpb.sorter.max_groups == 3

    def test_streaming_twice_starts_over(self) -> None:
        out = io.StringIO()
        rpb = RecordPathBuilder('{"a":1,"b":2}', out, _line)
        rpb.stream_record_paths()
        first = out.getvalue()
        rpb.stream_record_paths()
        assert out.getvalue() == first + first
        assert rpb.inum == 2

    def test_record_written_when_it_ends(self) -> None:
        out = io.StringIO()
        rpb = RecordPathBuilder("unused", out, _line)
        rpb.sorter = PathSorter(ArrayElementAcceptStrategy(max_array_level=0), max_groups=2)
        rpb.visitor(1, ("items", 0, "a"))
        assert out.getvalue() == ""
        rpb.visitor(2, ("items", 1, "a"))
        assert out.getvalue().splitlines() == ["0|0|.items[0].a|1"]
        assert rpb.rec_id == 1
        assert rpb.inum == 2

    def test_visitor_without_sorter_only_counts(self) -> None:
        out = io.StringIO()
        rpb = RecordPathBuilder("unused", out, _line)
        rpb.visitor(1, ("a",))
        assert out.getvalue() == ""
        assert rpb.inum == 1

    def test_write_group_uses_current_rec_id(self) -> None:
        out = io.StringIO()
        rpb = RecordPathBuilder("unused", out, _line)
        rpb.rec_id = 7
        g = PathGroup(ArrayElementAcceptStrategy(0))
        g.accept(Path(".b", 2, line_num=1))
        g.accept(Path(".a", 1, line_num=0))
        rpb.write_group(g)
        assert out.getvalue().splitlines() == ["7|0|.a|1", "7|1|.b|2"]

    def test_file_source(self, tmp_path: Any) -> None:
        f = tmp_path / "doc.json"
        f.write_text('{"a":1,"b":2}', encoding="utf-8")
        assert _records(str(f)) == ["0|0|.a|1", "0|1|.b|2"]


class TestGetRecordsDf:
    def test_single_record(self) -> None:
        df = get_records_df('{"a":1,"b":2}')
        assert list(df.columns) == ["rec_id", "line_num", "jq_path", "item"]
        assert df.values.tolist() == [[0, 0, ".a", 1], [0, 1, ".b", 2]]

    @pytest.mark.xfail(strict=True, reason=f"{RESPECIFIED}: one record per top-level array element")
    def test_records(self) -> None:
        df = get_records_df('[{"a":1},{"a":2}]')
        assert df.values.tolist() == [[0, 0, "[0].a", 1], [1, 1, "[1].a", 2]]
