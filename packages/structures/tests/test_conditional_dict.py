from dataknobs_structures.conditional_dict import cdict


def test_basics():
    # only accept the first value for any key
    cd = cdict(lambda d, k, v: k not in d)
    cd["a"] = 1
    cd["b"] = 2
    assert cd == {"a": 1, "b": 2}
    assert len(cd.rejected) == 0
    cd["a"] = 3
    assert cd == {"a": 1, "b": 2}
    assert cd.rejected == {"a": 3}


def test_enforce_constraints_on_construction():
    cd = cdict(lambda d, k, v: v != 3, {"a": 1, "b": 3, "c": 2})
    assert cd == {"a": 1, "c": 2}
    assert cd.rejected == {"b": 3}


def test_enforce_constraints_on_update():
    cd = cdict(lambda d, k, v: v != 3)
    cd.update({"a": 1, "b": 3, "c": 2})
    assert cd == {"a": 1, "c": 2}
    assert cd.rejected == {"b": 3}


def test_enforce_constraints_on_setdefault():
    cd = cdict(lambda d, k, v: v != 3, {"a": 1, "b": 3, "c": 2})
    assert cd.rejected == {"b": 3}
    assert cd.setdefault("a", 5) == 1
    assert cd.setdefault("c", 3) == 2
    assert cd.setdefault("d", 3) is None
    assert cd == {"a": 1, "c": 2}
    assert cd.rejected == {"b": 3, "d": 3}
