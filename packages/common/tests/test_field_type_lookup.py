# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""``FieldType.lookup``: the one reading of a declared record type.

Two readers took a type word -- the ontology loader here and
``dataknobs-data``'s schema reader -- and each folded case with its own
``.lower()``. They had already drifted: one took the member itself and the
other refused it. Both now ask this.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_common.fields import FieldType


@pytest.mark.parametrize("member", list(FieldType), ids=[m.value for m in FieldType])
def test_a_member_is_itself(member: FieldType) -> None:
    assert FieldType.lookup(member) is member


@pytest.mark.parametrize("member", list(FieldType), ids=[m.value for m in FieldType])
def test_a_name_is_its_member_whatever_its_case(member: FieldType) -> None:
    assert FieldType.lookup(member.value) is member
    assert FieldType.lookup(member.value.upper()) is member
    assert FieldType.lookup(member.value.title()) is member


@pytest.mark.parametrize(
    "value",
    [None, "", "number", "entity", "fieldtype.integer", 5, 1.5, ["string"], b"string"],
    ids=[
        "none",
        "empty",
        "vocabulary-number",
        "vocabulary-entity",
        "member-str",
        "int",
        "float",
        "list",
        "bytes",
    ],
)
def test_anything_else_names_no_member(value: Any) -> None:
    """``None`` rather than a refusal: whether that is an error is the caller's question.

    The ontology loader keeps ``type:`` open, so ``number`` is a word with no
    record counterpart; the schema reader refuses it. The member's own
    ``str()`` is refused too, which is the spelling a ``str(value).lower()``
    reader turned a member into.
    """
    assert FieldType.lookup(value) is None
