"""Unit tests for AsyncPostgresDatabase._build_vector_params.

_build_vector_params is a pure static method — no DB connection needed.
It translates [(quoted_col, vec_str), ...] into the three parallel lists
(columns, placeholders, values) that create/update/upsert splice into SQL.
"""

import pytest

from dataknobs_data.backends.postgres import AsyncPostgresDatabase


class TestBuildVectorParams:
    def test_empty_list(self):
        cols, placeholders, values = AsyncPostgresDatabase._build_vector_params(
            [], start_param=4
        )
        assert cols == []
        assert placeholders == []
        assert values == []

    def test_single_lowercase_field(self):
        inserts = [('"vector_embedding"', "[0.1,0.2,0.3]")]
        cols, placeholders, values = AsyncPostgresDatabase._build_vector_params(
            inserts, start_param=4
        )
        assert cols == ['"vector_embedding"']
        assert placeholders == ["$4::vector"]
        assert values == ["[0.1,0.2,0.3]"]

    def test_single_mixed_case_field_preserves_quoting(self):
        """Quoted column name must pass through unchanged — quoting already applied."""
        inserts = [('"vector_MyEmbedding"', "[0.1,0.2,0.3,0.4]")]
        cols, placeholders, values = AsyncPostgresDatabase._build_vector_params(
            inserts, start_param=4
        )
        assert cols == ['"vector_MyEmbedding"']
        assert placeholders == ["$4::vector"]
        assert values == ["[0.1,0.2,0.3,0.4]"]

    def test_multiple_fields_param_numbering(self):
        """Each successive field gets the next $N index."""
        inserts = [
            ('"vector_fieldA"', "[0.1]"),
            ('"vector_FieldB"', "[0.2]"),
            ('"vector_field_c"', "[0.3]"),
        ]
        cols, placeholders, values = AsyncPostgresDatabase._build_vector_params(
            inserts, start_param=4
        )
        assert placeholders == ["$4::vector", "$5::vector", "$6::vector"]
        assert cols == ['"vector_fieldA"', '"vector_FieldB"', '"vector_field_c"']
        assert values == ["[0.1]", "[0.2]", "[0.3]"]

    def test_start_param_respected(self):
        """start_param controls the first $N — callers may have variable base args."""
        inserts = [('"vector_x"', "[1.0]")]
        _, placeholders, _ = AsyncPostgresDatabase._build_vector_params(
            inserts, start_param=1
        )
        assert placeholders == ["$1::vector"]

        _, placeholders, _ = AsyncPostgresDatabase._build_vector_params(
            inserts, start_param=7
        )
        assert placeholders == ["$7::vector"]

    def test_placeholder_always_has_vector_cast(self):
        """Every placeholder must end with ::vector regardless of field name."""
        inserts = [('"vector_foo"', "[0.0]")]
        _, placeholders, _ = AsyncPostgresDatabase._build_vector_params(
            inserts, start_param=4
        )
        assert all(p.endswith("::vector") for p in placeholders)
