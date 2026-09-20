"""Every public door on the Postgres backends answers the same way unconnected.

Needs no server: the failures under test happen before any socket is opened,
on a freshly constructed object whose pool (async) or ``PostgresDB`` handle
(sync) is still ``None``.

Reproduce-first for the ``enable_vector_support`` pair. Both twins reached a
connection-bearing call without checking that the connection existed, and each
failed in its own way: the async one dereferenced ``None`` and the sync one
swallowed that dereference into a ``False`` return, reporting "no vector
support" for a database whose vector support it never got to look at.
"""

import ast
import inspect
import re

import pytest

from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase

CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "unused_db",
    "user": "unused_user",
    "password": "unused_pass",
    "table": "unused_table",
}

NOT_CONNECTED = re.escape("Database not connected. Call connect() first.")


@pytest.mark.asyncio
async def test_async_enable_vector_support_refuses_like_every_other_door():
    """``enable_vector_support()`` gives the named error, not ``AttributeError``.

    It awaited ``_detect_vector_support()``, which opened
    ``async with self._pool.acquire()`` with no check on either side, so an
    unconnected database answered ``AttributeError: 'NoneType' object has no
    attribute 'acquire'`` -- naming neither the class nor the missing
    ``connect()``.

    ``read()`` is the positive control: it is the behaviour the rest of the
    class already had, measured on the same object, so a green assertion here
    is about this door rather than about the message existing at all.
    """
    db = AsyncPostgresDatabase(CONFIG)

    with pytest.raises(RuntimeError, match=NOT_CONNECTED):
        await db.read("any-id")

    with pytest.raises(RuntimeError, match=NOT_CONNECTED):
        await db.enable_vector_support()


def test_sync_enable_vector_support_refuses_rather_than_reporting_no_support():
    """The sync twin raises too, instead of returning a ``False`` it cannot know.

    ``_detect_vector_support`` wraps its probe in ``except Exception`` so that a
    database genuinely lacking pgvector answers ``False``. An unconnected one
    took the same branch: the ``AttributeError`` from ``self.db`` being ``None``
    was caught, logged as "Could not install pgvector extension", and reported
    as a capability answer.

    A missing extension is still ``False``; a missing connection is not.
    """
    db = SyncPostgresDatabase(CONFIG)

    with pytest.raises(RuntimeError, match=NOT_CONNECTED):
        db.read("any-id")

    with pytest.raises(RuntimeError, match=NOT_CONNECTED):
        db.enable_vector_support()


def test_async_pool_is_reached_only_through_the_accessor():
    """No method of ``AsyncPostgresDatabase`` acquires off the raw attribute.

    The recurrence guard. ``_check_async_connection`` tests the pool through
    ``getattr(self, "_pool", None)`` on a mixin that never declares it, so no
    narrowing reaches the use sites and the type checker cannot tell a site
    that checked from one that did not -- which is how twenty-five guarded
    sites and one unguarded one read identically for as long as they did.

    Routing every acquire through ``_require_pool()`` is what makes the
    invariant checkable, and this is what keeps the twenty-sixth site from
    being written the old way.
    """
    tree = ast.parse(inspect.getsource(AsyncPostgresDatabase))

    raw = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "acquire"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "_pool"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
    ]

    assert raw == [], (
        "self._pool.acquire() at class-relative line(s) "
        f"{raw} -- use self._require_pool().acquire() so the site states the "
        "invariant the checker cannot follow from _check_connection()."
    )
