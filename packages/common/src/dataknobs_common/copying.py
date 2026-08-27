"""Copy nested containers without duplicating what they hold.

One function, for the case that sits between ``dict(value)`` and
``copy.deepcopy(value)``: a hand-out whose *structure* must be the
caller's own while its *values* stay whatever the source had.

See :func:`copy_structure` for when each of the three is right.
"""

from typing import Any

__all__ = ["copy_structure"]

# The keep-alive list is filed under the memo's *own* id, which is what
# ``copy.deepcopy`` does. It cannot collide with the id of anything being
# copied -- the memo is not part of the structure -- and it keeps the memo
# honestly typed as ``dict[int, Any]`` rather than widening it to admit a
# sentinel key of another type.


def copy_structure(value: Any, seen: dict[int, Any] | None = None) -> Any:
    """Rebuild nested dicts and lists; pass every other value through.

    A shallow ``dict(value)`` isolates only the top level: every nested
    container in the result is still the source's own object, so a
    consumer adjusting a nested section writes back into a structure that
    outlives the hand-out. That is the defect this function exists to
    prevent, and it is the reason a "returns a copy" docstring is so
    often wrong about what it returns.

    **``copy.deepcopy`` would overshoot.** A structure assembled in Python
    may hold a live object -- a connection pool, a prebuilt provider, a
    lock -- and duplicating one silently gives its owner a second pool,
    while a value that cannot be pickled raises out of what is meant to
    be an ordinary read. Neither is a risk the aliasing this copy
    prevents justifies taking. It is also cheaper: on a wizard stage
    config, roughly half the cost of ``deepcopy``.

    So the boundary is: **containers are rebuilt, leaves are shared.**
    Mutating the returned structure never reaches the source; mutating a
    leaf object reachable from it still does, because that object was
    never copied. When the leaves are immutable -- the ordinary case for
    configuration loaded from YAML or JSON -- the result is a full
    isolation at a fraction of the price.

    A ``set`` is the leaf worth naming, because it is the mutable one. It
    passes through like any other non-container, so ``add()`` on a set
    reachable from the result reaches the source. Nothing nests below it
    -- a set cannot hold a dict or a list -- but the set itself is shared,
    and a caller who will be adding to one should rebuild it.

    ``seen`` is the memo, kept from ``deepcopy`` for the same reason
    ``deepcopy`` has one. A container reached twice is copied once and
    the same copy used both times, so a structure that refers back to
    itself terminates instead of raising ``RecursionError``, and one that
    merely shares a subtree between two keys keeps sharing it. A caller
    assembling a single hand-out from several values should pass one memo
    to each call, so the result's sharing reflects the source's; a caller
    copying one value omits it.

    The memo also holds a reference to every source it has seen, which
    ``deepcopy`` does through ``_keep_alive`` and for the same reason: it
    is keyed on ``id()``, and an id is only unique among *live* objects.
    Without this, a source freed between two calls sharing one memo can
    have its id reused by the next -- and the memo then answers for an
    unrelated object, returning the earlier call's copy. Not theoretical:
    two successive calls passing a temporary dict literal reproduce it.

    Args:
        value: The value to copy. Any type; only ``dict`` and ``list``
            are rebuilt.
        seen: Optional memo mapping ``id()`` to the copy already made for
            it. Pass the same dict across several calls that build one
            hand-out; omit it otherwise.

    Returns:
        A structure whose dicts and lists are new objects and whose other
        values are the originals.

    Example:
        ```python
        from dataknobs_common import copy_structure

        source = {"schema": {"type": "object"}, "prompt": "Name?"}
        handed_out = copy_structure(source)

        handed_out["schema"]["type"] = "array"
        assert source["schema"]["type"] == "object"   # not reached

        # dict() would not have held:
        shallow = dict(source)
        shallow["schema"]["type"] = "array"
        assert source["schema"]["type"] == "array"    # reached
        ```
    """
    if seen is None:
        seen = {}
    marker = id(value)
    if marker in seen:
        return seen[marker]

    if isinstance(value, (dict, list)):
        # Hold the source so its id cannot be reused while the memo
        # lives; see the note on ``seen`` above.
        seen.setdefault(id(seen), []).append(value)

    if isinstance(value, dict):
        copied_dict: dict[Any, Any] = {}
        # Registered before recursing, so a self-reference finds it.
        seen[marker] = copied_dict
        for key, item in value.items():
            copied_dict[key] = copy_structure(item, seen)
        return copied_dict
    if isinstance(value, list):
        copied_list: list[Any] = []
        seen[marker] = copied_list
        for item in value:
            copied_list.append(copy_structure(item, seen))
        return copied_list
    return value
