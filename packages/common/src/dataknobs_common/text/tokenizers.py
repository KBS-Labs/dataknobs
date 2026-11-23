# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Where a scan's tokens come from, by a name a configuration can write.

A **tokenizer** takes a string and returns its tokens as half-open spans into
it, in order and not overlapping -- the shape :func:`~dataknobs_common.text.token_spans`
returns and :func:`~dataknobs_common.text.probe_spans` takes as ``tokens``.
The scanning rungs take one as ``tokenizer=``, so a vocabulary whose forms are
not runs of letters, digits and their marks -- a currency sign, a grouped
number, a script written without spaces -- can be scanned over units that
reach them.

A rung takes the callable itself. A document cannot write a callable, so a
configuration names one instead, and :data:`TOKENIZERS` is where the name is
looked up. Two entries ship:

- ``default`` is :func:`~dataknobs_common.text.token_spans` itself rather than
  a wrapper around it: a rung asks whether its tokenizer *is* that function to
  decide whether the source's own measurements still bound its windows, so
  identity is the property, not a detail.
- ``words`` is :func:`~dataknobs_common.text.word_spans`: the same tokens with
  an in-word apostrophe kept, for a reader that takes a token as a word.
  ``{"name": "words", "apostrophes": "clitic"}`` splits *don't* as ``don``
  ``'t`` instead. A rung over it takes the general path, since it is not
  ``token_spans``.

Register your own with a factory taking that tokenizer's parameters -- the
spec's keys other than ``name`` -- and returning the tokenizer::

    from dataknobs_common.text import TOKENIZERS

    def _symbols(config):
        # ... read parameters from ``config``, refuse unknown ones ...
        return my_tokenizer

    TOKENIZERS.register("symbols", _symbols)

A separate module from :mod:`~dataknobs_common.text.scan` because that one
imports only the standard library, and a registry is an import.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, TypeAlias

from dataknobs_common.exceptions import OperationError, ValidationError, _written_by_dataknobs
from dataknobs_common.registry import PluginRegistry, _shown_failure
from dataknobs_common.text.scan import _APOSTROPHE_MODES, token_spans, word_spans

__all__ = ["TOKENIZERS", "Tokenizer", "tokenizer_from_config"]

#: A string in, its tokens out: half-open spans into the string, in order, not
#: overlapping, none empty. Spans may touch.
Tokenizer: TypeAlias = Callable[[str], Sequence[tuple[int, int]]]


def _default_factory(config: Mapping[str, Any]) -> Tokenizer:
    if config:
        raise _written_by_dataknobs(ValueError(f"takes no parameters; given {sorted(config)}"))
    return token_spans


def _words_factory(config: Mapping[str, Any]) -> Tokenizer:
    unknown = sorted(set(config) - {"apostrophes"})
    if unknown:
        raise _written_by_dataknobs(ValueError(f"takes only 'apostrophes'; given {unknown}"))
    apostrophes = config.get("apostrophes", "keep")
    if apostrophes == "keep":
        return word_spans
    # word_spans refuses an unknown mode, so check it now rather than on the
    # first string the tokenizer is handed -- here, where the refusal is
    # marked as dataknobs' text and so reaches the author through a registry.
    if apostrophes not in _APOSTROPHE_MODES:
        raise _written_by_dataknobs(
            ValueError(
                f"apostrophes must be one of {list(_APOSTROPHE_MODES)}; given {apostrophes!r}"
            )
        )
    return partial(word_spans, apostrophes=apostrophes)


#: Every tokenizer a configuration can name, by name. Each entry is a factory
#: taking the tokenizer's parameters -- the spec's keys other than ``name`` --
#: and returning a :data:`Tokenizer`.
TOKENIZERS: PluginRegistry[Tokenizer] = PluginRegistry(
    "tokenizers",
    config_key="name",
    strip_config_key=True,
    not_found_kind="tokenizer",
)
TOKENIZERS.register("default", _default_factory)
TOKENIZERS.register("words", _words_factory)


def tokenizer_from_config(
    spec: str | Mapping[str, Any] | Tokenizer | None,
) -> Tokenizer | None:
    """The tokenizer a configuration names, built.

    - ``None`` stays ``None``: the rung's default.
    - A callable is returned as given, for a configuration built in code.
    - A string is a name in :data:`TOKENIZERS`.
    - A mapping names one under ``name``, with its parameters beside it:
      ``{"name": "symbols", "keep": "$"}``.

    Args:
        spec: What the configuration's ``tokenizer`` key holds.

    Returns:
        The tokenizer, or ``None``.

    Raises:
        ValidationError: For a spec of any other shape, a name
            :data:`TOKENIZERS` does not hold, parameters the named factory
            refuses, or a factory that returns something not callable. The
            message names the tokenizer and an unknown name lists the
            registered ones.
    """
    if spec is None:
        return None
    if callable(spec):
        return spec
    config = {"name": spec} if isinstance(spec, str) else spec
    name = config.get("name") if isinstance(config, Mapping) else None
    if not isinstance(name, str) or not name:
        raise _written_by_dataknobs(
            ValidationError(
                f"tokenizer must be a tokenizer name, a mapping with a 'name' naming "
                f"one, or a callable; given {spec!r}",
                context={"spec": spec},
            )
        )
    if not TOKENIZERS.is_registered(name):
        raise _written_by_dataknobs(
            ValidationError(
                f"unknown tokenizer {name!r}. Registered tokenizers: "
                f"{sorted(TOKENIZERS.list_keys())}",
                context={"tokenizer": name},
            )
        )
    try:
        tokenizer = TOKENIZERS.create(config=dict(config))
    except OperationError as error:
        # A consumer factory's text may carry what it was handed, so unless
        # dataknobs wrote it, it is named by type and kept on ``__cause__``.
        reason = _shown_failure(error)
        if reason is None:
            # Names only the tokenizer and the error's type, so it is ours.
            raise _written_by_dataknobs(
                ValidationError(
                    f"tokenizer {name!r} could not be built ({type(error.__cause__).__name__})",
                    context={"tokenizer": name},
                )
            ) from error
        raise _written_by_dataknobs(
            ValidationError(f"tokenizer {name!r} {reason}", context={"tokenizer": name})
        ) from error
    if not callable(tokenizer):
        raise _written_by_dataknobs(
            ValidationError(
                f"tokenizer {name!r}'s factory returned a {type(tokenizer).__name__}, "
                f"which is not callable; a factory returns a tokenizer",
                context={"tokenizer": name},
            )
        )
    return tokenizer
