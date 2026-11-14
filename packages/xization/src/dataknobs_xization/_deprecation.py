# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""What the deprecated authority stack says, written once.

The stack -- :mod:`~dataknobs_xization.authorities`,
:mod:`~dataknobs_xization.lexicon`, :mod:`~dataknobs_xization.masking_tokenizer`
and the half of :mod:`~dataknobs_xization.annotations` nothing outside it uses
-- is removed in the next major release. Each warning is raised from more than
one place: a module warns when it is imported, the package's lazy door warns
when it reaches that module, and ``annotations`` warns when a deprecated name
is reached through it. Holding the text here is what keeps the places saying
the same thing.

Every function warns with ``stacklevel=3``: one frame for itself and one for
the module body or ``__getattr__`` that calls it. For an ``import`` or ``from``
statement and for attribute access, that attributes the warning to the
caller's own line, because the import system's bootstrap frames are skipped.
That attribution is not cosmetic. Python's default filter shows a
``DeprecationWarning`` only when it is attributed to ``__main__``, so one
attributed to this package would never be seen outside a test run.
``importlib.import_module`` is the exception: its own frame is not skipped, so
a module imported through it -- by a dotted-path loader, say -- warns from
``importlib``, and the default filter hides that.

When the package door imports a module, the door has already warned, so that
module's own import-time warning is withheld (:func:`door_import`). Only that
module's: a deprecated module it imports in turn still warns, and nothing else
raised during the import is touched.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_REMOVAL = "will be removed in the next major release"

_DICTIONARY = (
    "For matching a dictionary of entity names in text, use ScanningSignal (from "
    "dataknobs_common.entity_resolution) over an EntitySource (from "
    "dataknobs_common.ontology)"
)

_TOKENS = "For token boundaries, use dataknobs_common.text.token_spans."

#: Each deprecated module's replacement, or what has none.
_MODULE_ADVICE = {
    "authorities": (
        f"{_DICTIONARY}. Regex authorities and annotation validators have no replacement yet."
    ),
    "lexicon": (
        f"{_DICTIONARY}, and declare each variation as a form of its entity. An "
        "ontology document's variants: section generates them, with the rules in "
        "dataknobs_common.text."
    ),
    "masking_tokenizer": _TOKENS,
}

#: The deprecated names of ``annotations``, which live in
#: ``_annotations_deprecated`` and are reached through ``annotations``.
DEPRECATED_ANNOTATION_NAMES = (
    "RowData",
    "AnnotationsGroup",
    "MergeStrategy",
    "PositionalAnnotationsGroup",
    "OverlapGroupIterator",
    "merge",
    "AnnotationsGroupList",
    "BasicAnnotator",
    "SyntacticParser",
    "EntityAnnotator",
    "HtmlHighlighter",
    "AnnotatorKernel",
    "CompoundAnnotator",
)

_NO_DF = "It does not work: it reads Annotations.as_df, which does not exist."

#: The deprecated names that are also broken. Said in the warning, so that a
#: reader is not left to find it out; deprecated rather than fixed, and a
#: consumer who needs one before its removal gets a fix and a test then.
_ANNOTATION_DEFECTS = {
    "merge": _NO_DF,
    "BasicAnnotator": _NO_DF.replace("it reads", "its annotate_input reads"),
    "SyntacticParser": _NO_DF.replace("it reads", "its annotate_input reads"),
    "EntityAnnotator": (
        "It does not work: the annotate_input it inherits from BasicAnnotator reads "
        "Annotations.as_df, which does not exist."
    ),
    "HtmlHighlighter": (
        "It does not work: it drops the text after the last highlight, repeats the "
        "text once per highlighted column, and does not escape HTML."
    ),
}

#: What the package door reaches lazily, mapped to the module holding it.
PACKAGE_DOOR = {
    "authorities": "authorities",
    "lexicon": "lexicon",
    "masking_tokenizer": "masking_tokenizer",
    "CharacterFeatures": "masking_tokenizer",
    "TextFeatures": "masking_tokenizer",
}


def module_message(module: str) -> str:
    """The warning a deprecated module raises when it is imported."""
    return f"dataknobs_xization.{module} is deprecated and {_REMOVAL}. {_MODULE_ADVICE[module]}"


def door_message(name: str) -> str:
    """The warning the package door raises when it reaches ``name``."""
    module = PACKAGE_DOOR[name]
    if name == module:
        return module_message(module)
    return (
        f"dataknobs_xization.{name} is deprecated and {_REMOVAL}, with "
        f"dataknobs_xization.{module}, where it lives. {_MODULE_ADVICE[module]}"
    )


def annotation_message(name: str) -> str:
    """The warning ``annotations`` raises when a deprecated name is reached."""
    defect = _ANNOTATION_DEFECTS.get(name)
    said = f" {defect}" if defect else ""
    return (
        f"dataknobs_xization.annotations.{name} is deprecated and {_REMOVAL}.{said} "
        "It has no replacement."
    )


#: The modules the package door is importing in this context, whose own
#: import-time warning it has already given.
_DOOR_IMPORTING: ContextVar[frozenset[str]] = ContextVar("_DOOR_IMPORTING", default=frozenset())


@contextmanager
def door_import(module: str) -> Iterator[None]:
    """Withhold ``module``'s own import-time warning while the door imports it.

    A context variable rather than a warnings filter: ``catch_warnings``
    mutates process-wide filter state, and a filter broad enough to be simple
    also swallows the warnings of the deprecated modules ``module`` imports,
    and anything a dependency raises.
    """
    token = _DOOR_IMPORTING.set(_DOOR_IMPORTING.get() | {module})
    try:
        yield
    finally:
        _DOOR_IMPORTING.reset(token)


def warn_module(module: str) -> None:
    """Warn, from a deprecated module's body, that it is deprecated."""
    if module in _DOOR_IMPORTING.get():
        return
    warnings.warn(module_message(module), DeprecationWarning, stacklevel=3)


def warn_door(name: str) -> None:
    """Warn, from the package's ``__getattr__``, that ``name`` is deprecated."""
    warnings.warn(door_message(name), DeprecationWarning, stacklevel=3)


def warn_annotation_name(name: str) -> None:
    """Warn, from ``annotations.__getattr__``, that ``name`` is deprecated."""
    warnings.warn(annotation_message(name), DeprecationWarning, stacklevel=3)
