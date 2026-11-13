"""The authority stack warns, and nothing else in the package does.

``authorities``, ``lexicon`` and ``masking_tokenizer`` are deprecated whole, as
is the half of ``annotations`` nothing outside it uses. Each is removed in the
next major release. A consumer must hear that from every way in:

* importing a deprecated module, by any spelling;
* reaching one through the package door, ``dataknobs_xization.<name>``,
  including ``CharacterFeatures`` and ``TextFeatures``;
* reaching a deprecated name in ``annotations``, which is how a subclass or an
  ``isinstance`` check reaches it too.

And ``import dataknobs_xization`` must stay silent, which it could not while
the package imported the stack eagerly. That one is the control: it passes
before and after, and it is what the lazy door exists to keep true.
"""

import importlib
import subprocess
import sys
import warnings
from collections.abc import Iterator

import pytest

import dataknobs_xization
from dataknobs_xization import annotations

PACKAGE = "dataknobs_xization"
DEPRECATED_MODULES = ("authorities", "lexicon", "masking_tokenizer")

#: Every name the deprecated half of ``annotations`` held, each with what its
#: message must say beyond the removal.
DEPRECATED_ANNOTATION_NAMES = {
    "RowData": "no replacement",
    "AnnotationsGroup": "no replacement",
    "MergeStrategy": "no replacement",
    "PositionalAnnotationsGroup": "no replacement",
    "OverlapGroupIterator": "no replacement",
    "merge": "does not work",
    "AnnotationsGroupList": "no replacement",
    "BasicAnnotator": "does not work",
    "SyntacticParser": "does not work",
    "EntityAnnotator": "does not work",
    "HtmlHighlighter": "does not work",
    "AnnotatorKernel": "no replacement",
    "CompoundAnnotator": "no replacement",
}


@pytest.fixture
def fresh_stack() -> Iterator[None]:
    """Evict the package's modules, and put the originals back afterwards.

    A module warns once, when it is first executed, so each test needs to be
    the first. Restoring the original objects matters as much: the other test
    modules hold references to them, and a class re-created by a second import
    is not the class those tests ``isinstance``-check against.
    """
    saved = {k: v for k, v in sys.modules.items() if k.startswith(f"{PACKAGE}.")}
    saved_attrs = {
        name: getattr(dataknobs_xization, name)
        for name in DEPRECATED_MODULES
        if name in vars(dataknobs_xization)
    }
    for name in saved:
        if name.split(".")[1] in DEPRECATED_MODULES:
            del sys.modules[name]
    for name in saved_attrs:
        delattr(dataknobs_xization, name)
    try:
        yield
    finally:
        for name in [k for k in sys.modules if k.startswith(f"{PACKAGE}.")]:
            if name not in saved:
                del sys.modules[name]
        sys.modules.update(saved)
        for name in DEPRECATED_MODULES:
            if name in vars(dataknobs_xization) and name not in saved_attrs:
                delattr(dataknobs_xization, name)
        for name, value in saved_attrs.items():
            setattr(dataknobs_xization, name, value)


def _message(record: pytest.WarningsRecorder, subject: str) -> str:
    texts = [str(w.message) for w in record if issubclass(w.category, DeprecationWarning)]
    matching = [t for t in texts if subject in t]
    assert matching, f"no DeprecationWarning naming {subject!r}; got {texts!r}"
    return matching[0]


@pytest.mark.parametrize("module", DEPRECATED_MODULES)
def test_importing_a_deprecated_module_warns(module: str, fresh_stack: None) -> None:
    with pytest.warns(DeprecationWarning) as record:
        importlib.import_module(f"{PACKAGE}.{module}")

    message = _message(record, f"{PACKAGE}.{module}")
    assert "next major release" in message


@pytest.mark.parametrize("module", DEPRECATED_MODULES)
def test_the_package_door_warns_for_a_deprecated_module(module: str, fresh_stack: None) -> None:
    with pytest.warns(DeprecationWarning) as record:
        reached = getattr(dataknobs_xization, module)

    assert reached is sys.modules[f"{PACKAGE}.{module}"]
    assert "next major release" in _message(record, f"{PACKAGE}.{module}")


def test_a_deprecated_module_the_door_imports_in_turn_still_warns(fresh_stack: None) -> None:
    """Only the reached module's own warning is withheld, not its imports'.

    ``lexicon`` imports ``authorities`` and ``masking_tokenizer``, and both
    are bound as package attributes by that import, so the door is never asked
    for them afterwards. Withholding every warning during the door's import
    left ``masking_tokenizer`` reachable without ever warning.
    """
    with pytest.warns(DeprecationWarning) as record:
        dataknobs_xization.lexicon  # noqa: B018

    for module in DEPRECATED_MODULES:
        _message(record, f"{PACKAGE}.{module} ")
    lexicon_warnings = [w for w in record if f"{PACKAGE}.lexicon " in str(w.message)]
    assert len(lexicon_warnings) == 1


def test_the_package_door_warns_on_every_access_not_only_the_first() -> None:
    """A module warns once, at import; the door warns wherever it is used."""
    for _ in range(2):
        with pytest.warns(DeprecationWarning, match="masking_tokenizer"):
            dataknobs_xization.TextFeatures  # noqa: B018


@pytest.mark.parametrize("name", ["CharacterFeatures", "TextFeatures"])
def test_the_top_level_tokenizer_names_warn_and_name_the_replacement(name: str) -> None:
    with pytest.warns(DeprecationWarning) as record:
        reached = getattr(dataknobs_xization, name)

    from dataknobs_xization import masking_tokenizer

    assert reached is getattr(masking_tokenizer, name)
    message = _message(record, name)
    assert "token_spans" in message


def test_a_from_import_of_a_top_level_tokenizer_name_warns_once() -> None:
    """The import system asks ``hasattr`` before the statement reads the name."""
    with pytest.warns(DeprecationWarning) as record:
        from dataknobs_xization import TextFeatures  # noqa: F401

    assert len([w for w in record if "TextFeatures" in str(w.message)]) == 1


def _attributed_here(record: pytest.WarningsRecorder, subject: str) -> None:
    """The warning naming ``subject`` points at this file, the caller's."""
    files = [w.filename for w in record if subject in str(w.message)]
    assert files == [__file__], files


def test_an_import_statement_attributes_the_warning_to_its_line(fresh_stack: None) -> None:
    with pytest.warns(DeprecationWarning) as record:
        import dataknobs_xization.authorities  # noqa: F401

    _attributed_here(record, f"{PACKAGE}.authorities ")


def test_the_door_attributes_the_warning_to_the_callers_line(fresh_stack: None) -> None:
    with pytest.warns(DeprecationWarning) as record:
        dataknobs_xization.masking_tokenizer  # noqa: B018

    _attributed_here(record, f"{PACKAGE}.masking_tokenizer ")


def test_a_from_import_attributes_the_warning_to_its_line() -> None:
    with pytest.warns(DeprecationWarning) as record:
        from dataknobs_xization import CharacterFeatures  # noqa: F401

    _attributed_here(record, "CharacterFeatures")


def test_an_annotations_name_attributes_the_warning_to_the_callers_line() -> None:
    with pytest.warns(DeprecationWarning) as record:
        annotations.MergeStrategy  # noqa: B018

    _attributed_here(record, "annotations.MergeStrategy")


def test_the_deprecated_names_left_the_package_all() -> None:
    exported = set(dataknobs_xization.__all__)

    assert exported.isdisjoint({*DEPRECATED_MODULES, "CharacterFeatures", "TextFeatures"})
    assert {"annotations", "normalize"} <= exported


def test_an_unknown_package_attribute_is_still_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no_such_name"):
        dataknobs_xization.no_such_name  # noqa: B018


@pytest.mark.parametrize(("name", "says"), DEPRECATED_ANNOTATION_NAMES.items())
def test_a_deprecated_annotations_name_warns_on_access(name: str, says: str) -> None:
    with pytest.warns(DeprecationWarning) as record:
        reached = getattr(annotations, name)

    message = _message(record, f"annotations.{name}")
    assert "next major release" in message
    assert says in message
    assert reached.__module__ == f"{PACKAGE}._annotations_deprecated"


def test_a_from_import_of_a_deprecated_annotations_name_warns() -> None:
    with pytest.warns(DeprecationWarning, match="annotations.HtmlHighlighter"):
        from dataknobs_xization.annotations import HtmlHighlighter  # noqa: F401


def test_the_names_that_stay_in_annotations_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for name in (
            "Annotations",
            "AnnotationsBuilder",
            "AnnotationsMetaData",
            "AnnotationsRowAccessor",
            "AnnotatedText",
            "Annotator",
            "DerivedAnnotationColumns",
            "KEY_START_POS_COL",
        ):
            getattr(annotations, name)


def test_an_unknown_annotations_attribute_is_still_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no_such_name"):
        annotations.no_such_name  # noqa: B018


def test_importing_the_package_emits_no_deprecation_warning() -> None:
    """The control. Run in a fresh interpreter, where nothing is cached.

    Only warnings that name this package are counted: a dependency's own
    deprecations are not this package's to answer for.
    """
    script = (
        "import warnings\n"
        "warnings.simplefilter('always')\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import dataknobs_xization\n"
        "    from dataknobs_xization import *\n"
        "    import dataknobs_xization.annotations\n"
        "    import dataknobs_xization.normalize\n"
        "for w in caught:\n"
        "    if issubclass(w.category, DeprecationWarning) and 'dataknobs_xization' in str(w.message):\n"
        "        print(w.message)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=True
    )

    assert result.stdout == ""


@pytest.mark.parametrize("module", [*DEPRECATED_MODULES, "_annotations_deprecated"])
def test_every_public_name_carries_the_deprecated_directive(module: str) -> None:
    """The docs guard reads the directive, not the warning.

    The root guard that refuses a document teaching a deprecated name in
    silence recognises a deprecation by ``.. deprecated::`` in a class's or
    function's own docstring, and never treats a module as deprecated. A
    module that warns but whose classes carry no directive is invisible to it,
    and it passes over every page that teaches one of them.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        loaded = importlib.import_module(f"{PACKAGE}.{module}")
    defined_here = [
        (name, value)
        for name, value in vars(loaded).items()
        if not name.startswith("_")
        and callable(value)
        and getattr(value, "__module__", None) == loaded.__name__
    ]

    assert defined_here
    missing = [
        name for name, value in defined_here if ".. deprecated::" not in (value.__doc__ or "")
    ]
    assert not missing, f"{module}: no deprecated directive on {missing}"
