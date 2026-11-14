# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Variants of an authored form, made once where a vocabulary is built.

A fold handles character-wise conventions: case, and whatever else the fold an
index was built with decides two spellings share. A fold may do no more than
that, because it runs on every query too. Two kinds of difference between how
a form is written and how a person types it are outside any fold:

- **structure**: ``&`` against *and*, ``/`` against *or*, and a parenthetical
  carrying a gloss rather than part of the name: *Fruit & Veg (fresh)*;
- **morphology**: *apples* against *Apple*.

So the variants are generated where the vocabulary is built, and each one is
written as a form of its entity like any other. :func:`generate` makes them,
and :func:`admit` decides which are kept.

**A rule** is a function from one form, and a context, to that form's
variants: :data:`Rule`. The context is a mapping the caller fills -- an
entity's metadata, for the ontology's doors -- and most rules ignore it.
:func:`strip_prefix` is the one built in that reads it. Rules compose in the
order given, and **each rule sees what the rules before it made**, so
*Fruit & Veg (fresh)* reaches *Fruit and Veg* through two rules.

**A variant must not make a collision its authors did not.** Two entities
that both declare *Apple* made one form with two readings, and choosing
between them is the resolver's job. Two entities whose *generated* forms meet
made a collision nobody wrote: the plural of *Glass* is *Glasses*, which
another entity may declare. :func:`admit` refuses a variant whose readings
would be more than its source form's readings, and reports it in the
returned :class:`Variants` rather than dropping it.

**Named rules are a registry**, :data:`VARIANT_RULES`. A configuration names a
rule and its parameters (``{rule: split, sep: "/"}``), and
:func:`rules_from_config` builds them; a consumer registers a rule of their
own -- another language's inflection -- under a name of their own, and every
configuration route reaches it.

**English, and replaceable.** :data:`DEFAULT_RULES` --
:func:`drop_parentheticals`, :func:`ampersand`, :func:`slash_or` and
:func:`inflect` -- are English rules. A vocabulary in another language passes
its own.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from dataknobs_common.exceptions import OperationError, ValidationError
from dataknobs_common.registry import PluginRegistry
from dataknobs_common.text.scan import default_normalizer, token_spans

__all__ = [
    "DEFAULT_RULES",
    "VARIANT_RULES",
    "Declared",
    "Refusal",
    "Rule",
    "Variant",
    "Variants",
    "admit",
    "ampersand",
    "camelcase",
    "drop_parentheticals",
    "generate",
    "hyphen",
    "inflect",
    "inflect_keeping",
    "parenthetical",
    "rules_from_config",
    "slash_or",
    "split",
    "strip_prefix",
]

K = TypeVar("K")

#: One generation rule: a form and its context in, the form's variants out.
#: Returning the form itself, or nothing, is allowed; :func:`generate` drops
#: both. A rule that does not read the context ignores the second argument.
Rule = Callable[[str, Mapping[str, Any]], Iterable[str]]

#: The context a rule is handed when the caller has none.
_NO_CONTEXT: Mapping[str, Any] = MappingProxyType({})

_PARENTHETICAL = re.compile(r"\s*\([^()]*\)")
_PARENTHETICAL_INNER = re.compile(r"\(([^()]*)\)")
_AMPERSAND = re.compile(r"\s*&\s*")
_SLASH = re.compile(r"\s*/\s*")
_HYPHEN = re.compile(r"(?<=\w)-(?=\w)")
_CAMELCASE_LU = re.compile(r"([a-z]+)([A-Z])")
_CAMELCASE_UL = re.compile(r"([A-Z]+)([A-Z][a-z])")
_WORD = re.compile(r"[^\W\d_]+")
#: What may stand between a stripped prefix and the rest: whitespace, and the
#: separators a catalogue writes there. Not an apostrophe or a dot, which
#: continue a word -- *Acme's*, *Acme.com* -- rather than end it.
_PREFIX_SEPARATOR = re.compile(r"[\s,:;/|\-\u2013\u2014]+")

#: Word endings that take *-es* in the plural.
_SIBILANT = ("s", "x", "z", "ch", "sh")
#: Endings of a singular that also ends in *s*: *access*, *status*, *analysis*.
_SINGULAR_S = ("ss", "us", "is")
#: The shortest word inflected. *AI* and *us* are not.
_SHORTEST = 3


def _squeeze(text: str) -> str:
    """Runs of whitespace as one space, and none at either end."""
    return " ".join(text.split())


def drop_parentheticals(form: str, context: Mapping[str, Any] = _NO_CONTEXT) -> tuple[str, ...]:
    """*Fruit & Veg (fresh)* gives *Fruit & Veg*. Nothing where nothing is left."""
    dropped = _squeeze(_PARENTHETICAL.sub("", form))
    return (dropped,) if dropped else ()


def parenthetical(form: str, context: Mapping[str, Any] = _NO_CONTEXT) -> tuple[str, ...]:
    """The text inside each parenthetical: *Soda (Pop)* gives *Pop*.

    Not in the default set: it makes a **fragment**, a form naming less than
    its source does, and a scan over text holding the whole form finds the
    fragment inside it too.
    """
    inner = (_squeeze(match.group(1)) for match in _PARENTHETICAL_INNER.finditer(form))
    return tuple(text for text in inner if text)


def ampersand(form: str, context: Mapping[str, Any] = _NO_CONTEXT) -> tuple[str, ...]:
    """*Fruit & Veg* gives *Fruit and Veg*."""
    if "&" not in form:
        return ()
    return (_squeeze(_AMPERSAND.sub(" and ", form)),)


def slash_or(form: str, context: Mapping[str, Any] = _NO_CONTEXT) -> tuple[str, ...]:
    """*Jam / Jelly* gives *Jam or Jelly*.

    Not in a form holding a digit, where a slash is more often part of a code
    or a quantity than a choice between two names: *1/2 gallon* is left alone.
    """
    if "/" not in form or any(char.isdigit() for char in form):
        return ()
    return (_squeeze(_SLASH.sub(" or ", form)),)


def camelcase(form: str, context: Mapping[str, Any] = _NO_CONTEXT) -> tuple[str, ...]:
    """*BlueBerry* gives *Blue Berry*, and *BBQSauce* gives *BBQ Sauce*.

    The form with a space where a lower-case letter meets an upper-case one,
    and before the last capital of a run that a lower-case letter follows.
    Always one result, which is the form itself when it has neither; that is
    what a caller composing this with its own steps can rely on, and
    :func:`generate` drops it.
    """
    spaced = _CAMELCASE_LU.sub(r"\1 \2", form)
    return (_CAMELCASE_UL.sub(r"\1 \2", spaced),)


def hyphen(subs: Sequence[str] = (" ", "")) -> Rule:
    """A rule writing each hyphen between two word characters as each of ``subs``.

    The default gives *Gluten-Free* both *Gluten Free* and *GlutenFree*. Not
    in the default set: whether a hyphen is a variant or a question for the
    fold depends on the fold.

    The pattern is a hyphen only. A space between two words is not a hyphen
    written another way, and a rule that read it as one would give every
    multi-word form a squashed spelling.
    """
    replacements = tuple(subs)

    def rule(form: str, context: Mapping[str, Any] = _NO_CONTEXT) -> tuple[str, ...]:
        if not _HYPHEN.search(form):
            return ()
        return tuple(_HYPHEN.sub(sub, form) for sub in replacements)

    rule.__name__ = "hyphen"
    return rule


def split(sep: str = "/", min_token_len: int = 1) -> Rule:
    """A rule giving each part of a form ``sep`` divides: *Jam / Jelly* gives *Jam*, *Jelly*.

    Nothing unless the form divides into at least two parts, and nothing if
    any part is shorter than ``min_token_len`` characters. Not in the default
    set, because it makes fragments, as :func:`parenthetical` does. Unlike
    :func:`slash_or` it does not skip a form holding a digit: *1/2 Gallon*
    gives *1* and *2 Gallon*, and only the second has a letter to survive.
    """
    if not sep:
        raise ValueError("split needs a non-empty 'sep'")

    def rule(form: str, context: Mapping[str, Any] = _NO_CONTEXT) -> tuple[str, ...]:
        parts = [_squeeze(part) for part in form.split(sep)]
        if len(parts) < 2 or any(len(part) < min_token_len for part in parts):
            return ()
        return tuple(part for part in parts if part)

    rule.__name__ = "split"
    return rule


def strip_prefix(field: str) -> Rule:
    """A rule dropping a leading prefix that the context's ``field`` names.

    *Acme Peanut Butter* gives *Peanut Butter* when ``context[field]`` is
    *Acme*. The field may hold one prefix or a list of them, and each is also
    tried with its parentheticals dropped, so a brand recorded as
    *Acme (Acme Foods Inc)* strips *Acme*. A prefix is stripped only where
    whitespace or a separator -- ``,`` ``:`` ``;`` ``/`` ``|`` or a dash --
    follows it, and the separator goes with it: *Acme, Jam* gives *Jam*, and
    *Acme's Jam* gives nothing. Never the whole form, and the match ignores
    case.

    The one built-in rule that reads its context, and so not in the default
    set: which field holds a prefix is the vocabulary's to say.
    """
    if not field:
        raise ValueError("strip_prefix needs a non-empty 'field'")

    def prefixes(context: Mapping[str, Any]) -> list[str]:
        raw = context.get(field)
        values = raw if isinstance(raw, list | tuple) else [] if raw is None else [raw]
        found: list[str] = []
        for value in values:
            found.append(_squeeze(str(value)))
            found.extend(drop_parentheticals(str(value)))
        return [prefix for prefix in dict.fromkeys(found) if prefix]

    def rule(form: str, context: Mapping[str, Any] = _NO_CONTEXT) -> tuple[str, ...]:
        found: list[str] = []
        for prefix in prefixes(context):
            length = len(prefix)
            if len(form) <= length or form[:length].casefold() != prefix.casefold():
                continue
            separator = _PREFIX_SEPARATOR.match(form, length)
            if separator is not None:
                rest = _squeeze(form[separator.end() :])
                if rest:
                    found.append(rest)
        return tuple(found)

    rule.__name__ = "strip_prefix"
    return rule


def _plural(word: str) -> str | None:
    """The English plural of one word, or ``None`` where it is not inflected.

    Not a word that already reads as plural, not a word shorter than three
    letters, and not anything but letters. A word ending in *-is* takes
    *-es* in its place: *analysis*, *analyses*.
    """
    if len(word) < _SHORTEST or not word.isalpha() or _singular(word) is not None:
        return None
    lower = word.casefold()
    if lower.endswith("y") and lower[-2:-1] not in ("a", "e", "i", "o", "u"):
        return f"{word[:-1]}ies"
    if lower.endswith("is"):
        return f"{word[:-2]}es"
    if lower.endswith(_SIBILANT):
        return f"{word}es"
    return f"{word}s"


def _singular(word: str) -> str | None:
    """The English singular of one word that reads as plural, or ``None``.

    Only the regular endings. A singular that ends in *s* -- *access*,
    *status*, *analysis* -- is left alone.
    """
    if len(word) <= _SHORTEST or not word.isalpha():
        return None
    lower = word.casefold()
    if lower.endswith("ies"):
        return f"{word[:-3]}y"
    if lower.endswith(tuple(f"{ending}es" for ending in ("x", "z", "ch", "sh", "ss"))):
        return word[:-2]
    if lower.endswith(_SINGULAR_S) or not lower.endswith("s"):
        return None
    return word[:-1]


def inflect(form: str, context: Mapping[str, Any] = _NO_CONTEXT) -> tuple[str, ...]:
    """The form with its last word made plural, or made singular.

    *Apple* gives *Apples*; *Cherries* gives *Cherry*; *Peanut Butter Cup*
    gives *Peanut Butter Cups*. The last word only, and only the regular
    English endings. A last word that is not all letters (a code, a closing
    parenthesis), or is shorter than three letters, gives nothing.

    **Regular endings misread some words**, and :func:`admit` refuses only a
    collision between entities, so a misreading that meets no other entity is
    admitted. Most give a spelling nobody types (*Series* gives *Sery*), but a
    few give a real word: *News* gives *New*, and *Jeans* gives *Jean*. Name
    such words in the registered rule's ``keep`` parameter --
    ``{rule: inflect, keep: [news, jeans]}`` -- or with :func:`inflect_keeping`,
    and they are left as written.
    """
    return _inflect(form, frozenset())


def inflect_keeping(keep: Iterable[str]) -> Rule:
    """:func:`inflect`, leaving alone a form whose last word is one of ``keep``, ignoring case."""
    kept = frozenset(word.casefold() for word in keep)

    def rule(form: str, context: Mapping[str, Any] = _NO_CONTEXT) -> tuple[str, ...]:
        return _inflect(form, kept)

    rule.__name__ = "inflect"
    return rule


def _inflect(form: str, kept: frozenset[str]) -> tuple[str, ...]:
    words = form.split()
    if not words or not _WORD.fullmatch(words[-1]) or words[-1].casefold() in kept:
        return ()
    found: list[str] = []
    for changed in (_plural(words[-1]), _singular(words[-1])):
        if changed is not None:
            found.append(" ".join((*words[:-1], changed)))
    return tuple(found)


#: The default rules, in the order they compose: English.
DEFAULT_RULES: tuple[Rule, ...] = (drop_parentheticals, ampersand, slash_or, inflect)


def _rule_name(rule: Rule) -> str:
    return str(getattr(rule, "__name__", None) or type(rule).__name__)


def _generate_marked(
    form: str, rules: Sequence[Rule], context: Mapping[str, Any]
) -> list[tuple[str, str]]:
    """Every variant of ``form`` with the name of the rule that made it, in the order made."""
    shapes: dict[str, str] = {form: ""}
    for rule in rules:
        name = _rule_name(rule)
        made: list[str] = []
        for shape in list(shapes):
            result = rule(shape, context)
            if isinstance(result, str):
                # A str is an Iterable[str]; read as one, each letter would
                # become a variant, and nothing downstream could tell.
                raise TypeError(
                    f"variant rule {name!r} returned the string {result!r}; a rule "
                    f"returns an iterable of forms, such as a tuple"
                )
            made.extend(
                variant for variant in result if variant and any(c.isalpha() for c in variant)
            )
        for variant in made:
            shapes.setdefault(variant, name)
    return [(variant, name) for variant, name in shapes.items() if variant != form]


def generate(
    form: str,
    rules: Sequence[Rule] = DEFAULT_RULES,
    context: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Every variant of one authored form, in the order made, the form itself excluded.

    Each rule runs over the form and over every variant the rules before it
    made. A variant with no letter in it is dropped: a prefix stripped from
    *Acme 2* leaves *2*, which names nothing. The variants are **unfolded**;
    :func:`admit` folds them with the index's own fold.

    Args:
        form: The form as authored.
        rules: The rules to apply, in order. Defaults to :data:`DEFAULT_RULES`.
        context: What a rule reading more than the form is handed. Empty by
            default.

    Returns:
        The variants, without duplicates.
    """
    marked = _generate_marked(form, rules, _NO_CONTEXT if context is None else context)
    return tuple(variant for variant, _ in marked)


@dataclass
class Declared(Generic[K]):
    """One authored form of one entity, keyed by whatever the caller's entities are keyed by.

    A record two of which really can be equal, so it compares field-wise and
    is not frozen: ``context`` is a mapping, and a frozen dataclass holding one
    would claim to be hashable and raise at the call.

    Attributes:
        key: The entity the form belongs to.
        form: The form as authored.
        context: Handed to every rule with this form. An entity's metadata, for
            the ontology's doors.
        generates: Whether variants are made from this form. ``False`` for a
            form that is held -- a collision is counted against it -- but not
            varied, which is what an entity's id is.
    """

    key: K
    form: str
    context: Mapping[str, Any] = field(default_factory=dict)
    generates: bool = True


@dataclass(frozen=True)
class Variant(Generic[K]):
    """One generated form of one entity.

    Attributes:
        key: The entity it is a form of.
        form: The generated form, unfolded, so it can be written as an alias
            with the casing it was authored in.
        source: The authored form it was generated from.
        rule: The name of the rule that made it -- the last, where several
            rules composed. A rule built by :func:`rules_from_config` carries
            the name it is registered by; any other carries its ``__name__``.
    """

    key: K
    form: str
    source: str
    rule: str


@dataclass(frozen=True)
class Refusal(Generic[K]):
    """A variant not admitted, and the entities it would have collided with.

    Attributes:
        variant: The variant refused.
        others: Entities holding the variant's folded form -- by declaring it
            or by generating it -- that its source form does not already name.
    """

    variant: Variant[K]
    others: tuple[K, ...]


@dataclass(frozen=True)
class Variants(Generic[K]):
    """What :func:`admit` kept and what it refused."""

    admitted: tuple[Variant[K], ...] = ()
    refused: tuple[Refusal[K], ...] = ()

    def of(self, key: K) -> tuple[str, ...]:
        """The admitted forms of one entity, unfolded, in the order made."""
        return tuple(variant.form for variant in self.admitted if variant.key == key)

    def longest_form_tokens(self, fold: Callable[[str], str] = default_normalizer) -> int:
        """The most tokens any admitted variant has once folded; ``0`` with none admitted.

        A bound over the variants alone. A store that declares its widest
        form by hand -- a consumer's own side table, filled from
        :attr:`admitted` -- must re-declare it as the larger of this and its
        own, because :func:`ampersand` and :func:`slash_or` each add a token.
        """
        return max((len(token_spans(fold(variant.form))) for variant in self.admitted), default=0)


def admit(
    declared: Iterable[Declared[K]],
    rules: Sequence[Rule] = DEFAULT_RULES,
    *,
    fold: Callable[[str], str] = default_normalizer,
) -> Variants[K]:
    """Generate every variant of every declared form, and keep those that make no new collision.

    **The rule.** A variant is admitted when every entity that would hold its
    folded form -- by declaring it, or by generating it too -- already holds
    the form it was generated from. So a variant can repeat an ambiguity its
    source has, and cannot create one: two entities both declaring *Apple*
    each keep *Apples*, and *Glass*'s *Glasses* is refused where another
    entity declares *Glasses*.

    A variant whose folded form its own entity already declares is dropped
    without record, since it adds nothing.

    **Conservative, and independent of order.** An entity that would generate
    a form counts as holding it whether or not its own claim is admitted, so a
    variant two entities would both generate is refused for each of them even
    where one of them is refused for another reason. That keeps the decision a
    function of the declared forms alone: the same set is admitted in
    whatever order they are given.

    Args:
        declared: Every authored form of every entity sharing the index: the
            whole scope a collision is counted over.
        rules: What makes an authored form's variants, as :func:`generate`
            applies them.
        fold: The index's own fold. Two forms collide when it folds them to
            one key, so this must be the fold the index the variants are
            written into was built with.

    Returns:
        The admitted variants and the refusals, each in the order made.
    """
    held = list(declared)
    authored: dict[str, set[K]] = {}
    for item in held:
        authored.setdefault(fold(item.form), set()).add(item.key)

    candidates: list[tuple[Declared[K], str, str, str]] = []
    claims: dict[str, set[K]] = {}
    for item in held:
        if not item.generates:
            continue
        for form, rule in _generate_marked(item.form, rules, item.context):
            key = fold(form)
            if not key or item.key in authored.get(key, ()):
                continue
            candidates.append((item, form, rule, key))
            claims.setdefault(key, set()).add(item.key)

    admitted: dict[tuple[K, str], Variant[K]] = {}
    refused: dict[tuple[K, str], Refusal[K]] = {}
    for item, form, rule, key in candidates:
        slot = (item.key, key)
        if slot in admitted:
            continue
        made = Variant(key=item.key, form=form, source=item.form, rule=rule)
        holders = authored.get(key, set()) | claims[key]
        others = holders - authored[fold(item.form)]
        if others:
            refused.setdefault(slot, Refusal(variant=made, others=tuple(sorted(others, key=str))))
        else:
            admitted[slot] = made
            refused.pop(slot, None)
    return Variants(admitted=tuple(admitted.values()), refused=tuple(refused.values()))


def _parameterless(rule: Rule) -> Callable[[Mapping[str, Any]], Rule]:
    """A registry factory for a rule that takes no parameters, refusing any it is given."""

    def factory(config: Mapping[str, Any]) -> Rule:
        if config:
            raise ValueError(f"takes no parameters; given {sorted(config)}")
        return rule

    return factory


def _parameters(
    config: Mapping[str, Any], accepted: Mapping[str, type | tuple[type, ...]]
) -> dict[str, Any]:
    """``config`` checked against the parameters a rule accepts, by name and by type."""
    unknown = sorted(set(config) - set(accepted))
    if unknown:
        raise ValueError(f"does not take {unknown}; it takes {sorted(accepted)}")
    for name, value in config.items():
        expected = accepted[name]
        if not isinstance(value, expected) or (isinstance(value, bool) and expected is int):
            raise ValueError(f"'{name}' must be {_describe(expected)}; given {value!r}")
    return dict(config)


def _describe(expected: type | tuple[type, ...]) -> str:
    kinds = expected if isinstance(expected, tuple) else (expected,)
    return " or ".join(kind.__name__ for kind in kinds)


def _hyphen_factory(config: Mapping[str, Any]) -> Rule:
    params = _parameters(config, {"subs": (list, tuple)})
    subs = params.get("subs", (" ", ""))
    if not all(isinstance(sub, str) for sub in subs):
        raise ValueError(f"'subs' must be a list of strings; given {subs!r}")
    return hyphen(tuple(subs))


def _split_factory(config: Mapping[str, Any]) -> Rule:
    params = _parameters(config, {"sep": str, "min_token_len": int})
    return split(params.get("sep", "/"), params.get("min_token_len", 1))


def _inflect_factory(config: Mapping[str, Any]) -> Rule:
    params = _parameters(config, {"keep": (list, tuple)})
    if "keep" not in params:
        return inflect
    keep = params["keep"]
    if not all(isinstance(word, str) for word in keep):
        raise ValueError(f"'keep' must be a list of strings; given {keep!r}")
    return inflect_keeping(keep)


def _strip_prefix_factory(config: Mapping[str, Any]) -> Rule:
    params = _parameters(config, {"field": str})
    if "field" not in params:
        raise ValueError("needs 'field', the context key holding the prefixes")
    return strip_prefix(params["field"])


#: Every rule a configuration can name, by name. Each entry is a factory taking
#: the rule's parameters -- the spec's keys other than ``rule`` -- and returning
#: a :data:`Rule`. Register a rule of your own with
#: ``VARIANT_RULES.register("name", factory)``.
VARIANT_RULES: PluginRegistry[Rule] = PluginRegistry(
    "variant_rules",
    config_key="rule",
    strip_config_key=True,
    not_found_kind="variant rule",
)
for _name, _rule in (
    ("drop_parentheticals", drop_parentheticals),
    ("parenthetical", parenthetical),
    ("ampersand", ampersand),
    ("slash_or", slash_or),
    ("camelcase", camelcase),
):
    VARIANT_RULES.register(_name, _parameterless(_rule))
VARIANT_RULES.register("inflect", _inflect_factory)
VARIANT_RULES.register("hyphen", _hyphen_factory)
VARIANT_RULES.register("split", _split_factory)
VARIANT_RULES.register("strip_prefix", _strip_prefix_factory)
del _name, _rule


def rules_from_config(specs: Sequence[str | Mapping[str, Any]]) -> tuple[Rule, ...]:
    """The rules a configuration names, built, in the order given.

    Each spec is a rule's name, or a mapping naming it under ``rule`` with its
    parameters beside it::

        ["drop_parentheticals", "ampersand", {"rule": "split", "sep": "/"}]

    Args:
        specs: The rule specs, in the order the rules compose.

    Returns:
        The built rules.

    Raises:
        ValidationError: For a spec that is neither a name nor a mapping with a
            ``rule``, a name :data:`VARIANT_RULES` does not hold, or parameters
            the named rule refuses. The message names the rule.
    """
    if isinstance(specs, str | bytes) or not isinstance(specs, Sequence):
        raise ValidationError(
            f"variant rules must be a list of rule specs; given {specs!r}",
            context={"specs": specs},
        )
    built: list[Rule] = []
    for position, spec in enumerate(specs):
        config = {"rule": spec} if isinstance(spec, str) else spec
        name = config.get("rule") if isinstance(config, Mapping) else None
        if not isinstance(name, str) or not name:
            raise ValidationError(
                f"variant rule {position} must be a rule name or a mapping with a "
                f"'rule' naming one; given {spec!r}",
                context={"position": position, "spec": spec},
            )
        if not VARIANT_RULES.is_registered(name):
            raise ValidationError(
                f"unknown variant rule {name!r}. Registered rules: "
                f"{sorted(VARIANT_RULES.list_keys())}",
                context={"rule": name, "position": position},
            )
        try:
            rule = VARIANT_RULES.create(config=dict(config))
        except OperationError as error:
            reason = error.__cause__ if error.__cause__ is not None else error
            raise ValidationError(
                f"variant rule {name!r} {reason}",
                context={"rule": name, "position": position},
            ) from error
        if not callable(rule):
            raise ValidationError(
                f"variant rule {name!r}'s factory returned {rule!r}, which is not "
                f"callable; a factory returns a rule",
                context={"rule": name, "position": position},
            )
        built.append(rule if _rule_name(rule) == name else _Named(rule, name))
    return tuple(built)


class _Named:
    """A rule under the name it was registered by, so a variant records that name."""

    def __init__(self, rule: Rule, name: str) -> None:
        self._rule = rule
        self.__name__ = name

    def __call__(self, form: str, context: Mapping[str, Any]) -> Iterable[str]:
        return self._rule(form, context)
