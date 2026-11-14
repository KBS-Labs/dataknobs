"""Variants of an authored form: the rules, their composition, and admission.

Every case is on a grocery catalogue, which is neutral ground for the English
default rules: the measurement of those defaults is these tests, and the
module says they are English and replaceable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytest

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.text import (
    DEFAULT_RULES,
    VARIANT_RULES,
    Declared,
    admit,
    ampersand,
    camelcase,
    drop_parentheticals,
    generate,
    hyphen,
    inflect,
    parenthetical,
    rules_from_config,
    slash_or,
    split,
    strip_prefix,
)

# ---- each rule: its documented example, and what it leaves alone ----


def test_drop_parentheticals() -> None:
    assert drop_parentheticals("Fruit & Veg (fresh)") == ("Fruit & Veg",)
    assert drop_parentheticals("Milk (2%) (organic) Gallon") == ("Milk Gallon",)
    assert drop_parentheticals("(fresh)") == ()


def test_parenthetical() -> None:
    assert parenthetical("Soda (Pop)") == ("Pop",)
    assert parenthetical("Soda") == ()


def test_ampersand() -> None:
    assert ampersand("Fruit & Veg") == ("Fruit and Veg",)
    assert ampersand("Fruit&Veg") == ("Fruit and Veg",)
    assert ampersand("Fruit and Veg") == ()


def test_slash_or_leaves_a_form_holding_a_digit_alone() -> None:
    assert slash_or("Jam / Jelly") == ("Jam or Jelly",)
    assert slash_or("Jam/Jelly") == ("Jam or Jelly",)
    assert slash_or("1/2 Gallon Milk") == ()


def test_camelcase() -> None:
    assert camelcase("BlueBerry") == ("Blue Berry",)
    assert camelcase("BBQSauce") == ("BBQ Sauce",)
    assert camelcase("Bread") == ("Bread",)


def test_hyphen_reads_a_hyphen_and_not_a_space() -> None:
    rule = hyphen()
    assert rule("Gluten-Free Bread", {}) == ("Gluten Free Bread", "GlutenFree Bread")
    assert rule("Whole Wheat Bread", {}) == ()


def test_split() -> None:
    assert split()("Jam / Jelly", {}) == ("Jam", "Jelly")
    assert split(sep="-", min_token_len=3)("Ox-Tail", {}) == ()
    assert split()("Bread", {}) == ()
    with pytest.raises(ValueError, match="sep"):
        split(sep="")


@pytest.mark.parametrize(
    ("form", "expected"),
    [
        ("Apple", ("Apples",)),
        ("Cherries", ("Cherry",)),
        ("Peach", ("Peaches",)),
        ("Peanut Butter Cup", ("Peanut Butter Cups",)),
        ("Glasses", ("Glass",)),
        ("Glass", ("Glasses",)),
    ],
)
def test_inflect(form: str, expected: tuple[str, ...]) -> None:
    assert inflect(form) == expected


@pytest.mark.parametrize("form", ["Ox", "Milk 2%", "Cola 12", "Soda (Pop)", ""])
def test_inflect_leaves_alone(form: str) -> None:
    """A short word, and a last word that is not all letters."""
    assert inflect(form) == ()


@pytest.mark.parametrize(
    ("form", "plural"),
    [("Access", "Accesses"), ("Status", "Statuses"), ("Analysis", "Analyses")],
)
def test_a_singular_ending_in_s_is_pluralised_and_not_singularised(form: str, plural: str) -> None:
    assert inflect(form) == (plural,)


def test_strip_prefix_reads_the_context() -> None:
    rule = strip_prefix("brand")
    assert rule("Acme Peanut Butter", {"brand": "Acme"}) == ("Peanut Butter",)
    assert rule("Acme Peanut Butter", {"brand": "Acme (Acme Foods Inc)"}) == ("Peanut Butter",)
    assert rule("Acme Peanut Butter", {"brand": ["Zed", "acme"]}) == ("Peanut Butter",)
    # Not mid-word, not the whole form, and not without the field.
    assert rule("Acmeville Jam", {"brand": "Acme"}) == ()
    assert rule("Acme", {"brand": "Acme"}) == ()
    assert rule("Acme Peanut Butter", {}) == ()


# ---- composition ----


def test_a_rule_sees_what_the_rules_before_it_made() -> None:
    """*Fruit and Veg* needs both the parenthetical dropped and the ampersand read."""
    variants = generate("Fruit & Veg (fresh)", (drop_parentheticals, ampersand))
    assert variants == ("Fruit & Veg", "Fruit and Veg (fresh)", "Fruit and Veg")


def test_the_default_rules_on_one_form() -> None:
    assert generate("Jam / Jelly") == ("Jam or Jelly", "Jam / Jellies", "Jam or Jellies")


def test_generate_excludes_the_form_and_any_variant_with_no_letter() -> None:
    def to_digits(form: str, context: Mapping[str, Any]) -> Iterable[str]:
        return (form, "12", "")

    assert generate("Cola 12", (to_digits,)) == ()


def test_generate_hands_the_context_to_every_rule() -> None:
    assert generate("Acme Jam", (strip_prefix("brand"), inflect), {"brand": "Acme"}) == (
        "Jam",
        "Acme Jams",
        "Jams",
    )


# ---- admission ----


def _catalogue(*rows: tuple[str, str]) -> list[Declared[str]]:
    return [Declared(key, form) for key, form in rows]


def test_glasses_and_glass_refuse_each_other() -> None:
    """Two entities, and the singular of one is the other."""
    report = admit(_catalogue(("glass", "Glass"), ("glasses", "Glasses")))
    assert report.admitted == ()
    refused = {(each.variant.key, each.variant.form): each.others for each in report.refused}
    assert refused == {("glass", "Glasses"): ("glasses",), ("glasses", "Glass"): ("glass",)}


def test_a_variant_may_repeat_its_sources_ambiguity() -> None:
    """Two entities both declaring *Apple* each keep *Apples*."""
    report = admit(_catalogue(("apple-fruit", "Apple"), ("apple-juice", "Apple")))
    assert report.of("apple-fruit") == ("Apples",)
    assert report.of("apple-juice") == ("Apples",)
    assert report.refused == ()


def test_two_generated_forms_meeting_is_a_collision_too() -> None:
    """Nobody declares *Peaches*; two entities would both generate it from different forms."""
    report = admit(_catalogue(("peach", "Peach"), ("peaches-canned", "Peaches (canned)")))
    refused = {(refusal.variant.key, refusal.variant.form) for refusal in report.refused}
    assert ("peach", "Peaches") in refused
    assert ("peaches-canned", "Peaches") in refused


def test_a_variant_its_own_entity_declares_is_dropped_silently() -> None:
    report = admit(_catalogue(("apple", "Apple"), ("apple", "Apples")))
    assert report.admitted == ()
    assert report.refused == ()


def test_a_held_form_is_not_varied_but_counts() -> None:
    """An id is held: a variant meeting it is refused, and it makes no variant of its own."""
    report = admit([Declared("cup", "Cup"), Declared("cups", "cups", generates=False)])
    assert [refusal.variant.form for refusal in report.refused] == ["Cups"]
    assert report.of("cups") == ()


def test_variants_are_emitted_unfolded_and_compared_folded() -> None:
    report = admit(_catalogue(("bread", "Rye Bread")), fold=str.casefold)
    assert report.of("bread") == ("Rye Breads",)
    # Each entity's inflection meets the other's form once case is folded away.
    folded = admit(_catalogue(("bread", "Rye Bread"), ("other", "rye breads")), fold=str.casefold)
    assert folded.admitted == ()
    assert {(r.variant.key, r.variant.form) for r in folded.refused} == {
        ("bread", "Rye Breads"),
        ("other", "rye bread"),
    }


def test_readmitting_under_the_default_rules_changes_nothing() -> None:
    """Over this catalogue, admitting the admitted set again, as declared forms, adds nothing."""
    catalogue = _catalogue(
        ("glass", "Glass"),
        ("glasses", "Glasses"),
        ("apple-fruit", "Apple"),
        ("apple-juice", "Apple"),
        ("fv", "Fruit & Veg (fresh)"),
        ("jam", "Jam / Jelly"),
        ("peach", "Peach"),
        ("peaches-canned", "Peaches (canned)"),
    )
    first = admit(catalogue)
    again = admit([*catalogue, *(Declared(v.key, v.form) for v in first.admitted)])
    assert again.admitted == ()
    assert again.refused == first.refused


def test_variant_records_its_source_and_rule() -> None:
    (variant,) = admit(_catalogue(("apple", "Apple"))).admitted
    assert (variant.form, variant.source, variant.rule) == ("Apples", "Apple", "inflect")


def test_longest_form_tokens_counts_the_admitted_variants() -> None:
    report = admit(_catalogue(("fv", "Fruit & Veg")))
    # "Fruit and Veg": `and` is a token where `&` was not.
    assert report.longest_form_tokens() == 3
    assert admit(_catalogue(("ox", "Ox"))).longest_form_tokens() == 0


# ---- the registry ----


def test_every_built_in_rule_is_registered() -> None:
    assert {
        "drop_parentheticals",
        "parenthetical",
        "ampersand",
        "slash_or",
        "camelcase",
        "hyphen",
        "split",
        "strip_prefix",
        "inflect",
    } <= set(VARIANT_RULES.list_keys())


def test_the_default_rules_by_name_are_the_default_rules() -> None:
    assert (
        rules_from_config(["drop_parentheticals", "ampersand", "slash_or", {"rule": "inflect"}])
        == DEFAULT_RULES
    )


def test_a_registered_rule_is_reachable_by_name() -> None:
    def suffix_rule(config: Mapping[str, Any]) -> Any:
        suffix = config["suffix"]
        return lambda form, context: (f"{form} {suffix}",)

    VARIANT_RULES.register("test_suffix", suffix_rule)
    try:
        (rule,) = rules_from_config([{"rule": "test_suffix", "suffix": "Pack"}])
        assert generate("Cola", (rule,)) == ("Cola Pack",)
    finally:
        VARIANT_RULES.unregister("test_suffix")


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        (["no_such_rule"], "unknown variant rule 'no_such_rule'"),
        ([{"rule": "inflect", "language": "fr"}], "'inflect' does not take \\['language'\\]"),
        ([{"rule": "drop_parentheticals", "x": 1}], "'drop_parentheticals' takes no parameters"),
        ([{"rule": "strip_prefix"}], "'strip_prefix' needs 'field'"),
        ([{"rule": "split", "min_token_len": "2"}], "'split' 'min_token_len' must be int"),
        ([{"rule": "hyphen", "subs": [1]}], "'hyphen' 'subs' must be a list of strings"),
        ([{"rule": "inflect", "keep": "news"}], "'inflect' 'keep' must be list or tuple"),
    ],
)
def test_a_bad_spec_is_refused_naming_the_rule_and_the_reason(
    specs: list[Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        rules_from_config(specs)


@pytest.mark.parametrize("specs", ["inflect", [{"sep": "/"}], [3]])
def test_a_malformed_list_is_refused(specs: Any) -> None:
    with pytest.raises(ValidationError, match="variant rule"):
        rules_from_config(specs)


# ---- found in review ----


def test_a_rule_returning_one_string_is_refused_rather_than_read_as_letters() -> None:
    """A ``str`` is an ``Iterable[str]``, so iterating it would admit every letter as a form."""

    def suffixed(form: str, context: Mapping[str, Any]) -> Any:
        return f"{form} Pack"

    with pytest.raises(TypeError, match="suffixed"):
        generate("Cola", (suffixed,))


def test_a_factory_returning_no_rule_is_refused_by_name() -> None:
    VARIANT_RULES.register("test_not_a_rule", lambda config: "not callable")
    try:
        with pytest.raises(ValidationError, match=r"test_not_a_rule.*callable"):
            rules_from_config(["test_not_a_rule"])
    finally:
        VARIANT_RULES.unregister("test_not_a_rule")


def test_a_registered_rule_reports_its_registered_name() -> None:
    VARIANT_RULES.register("test_pack", lambda config: lambda form, context: (f"{form} Pack",))
    try:
        rules = rules_from_config(["test_pack"])
        (variant,) = admit([Declared("cola", "Cola")], rules).admitted
        assert variant.rule == "test_pack"
    finally:
        VARIANT_RULES.unregister("test_pack")


@pytest.mark.parametrize("form", ["Acme's Peanut Butter", "Acme.com Jam"])
def test_strip_prefix_stops_at_a_word_and_not_at_any_punctuation(form: str) -> None:
    assert strip_prefix("brand")(form, {"brand": "Acme"}) == ()


@pytest.mark.parametrize("form", ["Acme, Jam", "Acme - Jam", "Acme: Jam", "Acme / Jam"])
def test_strip_prefix_drops_a_separator_after_the_prefix(form: str) -> None:
    assert strip_prefix("brand")(form, {"brand": "Acme"}) == ("Jam",)


def test_inflect_leaves_the_words_it_is_told_to_keep() -> None:
    """*News* is not the plural of *New*; a vocabulary says which words inflection must leave."""
    (rule,) = rules_from_config([{"rule": "inflect", "keep": ["news", "Jeans"]}])
    assert rule("Evening News", {}) == ()
    assert rule("Blue Jeans", {}) == ()
    assert rule("Apple", {}) == ("Apples",)
    assert inflect("Evening News") == ("Evening New",)  # the default keeps nothing
