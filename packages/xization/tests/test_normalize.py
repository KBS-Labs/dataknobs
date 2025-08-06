import dataknobs_xization.normalize as dk_norm


def test_expand_camelcase_fn():
    assert dk_norm.expand_camelcase_fn("ThisIsATest") == "This Is A Test"
    assert dk_norm.expand_camelcase_fn("TheNLPToolkit") == "The NLP Toolkit"


def test_drop_non_embedded_symbols_fn():
    assert (
        dk_norm.drop_non_embedded_symbols_fn("   .te-st. .this. .   .#$now..   ")
        == "   te-st this    now   "
    )


def test_drop_embedded_symbols_fn():
    assert (
        dk_norm.drop_embedded_symbols_fn("   .te-st. .this. .   .#$now..   ")
        == "   .test. .this. .   .#$now..   "
    )


def test_get_hyphen_slash_expansions_fn():
    f = dk_norm.get_hyphen_slash_expansions_fn
    assert f("abc-def", subs="/: ", add_self=True, do_split=True) == {
        "abc def",
        "abc-def",
        "def",
        "abc",
        "abc/def",
        "abc:def",
    }

    assert f("abc-def", subs="/: ", add_self=True, do_split=False) == {
        "abc def",
        "abc-def",
        "abc/def",
        "abc:def",
    }

    assert f("abc-def", subs=[" ", ""], add_self=False, do_split=True) == {
        "abc def",
        "abc",
        "def",
        "abcdef",
    }

    assert f("this is a test", subs="-", do_split=True, min_split_token_len=2) == {
        "this is a test",
        "this-is-a-test",
    }

    assert f(
        "a/b",
        subs=[" or "],
        add_self=True,
        do_split=True,
        min_split_token_len=1,
        hyphen_slash_re=dk_norm.SLASH_ONLY_RE,
    ) == {
        "a/b",
        "a or b",
        "a",
        "b",
    }


def test_drop_parentheticals_fn():
    f = dk_norm.drop_parentheticals_fn
    assert f("a (bc) d") == "a  d"
    assert f("a (b(cef(gh) d") == "a  d"


def test_expand_ampersand_fn():
    f = dk_norm.expand_ampersand_fn
    assert f("a&b") == "a and b"
    assert f("a & b") == "a and b"


def test_get_all_string_variations():
    assert len(dk_norm.get_lexical_variations("testing")) > 0


def test_int_to_en():
    assert dk_norm.int_to_en(-1) == "negative one"
    assert dk_norm.int_to_en(99) == "ninety nine"
    assert dk_norm.int_to_en(927) == "nine hundred and twenty seven"
    assert dk_norm.int_to_en(500000) == "five hundred thousand"
    assert dk_norm.int_to_en(10000000) == "ten million"
    assert dk_norm.int_to_en(2000000000) == "two billion"
    assert dk_norm.int_to_en(3000000000000) == "three trillion"
    assert dk_norm.int_to_en(3000000000001) == "three trillion one"


def get_year_variations_fn():
    assert dk_norm.year_variations_fn(2003) == {
        "twenty 03",
        "twenty oh three",
        "2003",
        "two thousand, three",
        "twenty and three",
        "two thousand and three",
        "oh three",
        "two thousand oh three",
        "three",
        "03",
        "two thousand 03",
    }

    assert dk_norm.year_variations_fn(2023) == {
        "23",
        "2023",
        "two thousand, twenty-three",
        "twenty-three",
        "two thousand and twenty-three",
        "two thousand twenty-three",
        "two thousand 23",
        "twenty 23",
        "twenty and twenty-three",
        "twenty twenty-three",
    }

    assert dk_norm.year_variations_fn(1973) == {
        "nineteen and seventy-three",
        "seventy-three",
        "nineteen hundred seventy-three",
        "nineteen hundred and seventy-three",
        "1973",
        "nineteen seventy-three",
        "one thousand, nine hundred and seventy-three",
    }

    assert dk_norm.year_variations_fn(-1) == {"-1"}
    assert dk_norm.year_variations_fn(10000) == {"10000"}


def test_replace_smart_quotes_fn():
    f = dk_norm.replace_smart_quotes_fn
    assert f("""a'b"c""") == """a'b"c"""
    assert f(chr(8220) + "test" + chr(8221)) == '"test"'
    assert f(chr(8216) + "test" + chr(8217)) == "'test'"


def test_basic_normalization_fn():
    f = dk_norm.basic_normalization_fn
    text = f"""I{chr(8216)}m  **  OhSOHappy  **  to pick-up my new WhizBang Mercedes for $100K!"""
    assert f(text, do_all=True) == "im oh so happy to pickup my new whiz bang mercedes for 100k"
    assert (
        f(
            text,
            drop_non_embedded_symbols=True,
        )
        == "i'm    oh so happy    to pick-up my new whiz bang mercedes for 100k"
    )
    assert (
        f(
            text,
            drop_embedded_symbols=True,
        )
        == "im  **  oh so happy  **  to pickup my new whiz bang mercedes for $100k!"
    )
    assert (
        f(
            text,
            simplify_quote_chars=True,
            drop_non_embedded_symbols=True,
            drop_embedded_symbols=False,
        )
        == "i'm    oh so happy    to pick-up my new whiz bang mercedes for 100k"
    )
