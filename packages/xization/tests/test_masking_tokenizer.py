import dataknobs_structures.document as dk_doc
import dataknobs_utils.emoji_utils as dk_emoji
import dataknobs_xization.masking_tokenizer as dk_tok


def test_empty_tokenization():
    tf = dk_tok.TextFeatures("")
    assert [tok.token_text for tok in tf.get_tokens()] == [""]


def test_dual_tokenization():
    # Where dual tokenization is used for camelcase
    text = "MrBill, esq., was born in Springville on July 4th!"
    tf = dk_tok.TextFeatures(
        text,
        split_camelcase=True,
        mark_alpha=True,
        mark_digit=True,
        mark_upper=True,
        mark_lower=True,
        emoji_data=None,
    )
    assert tf.text == text
    assert tf.text_col == dk_doc.TEXT_LABEL
    assert tf.padded_text == " " + text + " "

    tok = tf.build_first_token(normalize_fn=lambda x: x.lower())
    first_tok = tok
    next_tok = tok.next_token
    last_tok = next_tok.last_token

    # Check first token
    assert tf.cdf.shape == (len(text) + tf.roll_padding * 2, 12)
    assert "".join(tf.cdf[tf.text_col]) == tf.padded_text
    assert tok.doctext == tf.doctext
    assert tok.full_text == text
    assert tok.text_id == tf.text_id
    assert tok.token_num == 0
    assert tok.token_text == "Mr"
    assert tok.len == 2
    assert tok.norm_text == "mr"
    assert tok.start_pos == 0
    assert tok.end_pos == 2
    assert tok.token_pos == (0, 2)
    assert tok.pre_delims == ""
    assert tok.post_delims == ""
    assert tok.next_token == next_tok
    assert tok.prev_token == None

    # Check second token
    assert next_tok.token_text == "Bill"
    assert next_tok.token_pos == (2, 6)
    assert next_tok.pre_delims == ""
    assert next_tok.post_delims == ", "

    # Check last token
    assert last_tok.token_text == "4th"
    assert last_tok.token_pos == (46, 49)
    assert last_tok.pre_delims == " "
    assert last_tok.post_delims == "!"

    # Check consistency
    assert first_tok.last_token == last_tok
    assert last_tok.last_token == last_tok
    assert first_tok.first_token == first_tok
    assert last_tok.first_token == first_tok

    # Check repr doesn't choke
    assert len(str(first_tok)) > 0
    assert len(str(next_tok)) > 0
    assert len(str(last_tok)) > 0

    # Check rebuild thru CharacterInputFeatures
    cif = dk_tok.CharacterInputFeatures(
        tf.cdf, first_tok.token_mask, tf.doctext, roll_padding=tf.roll_padding
    )
    assert [tok.norm_text for tok in cif.get_tokens()] == [tok.norm_text for tok in tf.get_tokens()]
    assert (cif.cdf == tf.cdf).all().all()


def test_simple_tokenization():
    # Where simple tokenization is used for non camelcase
    text = "MrBill, esq., was born in Springville on July 4th!"
    tf = dk_tok.TextFeatures(
        text,
        split_camelcase=False,
        mark_alpha=True,
        mark_digit=True,
        mark_upper=True,
        mark_lower=True,
        emoji_data=None,
    )
    assert tf.text == text
    assert tf.text_col == dk_doc.TEXT_LABEL
    assert tf.padded_text == " " + text + " "

    tok = tf.build_first_token(normalize_fn=lambda x: x.lower())
    first_tok = tok
    next_tok = tok.next_token
    last_tok = next_tok.last_token

    # Check first token
    assert tf.cdf.shape == (len(text) + tf.roll_padding * 2, 10)
    assert "".join(tf.cdf[tf.text_col]) == tf.padded_text
    assert tok.doctext == tf.doctext
    assert tok.full_text == text
    assert tok.text_id == tf.text_id
    assert tok.token_num == 0
    assert tok.token_text == "MrBill"
    assert tok.len == 6
    assert tok.norm_text == "mrbill"
    assert tok.start_pos == 0
    assert tok.end_pos == 6
    assert tok.token_pos == (0, 6)
    assert tok.pre_delims == ""
    assert tok.post_delims == ", "
    assert tok.next_token == next_tok
    assert tok.prev_token == None

    # Check second token
    assert next_tok.token_text == "esq"
    assert next_tok.token_pos == (8, 11)
    assert next_tok.pre_delims == ", "
    assert next_tok.post_delims == "., "

    # Check last token
    assert last_tok.token_text == "4th"
    assert last_tok.token_pos == (46, 49)
    assert last_tok.pre_delims == " "
    assert last_tok.post_delims == "!"

    # Check consistency
    assert first_tok.last_token == last_tok
    assert last_tok.last_token == last_tok
    assert first_tok.first_token == first_tok
    assert last_tok.first_token == first_tok

    # Check repr doesn't choke
    assert len(str(first_tok)) > 0
    assert len(str(next_tok)) > 0
    assert len(str(last_tok)) > 0

    # Check rebuild thru CharacterInputFeatures
    cif = dk_tok.CharacterInputFeatures(
        tf.cdf, first_tok.token_mask, tf.doctext, roll_padding=tf.roll_padding
    )
    assert [tok.norm_text for tok in cif.get_tokens()] == [tok.norm_text for tok in tf.get_tokens()]
    assert (cif.cdf == tf.cdf).all().all()


def test_emoji_tokenization_with_camelcase():
    emoji_data = dk_emoji.load_emoji_data()
    if emoji_data is not None:
        emojies = list(emoji_data.emojis.keys())[:9]
        text = (
            "AbcDef"
            + "".join(emojies[:3])
            + "Gh "
            + "".join(emojies[3:5])
            + " Ijkl"
            + "".join(emojies[5:])
            + "!"
        )
        expected_toks = [
            "Abc",
            "Def",
            emojies[0],
            emojies[1],
            emojies[2],
            "Gh",
            emojies[3],
            emojies[4],
            "Ijkl",
            emojies[5],
            emojies[6],
            emojies[7],
            emojies[8],
        ]
        tf = dk_tok.TextFeatures(text, split_camelcase=True, emoji_data=emoji_data)
        tokens = tf.get_tokens()
        assert [token.norm_text for token in tokens] == expected_toks


def test_emoji_tokenization_without_camelcase():
    emoji_data = dk_emoji.load_emoji_data()
    if emoji_data is not None:
        emojies = list(emoji_data.emojis.keys())[:9]
        text = (
            "AbcDef"
            + "".join(emojies[:3])
            + "Gh "
            + "".join(emojies[3:5])
            + " Ijkl"
            + "".join(emojies[5:])
            + "!"
        )
        expected_toks = [
            "AbcDef" + "".join(emojies[0:3]) + "Gh",
            "".join(emojies[3:5]),
            "Ijkl" + "".join(emojies[5:]),
        ]
        tf = dk_tok.TextFeatures(text, split_camelcase=False, emoji_data=emoji_data)
        tokens = tf.get_tokens()
        assert [token.norm_text for token in tokens] == expected_toks
