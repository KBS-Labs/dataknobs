import numpy as np
import pandas as pd

import dataknobs_utils.pandas_utils as pd_utils

DEEP_DATA = [
    [
        {
            "id": 80435,
            "word": "xylophonist",
            "pos": "n",
            "sense_num": "01",
            "gloss": "xylophonist: someone who plays a xylophone",
            "synset_name": "xylophonist.n.01",
            "raw_gloss": "someone who plays a xylophone",
            "egloss": "xylophonist: someone who plays a xylophone",
            "synsets": "xylophonist",
            "sgloss": "someone_NN who_WP play_VBZ xylophone_NN",
            "score": 13.805645,
        }
    ],
    [
        {
            "id": 47073,
            "word": "vibraphone",
            "pos": "n",
            "sense_num": "01",
            "gloss": "vibraphone: a percussion instrument similar to a xylophone but having metal bars and rotating disks in the resonators that produce a vibrato sound",
            "synset_name": "vibraphone.n.01",
            "raw_gloss": "a percussion instrument similar to a xylophone but having metal bars and rotating disks in the resonators that produce a vibrato sound",
            "egloss": "vibraphone: a percussion instrument similar to a xylophone but having metal bars and rotating disks in the resonators that produce a vibrato sound",
            "synsets": "vibraphone vibraharp vibes",
            "sgloss": "percussion_NN instrument_NN similar_JJ to_IN xylophone_NN have_VBG metal_NN bar_NNS rotate_VBG disk_NNS in_IN resonator_NNS that_WDT produce_VBP vibrato_NN sound_NN",
            "score": 8.001685,
        }
    ],
]


DF_DATA = {
    "synset_num": {0: 80435, 1: 47073},
    "word": {0: "xylophonist", 1: "vibraphone"},
    "pos": {0: "n", 1: "n"},
    "sense_num": {0: "01", 1: "01"},
    "gloss": {
        0: "xylophonist: someone who plays a xylophone",
        1: "vibraphone: a percussion instrument similar to a xylophone but having metal bars and rotating disks in the resonators that produce a vibrato sound",
    },
    "synset_name": {0: "xylophonist.n.01", 1: "vibraphone.n.01"},
    "raw_gloss": {
        0: "someone who plays a xylophone",
        1: "a percussion instrument similar to a xylophone but having metal bars and rotating disks in the resonators that produce a vibrato sound",
    },
    "egloss": {
        0: "xylophonist: someone who plays a xylophone",
        1: "vibraphone: a percussion instrument similar to a xylophone but having metal bars and rotating disks in the resonators that produce a vibrato sound",
    },
    "synsets": {0: "xylophonist", 1: "vibraphone vibraharp vibes"},
    "sgloss": {
        0: "someone_NN who_WP play_VBZ xylophone_NN",
        1: "percussion_NN instrument_NN similar_JJ to_IN xylophone_NN have_VBG metal_NN bar_NNS rotate_VBG disk_NNS in_IN resonator_NNS that_WDT produce_VBP vibrato_NN sound_NN",
    },
    "score": {0: 13.805645, 1: 8.001685},
    "hit_id": {0: 0, 1: 1},
}


def test_dicts2df():
    df = pd_utils.dicts2df(DEEP_DATA, rename={"id": "synset_num"}, item_id="hit_id")
    assert df.to_dict() == DF_DATA


def test_sort_by_str_length():
    df = pd.DataFrame({"text": ["a", "abcd", "abcdef", "abc"]})
    assert pd_utils.sort_by_strlen(df, "text", ascending=True)["text"].tolist() == [
        "a",
        "abc",
        "abcd",
        "abcdef",
    ]
    assert pd_utils.sort_by_strlen(df, "text", ascending=False)["text"].tolist() == [
        "abcdef",
        "abcd",
        "abc",
        "a",
    ]


def test_group_manager_explode_empties():
    df = pd.DataFrame(["a", "b", "c", "a", "c"], columns=["A"])
    eser = pd.Series(np.nan, index=df.index, name="A_num")

    def check_ser():
        g = pd_utils.GroupManager(df, "A_num")
        # the expanded_df matches the collapsed_df when there are no groups yet
        pd.testing.assert_series_equal(eser, g.expanded_ser)

    check_ser()  # Num col is missing. No groups yet
    df["A_num"] = np.nan
    check_ser()  # Num col is al NaNs. No groups yet
    df["A_num"] = ""
    check_ser()  # Num col is all empty strings. No groups yet


def test_group_manager_explode():
    cdf = pd.DataFrame(
        {
            "A": ["a", "b", "c", "a", "c"],
            "A_num": ["[0, 1]", "[0, 1, 2, 3]", "[0, 2]", "[2, 3]", "[1, 3]"],
        }
    )
    es = pd.Series(
        [0, 1, 0, 1, 2, 3, 0, 2, 2, 3, 1, 3],
        index=[0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4],
        name="A_num",
    )

    g = pd_utils.GroupManager(cdf, "A_num")
    pd.testing.assert_frame_equal(cdf, g.collapsed_df, check_dtype=False)
    pd.testing.assert_series_equal(es, g.expanded_ser, check_dtype=False)


def test_group_manager_mark_and_unmark():
    df = pd.DataFrame({"A": ["a", "b", "c", "a", "c"]})
    g = pd_utils.GroupManager(df, "A_num")
    assert len(g.all_group_nums) == 0
    assert g.all_group_locs == {}
    assert g.grouped_locs == []
    g.mark_groups([[0, 1, 2], [0, 1, 4], [1, 2, 3]])
    assert g.all_group_nums == [0, 1, 2]
    assert g.all_group_locs == {0: [0, 1, 2], 1: [0, 1, 4], 2: [1, 2, 3]}
    assert g.grouped_locs == [0, 1, 2, 3, 4]
    assert g.ungrouped_locs == []
    cdf = pd.DataFrame(
        {"A": ["a", "b", "c", "a", "c"], "A_num": ["[0, 1]", "[0, 1, 2]", "[0, 2]", "[2]", "[1]"]}
    )
    eser = pd.Series(
        [0, 1, 0, 1, 2, 0, 2, 2, 1],
        index=[0, 0, 1, 1, 1, 2, 2, 3, 4],
        name="A_num",
    )
    mdf = pd.DataFrame(
        {
            "A_num_0": [True, True, True, False, False],
            "A_num_1": [True, True, False, False, True],
            "A_num_2": [False, True, True, True, False],
        },
        index=cdf.index,
    )
    pd.testing.assert_frame_equal(cdf, g.collapsed_df, check_dtype=False)
    pd.testing.assert_series_equal(eser, g.expanded_ser, check_dtype=False)
    pd.testing.assert_frame_equal(mdf, g.mask_df, check_dtype=False)

    assert g.get_group_locs(1) == [0, 1, 4]
    assert g.get_intra_ungrouped_locs(1) == [2, 3]
    g.unmark_group(1)
    assert g.get_group_locs(1) == []
    assert g.get_intra_ungrouped_locs(1) == []
    assert g.all_group_nums == [0, 2]
    assert g.all_group_locs == {0: [0, 1, 2], 2: [1, 2, 3]}
    assert g.grouped_locs == [0, 1, 2, 3]
    assert g.ungrouped_locs == [4]
    cdf = pd.DataFrame(
        {"A": ["a", "b", "c", "a", "c"], "A_num": ["[0]", "[0, 2]", "[0, 2]", "[2]", np.nan]}
    )
    eser = pd.Series(
        [0, 0, 2, 0, 2, 2],
        index=[0, 1, 1, 2, 2, 3],
        name="A_num",
    )
    mdf = pd.DataFrame(
        {
            "A_num_0": [True, True, True, False, False],
            "A_num_2": [False, True, True, True, False],
        },
        index=cdf.index,
    )
    pd.testing.assert_frame_equal(cdf, g.collapsed_df, check_dtype=False)
    pd.testing.assert_series_equal(eser, g.expanded_ser, check_dtype=False)
    pd.testing.assert_frame_equal(mdf, g.mask_df, check_dtype=False)

    g.unmark_group(2, [1, 3])
    assert g.all_group_nums == [0, 2]
    assert g.all_group_locs == {0: [0, 1, 2], 2: [2]}
    assert g.grouped_locs == [0, 1, 2]
    assert g.ungrouped_locs == [3, 4]
    cdf = pd.DataFrame(
        {"A": ["a", "b", "c", "a", "c"], "A_num": ["[0]", "[0]", "[0, 2]", np.nan, np.nan]}
    )
    eser = pd.Series(
        [0, 0, 0, 2],
        index=[0, 1, 2, 2],
        name="A_num",
    )
    mdf = pd.DataFrame(
        {
            "A_num_0": [True, True, True, False, False],
            "A_num_2": [False, False, True, False, False],
        },
        index=cdf.index,
    )
    pd.testing.assert_frame_equal(cdf, g.collapsed_df, check_dtype=False)
    pd.testing.assert_series_equal(eser, g.expanded_ser, check_dtype=False)
    pd.testing.assert_frame_equal(mdf, g.mask_df, check_dtype=False)


def test_group_manager_find_subsets():
    df = pd.DataFrame({"A": ["a", "b", "c", "a", "c"]})
    g = pd_utils.GroupManager(df, "A_num")
    g.mark_groups([[0, 1, 2], [0, 1, 4], [1, 2, 3]])
    assert len(g.find_subsets()) == 0
    g.mark_group([0, 1, 2])
    assert len(g.find_subsets(proper=False)) == 0
    assert g.find_subsets(proper=True) == {3}
    g.unmark_group(3)
    assert len(g.find_subsets()) == 0
    g.mark_group([1, 2])
    assert g.find_subsets() == {3}
    g.mark_group([2])
    assert g.find_subsets() == {3, 4}
    g.mark_group([1, 2, 3, 4])
    assert g.find_subsets() == {2, 3, 4}
    g.remove_subsets()
    assert g.all_group_nums == [0, 1, 5]
    g.reset_group_numbers(start_num=12)
    assert g.all_group_nums == [12, 13, 14]


def test_group_manager_subgroups():
    df = pd.DataFrame(
        {
            "entity_text": ["7", "7", "4", "4", "1776", "4", "4", "7", "7", "1776"],
            "date_field": [
                "day",
                "month",
                "day",
                "month",
                "year",
                "day",
                "month",
                "day",
                "month",
                "year",
            ],
        }
    )
    g_num = pd_utils.GroupManager(df, "date_num")
    g_num.mark_groups(
        [
            [0, 3, 4],
            [1, 2, 4],
            [5, 8, 9],
            [6, 7, 9],
        ]
    )
    g_rec = pd_utils.GroupManager(g_num.collapsed_df, "date_recsnum")
    g_rec.mark_groups(
        [
            [0, 3, 4, 5, 8, 9],
            [1, 2, 4, 6, 7, 9],
        ]
    )
    g_num_rec0 = g_rec.get_subgroup_manager(0, "date_num")
    assert g_num_rec0.all_group_locs == {
        0: [0, 3, 4],
        2: [5, 8, 9],
    }
    g_num_rec1 = g_rec.get_subgroup_manager(1, "date_num")
    assert g_num_rec1.all_group_locs == {
        1: [1, 2, 4],
        3: [6, 7, 9],
    }
