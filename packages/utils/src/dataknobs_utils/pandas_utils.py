"""Pandas DataFrame utility functions and data transformations.

Provides utilities for creating, transforming, and manipulating Pandas DataFrames,
including conversions between dicts, lists, and DataFrame formats.
"""

import itertools
import json
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd


def dicts2df(
    dicts: Union[List[Dict], List[List[Dict]]],
    rename: Dict[str, str] | None = None,
    item_id: str | None = "item_id",
) -> pd.DataFrame:
    """Create a DataFrame from dictionaries or lists of dictionaries.

    Converts a list of dictionaries or a list of lists of dictionaries into
    a single concatenated DataFrame with optional column renaming and indexing.

    Args:
        dicts: List of dictionaries or list of lists of dictionaries.
        rename: Optional mapping of column names to rename {from: to}.
            Defaults to None.
        item_id: Name of column to add containing the list index. Set to None
            to skip adding this column. Defaults to "item_id".

    Returns:
        pd.DataFrame: Concatenated DataFrame from all dictionaries.
    """
    dfs = [pd.DataFrame.from_records(rec) for rec in dicts]
    for idx, df in enumerate(dfs):
        if rename:
            dfs[idx] = df.rename(columns=rename)
        if item_id:
            dfs[idx][item_id] = idx
    df = pd.concat(dfs).reset_index(drop=True) if len(dfs) > 0 else pd.DataFrame()
    return df


def sort_by_strlen(
    df: pd.DataFrame,
    text_col: str,
    ascending: bool = False,
) -> pd.DataFrame:
    """Sort DataFrame by string length in a text column.

    Args:
        df: DataFrame to sort.
        text_col: Name of the text column to sort by.
        ascending: If True, sort shortest to longest; if False, longest to
            shortest. Defaults to False.

    Returns:
        pd.DataFrame: Sorted DataFrame (original is not modified).
    """
    return df.loc[df[text_col].str.len().sort_values(ascending=ascending).index]


def get_loc_range(bool_ser: pd.Series) -> Tuple[int, int]:
    """Find the range of True values in a boolean Series.

    Args:
        bool_ser: Boolean Series to analyze.

    Returns:
        Tuple[int, int]: Tuple of (first_loc, last_loc) containing indices of
            first and last True values. Returns (0, 0) if no True values exist.
    """
    # Find all True positions
    true_positions = bool_ser[bool_ser].index

    if len(true_positions) == 0:
        # No True values, return (0, 0) or raise an error
        return (0, 0)

    # Convert to int (handling both integer indices and other types)
    # Use tolist() to convert index values to Python types
    first_result = int(true_positions.tolist()[0])
    last_result = int(true_positions.tolist()[-1])

    return (first_result, last_result)


def explode_json_series(json_ser: pd.Series) -> pd.Series:
    """Explode a Series containing JSON-encoded lists into individual items.

    Parses JSON list strings and expands them so each list item becomes a
    separate row with a repeated index.

    Args:
        json_ser: Series with values as JSON-encoded lists (e.g., "[1, 2, 3]").

    Returns:
        pd.Series: Exploded Series with individual list items as values.
    """
    result = json_ser[np.logical_and(json_ser.notna(), json_ser != "")].apply(json.loads).explode()
    return pd.Series(result) if not isinstance(result, pd.Series) else result


class GroupManager:
    """Manage overlapping row groups in a DataFrame using JSON-encoded group lists.

    Handles DataFrames where rows can belong to multiple groups, with group
    membership stored as JSON lists of group numbers (e.g., "[1, 3]" means
    row belongs to groups 1 and 3). Provides utilities for marking, unmarking,
    querying, and analyzing group memberships.

    Attributes:
        idf: Original input DataFrame.
        gcol: Name of the group number column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        group_num_col: str,
    ):
        """Initialize group manager with DataFrame and group column.

        Args:
            df: DataFrame with rows to manage (will be re-indexed if needed).
            group_num_col: Name of the column containing JSON-encoded group
                number lists.
        """
        self.idf = df  # input dataframe
        self.gcol = group_num_col
        self._cdf = self._fix_index(df)  # collapsed dataframe
        self._es: pd.Series | None = None  # expanded series
        self._glocs: Dict[int, List[int]] | None = None  # Dict[group_num, loc_series]
        self._mdf: pd.DataFrame | None = None  # mask dataframe

    def _fix_index(self, df: pd.DataFrame) -> pd.DataFrame:
        # Index values must be unique and sorted in the collapsed form
        if df.index.value_counts().max() > 1 or not (df.index.sort_values() == df.index).all():
            # Index isn't unique or in order, so need to reset the index
            df = df.reset_index(
                drop=False, allow_duplicates=False, names="__orig_idx__"
            )
        return df

    @property
    def df(self) -> pd.DataFrame:
        """Get the DataFrame with JSON-encoded group lists.

        Returns:
            pd.DataFrame: DataFrame with group column containing JSON lists.
        """
        return self._cdf

    @property
    def collapsed_df(self) -> pd.DataFrame:
        """Get the DataFrame with JSON-encoded group lists.

        Alias for the df property.

        Returns:
            pd.DataFrame: DataFrame with group column containing JSON lists.
        """
        return self._cdf

    @property
    def expanded_ser(self) -> pd.Series:
        """Get Series with individual group numbers and repeated indices.

        Expands JSON group lists so each group number becomes a separate
        row with repeated indices for rows belonging to multiple groups.

        Returns:
            pd.Series: Series with individual group numbers, indices repeated
                for multi-group membership.
        """
        if self._es is None:
            if self._cdf is not None:
                if self.gcol in self._cdf.columns:
                    gser = self._cdf[self.gcol]
                    gser = explode_json_series(gser)
                    if len(gser) > 0:
                        self._es = gser
                if self._es is None:
                    self._es = pd.Series(
                        np.nan,
                        index=self._cdf.index,
                        name=self.gcol,
                    )
        return self._es if self._es is not None else pd.Series(dtype=float)

    @property
    def all_group_locs(self) -> Dict[int, List[int]]:
        """Get row indices for all groups.

        Returns:
            Dict[int, List[int]]: Mapping from group number to list of row indices.
        """
        if self._glocs is None:
            edf = pd.DataFrame(self.expanded_ser)
            edf["__tmp_idx__"] = edf.index
            glocs_dict = (
                edf.groupby(
                    self.gcol,
                    group_keys=True,
                )["__tmp_idx__"]
                .apply(list)
                .to_dict()
            )
            self._glocs = glocs_dict if glocs_dict else {}
        return self._glocs if self._glocs is not None else {}

    def get_group_locs(self, group_num: int) -> List[int]:
        """Get row indices for a specific group.

        Args:
            group_num: Group number to query.

        Returns:
            List[int]: List of row indices in the group (empty if group doesn't exist).
        """
        return self.all_group_locs.get(group_num, [])

    def get_intra_ungrouped_locs(self, group_num: int) -> List[int]:
        """Get row indices between group boundaries that aren't in the group.

        Finds rows that fall within the range from the first to last row of
        a group but aren't actually members of the group.

        Args:
            group_num: Group number to analyze.

        Returns:
            List[int]: Row indices within group range but not in the group.
        """
        result = None
        colname = f"{self.gcol}_{group_num}"
        if colname in self.mask_df:
            mcol = self.mask_df[colname]
            if mcol.any():
                startloc, endloc = get_loc_range(mcol)
                locs = ~mcol[startloc : endloc + 1]
                if locs.any():
                    result = locs[locs].index.tolist()
        return result if result is not None else []

    @property
    def grouped_locs(self) -> List[int]:
        """Get indices of all rows belonging to at least one group.

        Returns:
            List[int]: Row indices with group membership.
        """
        return self.mask_df.index[self.mask_df.sum(axis=1) > 0].tolist()

    @property
    def ungrouped_locs(self) -> List[int]:
        """Get indices of all rows not belonging to any group.

        Returns:
            List[int]: Row indices without group membership.
        """
        return self.mask_df.index[self.mask_df.sum(axis=1) == 0].tolist()

    @property
    def all_group_nums(self) -> List[int]:
        """Get all existing group numbers.

        Returns:
            List[int]: List of group numbers currently in use (empty if none).
        """
        result = self.expanded_ser.unique()
        return [int(x) for x in result if not np.isnan(x)]

    @property
    def max_group_num(self) -> int:
        """Get the highest group number in use.

        Returns:
            int: Maximum group number, or -1 if no groups exist.
        """
        result = self.expanded_ser.max()
        return -1 if np.isnan(result) else result

    @property
    def mask_df(self) -> pd.DataFrame:
        """Get DataFrame of boolean masks for each group.

        Returns:
            pd.DataFrame: DataFrame where each column is a boolean mask for a group,
                with column names like "{group_col}_{group_num}".
        """
        if self._mdf is None:
            cdf = self.collapsed_df
            if self.gcol in cdf:

                def build_mask(gnum: int) -> pd.Series:
                    m = pd.Series(False, index=cdf.index)
                    m.loc[self.get_group_locs(gnum)] = True
                    return m

                mdf = pd.DataFrame(
                    {f"{self.gcol}_{gnum}": build_mask(gnum) for gnum in self.all_group_nums}
                )
                self._mdf = mdf
            else:
                self._mdf = pd.DataFrame({}, index=cdf.index)
        return self._mdf if self._mdf is not None else pd.DataFrame()

    def mark_group(
        self, idx_values: Union[pd.Series, List[int]], group_num: int | None = None
    ) -> None:
        """Add rows to a group, creating or updating group membership.

        Assigns rows to a group number, either specified or auto-generated.
        If the group already exists, adds new rows to it without removing
        existing members.

        Args:
            idx_values: Row indices to include in the group.
            group_num: Group number to assign. If None, uses next available
                number (max + 1). Defaults to None.
        """
        df = self.collapsed_df
        if group_num is None:
            group_num = int(self.max_group_num) + 1
        if self.gcol not in df:
            df[self.gcol] = np.nan
        cur_values = df.loc[idx_values, self.gcol]

        def add_group(v: Any) -> str:
            if pd.notna(v) and v != "":
                groups = set(json.loads(v))
                groups.add(group_num)
                return json.dumps(list(groups))
            else:
                return f"[{group_num}]"

        df[self.gcol] = df[self.gcol].astype(object)
        df.loc[idx_values, self.gcol] = cur_values.apply(add_group)
        self._reset_edf()

    def mark_groups(
        self,
        idx_value_lists: List[Union[pd.Series, List[int]]],
        group_nums: List[int] | None = None,
    ) -> None:
        """Mark multiple groups in a single operation.

        Args:
            idx_value_lists: List where each element contains row indices for a group.
            group_nums: Group numbers corresponding to each idx_values list.
                If None, auto-generates consecutive numbers. Defaults to None.
        """
        for pos, idx_values in enumerate(idx_value_lists):
            self.mark_group(
                idx_values, group_num=group_nums[pos] if group_nums is not None else None
            )

    def unmark_group(self, group_num: int, idx_values: pd.Series | None = None) -> None:
        """Remove a group number from specified rows or entirely.

        Args:
            group_num: Group number to remove.
            idx_values: Row indices from which to remove the group. If None,
                removes the group from all rows. Defaults to None.
        """
        df = self.collapsed_df
        if self.gcol in df:
            if idx_values is None:
                idx_values = pd.Series(df.index)
            gser = df[self.gcol]
            mask = gser.mask(gser.index.isin(idx_values), False).astype(bool)

            def del_group(v: Any) -> Any:
                rv = v
                if pd.notna(v) and v != "":
                    groups = set(json.loads(v))
                    groups.discard(group_num)
                    rv = json.dumps(list(groups)) if len(groups) > 0 else np.nan
                return rv

            df[self.gcol] = gser.where(mask, gser.apply(del_group))
            self._reset_edf()

    def remove_groups(self, group_nums: Union[List[int], Set[int]]) -> None:
        """Remove multiple groups entirely.

        Args:
            group_nums: Collection of group numbers to remove completely.
        """
        for gnum in group_nums:
            self.unmark_group(gnum)

    def find_subsets(self, proper: bool = True) -> Set[int]:
        """Find groups that are subsets of other groups.

        Args:
            proper: If True, includes proper subsets (strict subsets) and
                identical groups. If False, only strict subsets. Defaults to True.

        Returns:
            Set[int]: Group numbers that are subsets of other groups.
        """
        rv = set()
        mdf = self.mask_df
        for g1, g2 in itertools.combinations(mdf.columns, 2):
            m1 = mdf[g1]
            m2 = mdf[g2]
            s1 = m1.sum()
            s2 = m2.sum()
            if proper or s1 != s2:
                combo_sum = (m1 & m2).sum()
                if combo_sum == s2:  # "later" smaller or equal
                    # g2 is a (proper or smaller) subset of g1
                    gn = int(g2[g2.rindex("_") + 1 :])
                    rv.add(gn)
                elif combo_sum == s1:  # "earlier" smaller
                    # g1 is a (smaller) subset of g2
                    gn = int(g1[g1.rindex("_") + 1 :])
                    rv.add(gn)
        return rv

    def remove_subsets(self, proper: bool = True) -> None:
        """Remove all groups that are subsets of other groups.

        Args:
            proper: If True, removes proper subsets and identical groups.
                If False, only removes strict subsets. Defaults to True.
        """
        self.remove_groups(list(self.find_subsets(proper=proper)))

    def clear_all_groups(self) -> None:
        """Remove all group assignments from the DataFrame.

        Resets the group column to NaN for all rows.
        """
        df = self.collapsed_df
        if self.gcol in df.columns:
            df[self.gcol] = np.nan
            self._reset_edf()

    def reset_group_numbers(self, start_num: int = 0) -> None:
        """Renumber all groups consecutively starting from a given number.

        Preserves group memberships but assigns new consecutive numbers.

        Args:
            start_num: Starting group number. Defaults to 0.
        """
        glocs = self.all_group_locs
        self.clear_all_groups()
        for idx, (_gnum, locs) in enumerate(glocs.items()):
            self.mark_group(locs, group_num=start_num + idx)

    def _reset_edf(self) -> None:
        """Reset the enhanced_df for recomputing."""
        self._es = None
        self._glocs = None
        self._mdf = None

    def get_subgroup_manager(
        self,
        group_num: int,
        subgroup_num_col: str,
    ) -> "GroupManager":
        """Create a GroupManager for subgroups within a specific group.

        Extracts subgroup information for rows belonging to a specific group,
        filtering out subgroups that are shared with other groups. Returns a
        new GroupManager with a copy of the DataFrame showing only the relevant
        subgroup structure.

        Args:
            group_num: Group number whose subgroups to extract.
            subgroup_num_col: Name of the DataFrame column containing
                subgroup number lists.

        Returns:
            GroupManager: New manager with only subgroups unique to this group.
        """
        group_locs = self.get_group_locs(group_num)
        # Get the subgroup column data as a Series
        subgroup_ser = pd.Series(self.collapsed_df.loc[group_locs, subgroup_num_col])

        # only subgroup_locs that are not shared should remain
        es = explode_json_series(subgroup_ser)
        vc = es.index.value_counts()
        all_nums = set(es.unique())
        keeper_nums = set(es[vc[vc == 1].index])
        discard_nums = all_nums.difference(keeper_nums)

        # Make (sub) group manager with a *COPY* of this manager's df
        # so as not to destroy the subgroup column's information.
        gm = GroupManager(self.collapsed_df.copy(), subgroup_num_col)
        gm.remove_groups(list(discard_nums))

        return gm
