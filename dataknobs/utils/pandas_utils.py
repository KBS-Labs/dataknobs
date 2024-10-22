import itertools
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Union


def dicts2df(
        dicts: Union[List[Dict], List[List[Dict]]],
        rename: Dict[str, str] = None,
        item_id: str = 'item_id'
):
    '''
    Create a dataframe from a list of or list of lists of dictionaries.
    :param dicts: A list of dictionaries
    :param rename: Dictionary of {from: to} column name mappings
    :param item_id: Name of column to add with list indices
    :return: A dataframe
    '''
    dfs = [
        pd.DataFrame.from_records(rec)
        for rec in dicts
    ]
    for idx, df in enumerate(dfs):
        if rename:
            df.rename(columns=rename, inplace=True)
        if item_id:
            df[item_id] = idx
    df = pd.concat(dfs).reset_index(drop=True) if len(dfs) > 0 else pd.DataFrame()
    return df


def sort_by_strlen(
        df: pd.DataFrame,
        text_col: str,
        ascending: bool = False,
) -> pd.DataFrame:
    '''
    Sort the dataframe according to the length of the strings in a text column
    :param df: The dataframe to sort
    :param text_col: The text column to sort by
    :return: The sorted dataframe
    '''
    return df.loc[
        df[text_col].str.len().sort_values(ascending=ascending).index
    ]


def get_loc_range(bool_ser: pd.Series) -> Tuple[int, int]:
    '''
    Get the first (start index) and last True locations in the boolean series.
    :param bool_ser: A boolean series
    :return: (first_loc, last_loc) for the first and last "True" locations
    '''
    first_loc = bool_ser.index[bool_ser.argmax()]  # first True from beginning
    rev_ser = bool_ser[::-1]  # reverse
    last_loc = rev_ser.index[rev_ser.argmax()]  # first True from end
    return (first_loc, last_loc)


def explode_json_series(json_ser: pd.Series) -> pd.Series:
    '''
    Given a series with each value a json list of items, explode the series
    items.
    :param json_ser: The series with json values
    :return: The exploded series
    '''
    return json_ser[
        np.logical_and(json_ser.notna(), json_ser != '')
    ].apply(json.loads).explode()


class GroupManager:
    '''
    Class to manage groups of rows in a dataframe identified by a json list of
    numbers in a group number column such that all rows sharing the same group
    number constitute a group.
    '''

    def __init__(
            self,
            df: pd.DataFrame,
            group_num_col: str,
    ):
        '''
        Initialize with the dataframe and group number column name.
        :param df: The dataframe with rows in sorted order
        :param group_num_col: The name of the group number column
        '''
        self.idf = df     # input dataframe
        self.gcol = group_num_col
        self._cdf = self._fix_index(df)  # collapsed dataframe
        self._es = None   # expanded series
        self._glocs = None  # Dict[group_num, loc_series]
        self._mdf = None  # mask dataframe

    def _fix_index(self, df: pd.DataFrame) -> pd.DataFrame:
        # Index values must be unique and sorted in the collapsed form
        if (
                df.index.value_counts().max() > 1 or
                not (df.index.sort_values() == df.index).all()
        ):
            # Index isn't unique or in order, so need to reset the index
            df = df.reset_index(
                drop=('__orig_idx__' == None),
                allow_duplicates=False,
                names='__orig_idx__'
            )
        return df

    @property
    def df(self) -> pd.DataFrame:
        '''
        Get the collapsed dataframe such that the group number column holds the
        list of group numbers as a json string in unique rows.
        '''
        return self._cdf

    @property
    def collapsed_df(self) -> pd.DataFrame:
        '''
        Get the collapsed dataframe such that the group number column holds the
        list of group numbers as a json string in unique rows.
        '''
        return self._cdf

    @property
    def expanded_ser(self) -> pd.Series:
        '''
        Get the expanded series such that the group number column holds an
        integer and the index values are repeated for members of multiple
        groups.
        '''
        if self._es is None:
            if self._cdf is not None:
                if self.gcol in self._cdf.columns:
                    gser = self._cdf[self.gcol]
                    gser = explode_json_series(gser)
                    if len(gser) > 0:
                        self._es = gser
                if self._es is None:
                    self._es = pd.Series(
                        np.NaN,
                        index=self._cdf.index,
                        name=self.gcol,
                    )
        return self._es

    @property
    def all_group_locs(self) -> Dict[int, List[int]]:
        '''
        Get all group row locs, indexed by group_num.
        '''
        if self._glocs is None:
            edf = pd.DataFrame(self.expanded_ser)
            edf['__tmp_idx__'] = edf.index
            self._glocs = edf.groupby(
                self.gcol, group_keys=True,
            )['__tmp_idx__'].apply(list).to_dict()
        return self._glocs

    def get_group_locs(self, group_num: int) -> List[int]:
        '''
        Get the row locs for the given group, or None.
        '''
        return self.all_group_locs.get(group_num, None)

    def get_intra_ungrouped_locs(self, group_num: int) -> List[int]:
        '''
        Get the locs for rows within the given group rows, but not in the group
        '''
        result = None
        colname = f'{self.gcol}_{group_num}'
        if colname in self.mask_df:
            mcol = self.mask_df[colname]
            if mcol.any():
                startloc, endloc = get_loc_range(mcol)
                locs = ~mcol[startloc:endloc+1]
                if locs.any():
                    result = locs[locs].index.tolist()
        return result if result is not None else list()

    @property
    def grouped_locs(self) -> List[int]:
        '''
        Get all row locs that are in at least one group
        '''
        return self.mask_df.index[self.mask_df.sum(axis=1) > 0].tolist()

    @property
    def ungrouped_locs(self) -> List[int]:
        '''
        Get all row locs that aren't in any group
        '''
        return self.mask_df.index[self.mask_df.sum(axis=1) == 0].tolist()

    @property
    def all_group_nums(self) -> List[int]:
        '''
        Get the existing group numbers, or an empty list.
        '''
        result = self.expanded_ser.unique()
        result = [x for x in result if not np.isnan(x)]
        return result

    @property
    def max_group_num(self) -> int:
        '''
        Get the maximum group number present, or -1 if there are none.
        '''
        result = self.expanded_ser.max()
        return -1 if np.isnan(result) else result

    @property
    def mask_df(self) -> pd.DataFrame:
        '''
        Get a dataframe of group masks identifying rows in each group.
        '''
        if self._mdf is None:
            cdf = self.collapsed_df
            if self.gcol in cdf:
                def build_mask(gnum):
                    m = pd.Series(False, index=cdf.index)
                    m.loc[self.get_group_locs(gnum)] = True
                    return m

                self._mdf = pd.DataFrame({
                    f'{self.gcol}_{gnum}': build_mask(gnum)
                    for gnum in self.all_group_nums
                })
            else:
                self._mdf = pd.DataFrame({}, index=cdf.index)
        return self._mdf

    def mark_group(self, idx_values: pd.Series, group_num: int = None):
        '''
        Mark the rows identified by their index values as a group, either with
        the next higher available group number or with the given group number.

        If the group_num already exists, then rows without the group_num will
        be added to the group.
        
        :param idx_values: The row locs to include in the group
        :param group_num: The group number to assign, or None to auto-assign.
        '''
        df = self.collapsed_df
        if group_num is None:
            group_num = self.max_group_num + 1
        if self.gcol not in df:
            df[self.gcol] = np.NaN
        cur_values = df.loc[idx_values, self.gcol]

        def add_group(v):
            if pd.notna(v) and v != '':
                groups = set(json.loads(v))
                groups.add(group_num)
                return json.dumps(list(groups))
            else:
                return f'[{group_num}]'

        df[self.gcol] = df[self.gcol].astype(object)
        df.loc[idx_values, self.gcol] = cur_values.apply(add_group)
        self._reset_edf()

    def mark_groups(
            self,
            idx_value_lists: List[pd.Series],
            group_nums: List[int] = None,
    ):
        '''
        Convenience method to mark multiple groups at once.
        :param idx_value_lists: A list of idx_values identifying groups
        :param group_nums: A list of group numbers corresponding to each
            idx_values list. If None, then auto-increment group_nums for
            each list.
        '''
        for pos, idx_values in enumerate(idx_value_lists):
            self.mark_group(
                idx_values,
                group_num=group_nums[pos] if group_nums is not None else None
            )

    def unmark_group(self, group_num: int, idx_values: pd.Series = None):
        '''
        Remove group_num from the specified rows, or entirely.
        :param group_num: The group_num to remove
        :param idx_values: The row locs from which to remove the group_num.
            If None, then group_num will be removed from all rows in which
            it exists.
        '''
        df = self.collapsed_df
        if self.gcol in df:
            if idx_values is None:
                idx_values = df.index
            gser = df[self.gcol]
            mask = gser.mask(gser.index.isin(idx_values), False).astype(bool)
    
            def del_group(v):
                rv = v
                if pd.notna(v) and v != '':
                    groups = set(json.loads(v))
                    groups.discard(group_num)
                    rv = json.dumps(list(groups)) if len(groups) > 0 else np.NaN
                return rv

            gser.where(mask, gser.apply(del_group), inplace=True)
            self._reset_edf()

    def remove_groups(self, group_nums: List[int]):
        '''
        Convenience method to unmark (remove) the listed groups.
        :param group_num: The group numbers to remove
        '''
        for gnum in group_nums:
            self.unmark_group(gnum)

    def find_subsets(self, proper: bool = True) -> Set[int]:
        '''
        Find the groups that are subsets of other groups.
        :param proper: If True, include proper (complete) subsets.
        :return: The set of group numbers that are subsets of other groups
        '''
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
                    gn = int(g2[g2.rindex('_')+1:])
                    rv.add(gn)
                elif combo_sum == s1:  # "earlier" smaller
                    # g1 is a (smaller) subset of g2
                    gn = int(g1[g1.rindex('_')+1:])
                    rv.add(gn)
        return rv

    def remove_subsets(self, proper: bool = True):
        '''
        Convenience method to remove all groups that are subsets of others.
        :param proper: If True, include proper (complete) subsets.
        '''
        self.remove_groups(self.find_subsets(proper=proper))

    def clear_all_groups(self):
        '''
        Remove (unmark) all groups.
        '''
        df = self.collapsed_df
        if self.gcol in df.columns:
            df[self.gcol] = np.NaN
            self._reset_edf()

    def reset_group_numbers(self, start_num: int = 0):
        '''
        Reset group numbers to increase consecutively from a start.
        :param start_num: The group number to start from.
        '''
        glocs = self.all_group_locs
        self.clear_all_groups()
        for idx, (_gnum, locs) in enumerate(glocs.items()):
            self.mark_group(locs, group_num=start_num+idx)

    def _reset_edf(self):
        '''
        Reset the enhanced_df for recomputing.
        '''
        self._es = None
        self._glocs = None
        self._mdf = None

    def get_subgroup_manager(
            self,
            group_num: int,
            subgroup_num_col: str,
    ):
        '''
        Given that this group manager's groups are comprised of groups from
        another group manager, reconstruct that group manager's groups
        identified by the given group_num from this manager.

        Or, get a GroupManager for the subgroup_num_col updated to reflect only
        groups identified by this manager's group_num. Note that this returns
        a copy of this manager's dataframe with the subgroup_num_col updated
        to exclude other subgroup's groupings.

        :param group_num: This manager's group num whose subgroups to get
        :param subgroup_num_col: The name of the collapsed dataframe column
            holding subgroup values.
        :return: A new GroupManager with only the relevant subgroups marked.
        '''
        group_locs = self.get_group_locs(group_num)
        subgroup_ser = self.collapsed_df.loc[group_locs, subgroup_num_col]

        # only subgroup_locs that are not shared should remain
        es = explode_json_series(subgroup_ser)
        vc = es.index.value_counts()
        all_nums = set(es.unique())
        keeper_nums = set(es[vc[vc == 1].index])
        discard_nums = all_nums.difference(keeper_nums)

        # Make (sub) group manager with a *COPY* of this manager's df
        # so as not to destroy the subgroup column's information.
        gm = GroupManager(self.collapsed_df.copy(), subgroup_num_col)
        gm.remove_groups(discard_nums)

        return gm
