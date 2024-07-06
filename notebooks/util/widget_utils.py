'''
Widget utilities.
'''
import ipywidgets as widgets
import functools
import math
import pandas as pd
from IPython.display import display, clear_output
from typing import Callable, Dict, List


class MultiCheckbox:
    '''
    A multi-select widget in the form of multiple checkboxes.
    
    Usage:
    
        import notebooks.util.widget_utils as widget_utils
        my_checkboxes = widget_utils.MultiCheckbox(options_list, selected_list)
        display(my_checkboxes.ui)

    An effective use is to add this (or at least the "display" statement) in a
    cell *before* an interactive function, making it appear as part of the
    interaction.

    To retrieve the list of selected items:

        selected = my_checkboxes.selected
    '''

    def __init__(
            self,
            options,
            selected=None,
            num_cols=4,
            col_width='100px',
    ):
        '''
        :param options: The list of possible options
        :param selected: The options to be initially selected:
            * True -- if all are selected
            * None or False -- if none are selected
            * else -- a sequence of options that are to be selected
        :param num_cols: The number of columns for laying out the widgets
        :param col_width: The column width for the grid cells
        '''
        self.options = options
        self._num_cols = num_cols
        self.col_width = col_width
        self._start_selected = self._init_selected(selected)
        self._checkboxes = self._init_checkboxes(self._start_selected)
        self._shape = None
        self._ui = None

    def _init_selected(self, selected):
        result = []
        if selected is not None:
            if isinstance(selected, bool):
                result = self.options
            else:
                result = list(selected)
        return result

    def _init_checkboxes(self, selected):
        return [
            widgets.Checkbox(
                value=(option in selected),
                description=option
            )
            for option in self.options
        ]

    @property
    def ui(self):
        if self._ui is None:
            self._ui = self._build_ui()
        return self._ui

    @property
    def shape(self):
        if self._shape is None:
            if self._num_cols:
                xdim = min(len(self.options), self._num_cols)
                ydim = math.ceil(len(self.options) / xdim)
                self._shape = (xdim, ydim)
            else:
                self._shape = (len(self.options),)
        return self._shape

    @property
    def num_cols(self):
        return self._num_cols if self._num_cols else self.shape[0]

    @num_cols.setter
    def num_cols(self, new_num_cols):
        self._num_cols = new_num_cols
        self._shape = None
        self._ui = None

    @property
    def selected(self):
        result = [
            cb.description
            for cb in self._checkboxes
            if cb.value
        ]
        return result

    def is_selected(self, option):
        return self._checkboxes[self.options.index(option)].value

    def _build_ui(self):
        return widgets.GridBox(
            self._checkboxes,
            layout=widgets.Layout(
                grid_template_columns=f'repeat({self.num_cols}, {self.col_width})'
            )
        )


ALL = 'ALL'


def unique_sorted_values_plus_ALL(array):
    '''
    Auxiliary to get the sorted values from an array and 'ALL', e.g., for
    a selection widget from the values in a dataframe column.

    :param array: The array whose values to get
    :return: The sorted unique values, including 'ALL' in front
    '''
    unique = array.unique().tolist()
    unique.sort()
    unique.insert(0, ALL)
    return unique


class CascadingDropdowns:
    '''
    Given a dataframe and selection columns, create widgets to show and filter
    the dataframe on those columns.

    Usage:
        > w = CascadingDropdowns(df, [col1, ..., colN])
        > w.display()

    The data for selected columns can be retrieved via, e.g.,:
        > w.fdf
        > w.get_selected_records()
    '''

    def __init__(self, df: pd.DataFrame, cascade_cols: List[str]):
        '''
        :param df: The dataframe to display
        :param cascade_cols: The selection columns
        '''
        self.df = df
        self.cols = cascade_cols
        self.fdf = None  # filter df
        self.filters = dict()
        # build the dropdowns
        self.dropdowns = {
            col: widgets.Dropdown(
                description=col, options=unique_sorted_values_plus_ALL(df[col])
            )
            for col in cascade_cols
        }
        self.output = (
            widgets.Output()
        )  # variable to store the common output for all dropdowns
        self.widget = widgets.VBox(
            [widgets.HBox(list(self.dropdowns.values())), self.output]
        )
        self.apply_filters()
        # bind handlers to the dropdowns
        for c, d in self.dropdowns.items():
            d.observe(self.dropdown_eventhandler, names='value')

    def get_selected_records(self) -> List[Dict]:
        '''
        Get the currently selected records as a dict.
        '''
        return self.fdf[self.cols].to_dict(orient='records')

    def display(self):
        '''
        Display the widget
        '''
        display(self.widget)

    def apply_filters(self, **kwargs) -> pd.DataFrame:
        '''
        Apply the current filters
        '''
        df = self.df
        filters = []
        for col, value in kwargs.items():
            if value is not ALL:
                filters.append(df[col] == value)
        if filters:
            df_filter = functools.reduce(lambda x, y: x & y, filters)
            df = df.loc[df_filter]
        with self.output:
            clear_output()
            display(df)
        self.fdf = df
        for c, d in self.dropdowns.items():
            if c not in self.filters or self.filters[c] == ALL:
                d.options = unique_sorted_values_plus_ALL(df[c])
        return df

    def dropdown_eventhandler(self, change):
        '''
        Configure the column filters per change event
        '''
        col = change['owner'].description
        self.filters[col] = change.new
        for c in self.cols[self.cols.index(col) + 1 :]:
            self.filters[c] = ALL
        self.apply_filters(**self.filters)


class FieldValuesDropdowns:
    '''
    Manage widgets for selecting a field and choosing from associated values

    Usage:
        > w = FieldValuesDropdowns(df=df, fields_label=<fields_colname>, values_label=<values_colname>)
        > w.display()

    The data for selected field and value can be retrieved via, e.g.,:
        > (w.selected_field, w.selected_value)
    '''

    def __init__(
        self,
        fields: List[str] = None,
        values_fn: Callable[[str], List[str]] = None,
        fields_label: str = 'Field',
        values_label: str = 'Value',
        df: pd.DataFrame = None,
        use_columns: bool = True,
        none_field: str = '<Choose>',
        none_value: str = '<Choose>',
    ):
        '''
        :param fields: The fields from which to choose
        :param values_fn: fn(field) that gives the values associated with the field
        :param fields_label: The label for the fields dropdown (or df fields column name)
        :param values_label: The label for the values dropdown (or df values column name)
        :param df: A dataframe holding the fields and/or values to be used instead of
            fields and/or values_fn.
        :param use_columns: True to use the dataframe columns as the fields and
            unique column values as the values
        :param none_field: The (first) selection field indicating no selected field
        :param none_value: The (first) selection value indicating no selected value
        '''
        self._original_fields = fields
        self.fields = fields
        self.values_fn = values_fn
        self.fields_label = fields_label
        self.values_label = values_label
        self.df = df
        self.use_columns = use_columns
        self.none_field = none_field
        self.none_value = none_value
        self._values = None
        self.selected_field = None
        self.selected_value = None
        # set up fields and values using the dataframe
        self._init_fields()
        if self.df is not None:
            if self.fields is None:
                if self.use_columns:
                    self.fields = sorted(self.df.columns)
                elif self.fields_label in self.df.columns:
                    self.fields = sorted(self.df[self.fields_label].unique())
                else:
                    self.fields = list()
            if self.values_fn is None:
                if self.use_columns:
                    self.values_fn = lambda field: sorted(self.df[field].unique())
                elif self.values_label in self.df.columns:
                    self.values_fn = lambda field: sorted(
                        self.df[self.df[self.fields_label] == field][
                            self.values_label
                        ].unique()
                    )
        # build the dropdowns
        self.fields_dropdown = widgets.Dropdown(
            description=self.fields_label,
            options=[self.none_field] + self.fields,
        )
        self.values_dropdown = widgets.Dropdown(
            description=self.values_label,
            options=[self.none_value],
        )
        self.widget = widgets.HBox([self.fields_dropdown, self.values_dropdown])
        # bind handlers to the dropdowns
        self.fields_dropdown.observe(self.fields_dropdown_eventhandler, names='value')
        self.values_dropdown.observe(self.values_dropdown_eventhandler, names='value')

    def display(self):
        '''
        Display the widget
        '''
        display(self.widget)

    def reset(self):
        '''
        Reset to the original state
        '''
        self.fields = self._original_fields
        self._init_fields()
        self.values_dropdown.options = [self.none_value]
        self.values_dropdown.value = self.none_value
        self.fields_dropdown.options = [self.none_field] + self.fields
        self.fields_dropdown.value = self.none_field

    def _init_fields(self):
        if self.df is not None:
            if self.fields is None:
                if self.use_columns:
                    self.fields = sorted(self.df.columns)
                elif self.fields_label in self.df.columns:
                    self.fields = self._get_unique_strs(self.df[self.fields_label])
                else:
                    self.fields = list()
            if self.values_fn is None:
                if self.use_columns:
                    self.values_fn = lambda field: self._get_unique_strs(self.df[field])
                elif self.values_label in self.df.columns:
                    self.values_fn = lambda field: self._get_unique_strs(
                        self.df[self.df[self.fields_label] == field][self.values_label]
                    )

    @staticmethod
    def _get_unique_strs(ser: pd.Series) -> List[str]:
        '''
        Get the unique series values as a sorted list of strings.
        '''
        return sorted(ser.dropna().astype(str).unique())
        # u = None
        # try:
        #    u = ser.dropna().unique()
        # except TypeError:
        #    u = ser.astype(str).unique()
        # return sorted(u)

    def fields_dropdown_eventhandler(self, change):
        '''
        Apply the currently selected field
        '''
        field = change.new
        if self.selected_field != field:
            if field == self.none_field:
                # Empty out the values
                self.values_dropdown.options = [self.none_value]
                self._values = list()
                self.selected_field = None
            else:
                self.selected_field = field
                self._values = self.values_fn(field)
                self.values_dropdown.options = [self.none_value] + self._values
            self.values_dropdown.value = self.none_value

    def values_dropdown_eventhandler(self, change):
        '''
        Apply the currently selected value
        '''
        if self.selected_value != change.new:
            self.selected_value = change.new
            if self.selected_value == self.none_value:
                self.selected_value = None

    def set_fields(self, fields: List[str], df: pd.DataFrame = None):
        self.fields = fields
        if df is not None:
            self.df = df
            self._init_fields()
        self.fields_dropdown.options = [self.none_field] + fields
        self.fields_dropdown.value = self.none_field
