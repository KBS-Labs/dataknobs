'''
Widget utilities.
'''
import ipywidgets as widgets
import math


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
