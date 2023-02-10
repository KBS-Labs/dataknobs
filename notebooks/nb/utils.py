#!/usr/bin/env python
# coding: utf-8

# # Notebook utilities
# 
# 
# This notebook defines utilities that are commonly useful across a variety of notebooks.
# 
# To "import" this notebook with its utilities, you first need to configure a notebook loader.
# 
# See https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Importing%20Notebooks.html
# 
# This has been implemented in notebooks/util/nbloader.py, whereas to use this notebook, for example:
# 
# ```nb
# import notebooks.util.nbloader
# import notebooks.nb.utils as nb_utils
# 
# nb_utils.fix_display()   # NOTE: This is currently autorun in a cell below,
#                          #       but you may need to re-run it manually
#                          #       if you've reloaded your notebook without
#                          #       restarting the kernel.
# my_ip = nb_utils.get_my_ip()
# ```

# ## Notebook display area utilities
# 
# * use_full_width -- expands the width of the cells to use the full browser window's width
# * prevent_auto_scrolling -- for cases where it is preferable to use the full browser scrolling instead of auto-scrolling within a cell's output

# In[ ]:


from IPython.display import display, HTML


def use_full_width():
    '''
    Running this widens the notebook display area to use the full browser width.
    '''
    display(HTML("<style>.container { width:100% !important; }</style>"))


def prevent_auto_scrolling():
    '''
    When "interactive" widgets display a lot of data and the auto-scrolling
    makes it difficult to see, running this command fixes the display to
    use the page level scrolling to see that output.
    '''
    style = '''
        <style>
           .jupyter-widgets-output-area .output_scroll {
                height: unset !important;
                border-radius: unset !important;
                -webkit-box-shadow: unset !important;
                box-shadow: unset !important;
            }
            .jupyter-widgets-output-area  {
                height: auto !important;
            }
        </style>
    '''
    display(HTML(style))
    
    
def fix_display():
    ''' Convenience function to apply all display fixes '''
    use_full_width()
    prevent_auto_scrolling()


# ## Notebook networking utilities
# 
# * get_my_ip -- Get the IP of the notebook server
# * get_subnet_ip -- Get the subnet IP corresponding to a name (e.g., when the notebook server and another server are both running in the same subnet such as in a docker subnet.)

# In[ ]:


import re


# Cache variables
_MY_IP = None
_SUBNET_IPS = {}


def get_my_ip(verbose=True):
    '''
    Get the IP address of this notebook('s server)
    :param verbose: True to also print the result
    :return: The IP address or None
    '''
    global _MY_IP
    my_ip = _MY_IP
    if not my_ip:
        ip = get_ipython().getoutput("ifconfig eth0 | grep inet | awk '{print $2}'")
        my_ip = ip[0] if len(ip) > 0 else None
        _MY_IP = my_ip
        if verbose:
            print(f'{_MY_IP=}')
    return my_ip


def _find_ip(s, n):
    ip = None
    rv = get_ipython().getoutput('nmap -sn $s | grep $n')
    if len(rv) > 0:
        m = re.search(r'(\d+\.?)+', rv[0])
        if m:
            ip = m.group(0)
    return ip


def get_subnet_ip(name, subnet=None, verbose=True):
    '''
    Get the IP of a named server on a subnet.
    :param name: The (partial) name of the server whose IP to find
    :param subnet: The subnet on which to search (of the form "172.18.0.*")
        If None, then the subnet of the running notebook is used.
    :param verbose: True to also print the result
    :return: The IP address or None
    '''
    the_ip = _SUBNET_IPS.get(name, None)
    if not the_ip:
        if subnet is None:
            my_ip = get_my_ip(verbose=verbose)
            if my_ip:
                c = my_ip.split('.')
                c[-1] = '*'
                subnet = '.'.join(c)
        if subnet:
            the_ip = _find_ip(subnet, name)
            if the_ip:
                _SUBNET_IPS[name] = the_ip
                if verbose:
                    print(f'IP["{name}"]={the_ip}')
    return the_ip


# In[ ]:





# ## Pandas dataframe display utilities

# In[ ]:


import pandas as pd


DEFAULT_MAX_COLWIDTH = pd.get_option('display.max_colwidth')


def display_full_df_text():
    '''
    Configure view of dataframe cells to show all text with wrapping
    instead of expanding the entire column width to the longest string.
    '''
    pd.set_option('display.max_colwidth', 0)
    
    
def truncate_df_text(colwidth=DEFAULT_MAX_COLWIDTH):
    '''
    Truncate text in dataframe cells (back to the default behavior)
    :param colwidth: Optionally specify a column width different from the default
    '''
    pd.set_option('display.max_colwidth', colwidth)
    
    
def display_df(df, max_rows=0, max_cols=0):
    '''
    Display the dataframe with the given limits without affecting the global
    context, where the following special "max" values apply:
    
      * If 0, then keep the current global limits
      * If "None", then show all columns or rows
      * Otherwise, set the value as the maximum
      
    :param max_rows: The maximum number of rows to show
    :param max_cols: The maximum number of columns to show
    '''
    
    context = list()
    
    if max_cols != 0:
        context.extend(
            ['display.max_columns', max_cols]
        )
    if max_rows != 0:
        context.extend([
            'display.max_rows', max_rows,
            'display.min_rows', max_rows,
        ])
        
    if len(context) == 0:
        display(df)
    else:
        with pd.option_context(*context):
            display(df)


# ### Demonstrate and test "display_df"
# 
# * By changing the next cell to "Code"

# # from ipywidgets import interact, interact_manual
# 
# # Concoct a large dataframe
# df = pd.DataFrame(
#     [
#         [f'{chr(c)}-{n}' for n in range(50)]
#         for c in list(
#             range(ord('a'), ord('z')+1)
#         ) + list(
#             range(ord('A'), ord('Z')+1)
#         ) + list(
#             range(ord('0'), ord('9')+1)
#         )
#     ],
#     columns=[f'char-{n}' for n in range(50)]
# )
# 
# limits = [('global limit', 0), ('unlimited', None), ('5', 5)]
# 
# # Interactively display with different parameters
# @interact_manual(
#     max_rows=limits,
#     max_cols=limits,
# )
# def test_display_df(max_rows, max_cols):
#     display_df(df, max_rows, max_cols)

# In[ ]:





# # Apply "default" configurations
# 
# ### Set preferred behavior when using these utilities

# In[ ]:


# Auto-run all display fixes on import
fix_display()


# In[ ]:


# Default to displaying full wrapped text in dataframes
display_full_df_text()

