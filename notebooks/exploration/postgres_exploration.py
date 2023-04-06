#!/usr/bin/env python
# coding: utf-8

# # Postgres Exploration
# 
# Explore a postgres db by indexing and querying WordNet data.
# 
# 
# ## Prerequisite Setup: Start Postgres server
# 
# For convenience, the following will start a postgres server through docker:
# 
# ```sh
# % bin/start_postgres.sh
# ```
# 
# ### Requirements:
# 
# Create directories/files
#   * $HOME/data/docker_pg/data
# 
# ## Data source: Wordnet through NLTK
# 
# References:
# 
# * https://www.nltk.org/howto/wordnet.html
# * https://wordnet.princeton.edu/

# In[ ]:


import notebooks.util.nbloader
import notebooks.nb.utils as nb_utils

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

import json
import numpy as np
import os
import pandas as pd
from datetime import datetime

import dataknobs.utils.requests_utils as requests_utils
import dataknobs.utils.resource_utils as resource_utils
import dataknobs.utils.sql_utils as sql_utils
import dataknobs.utils.elasticsearch_utils as es_utils


# ## Initializations

# In[ ]:


DATADIR = resource_utils.active_datadir()

##
## Derived environmental initializations
##
## NOTE: Double-check this cell's output in case the auto-derivation fails
##       and you will rely on one of these variables.
##

POSTGRES_IP = nb_utils.get_subnet_ip('postgres', verbose=True)

PG_DB = None

if POSTGRES_IP is not None:
    PG_DB = sql_utils.PostgresDB(
        host=POSTGRES_IP
    )
else:
    print(f'***WARNING: Postgres DB not initialized because postgres server is not up.')

    
ELASTICSEARCH_BATCHFILE = f'{DATADIR}/dev/wordnet/es_bulk.nltk.sgloss.jsonl'
WORDNET_DF = None

if 'wordnet' not in PG_DB.table_names and os.path.exists(ELASTICSEARCH_BATCHFILE):
    # Auto upload wordnet data to the DB
    print(f'NOTE: Uploading wordnet data to DB')
    WORDNET_DF = es_utils.collect_batchfile_records(ELASTICSEARCH_BATCHFILE)
    PG_DB.upload('wordnet', WORDNET_DF)
    PG_DB = sql_utils.PostgresDB(host=POSTGRES_IP)
    print(f'...Done')

##
## Global notebook variables
##

cur_table = None
columns_df = None
df = None
query_string = None


# In[ ]:





# ## Browse postgres tables

# In[ ]:


if PG_DB is not None:
    
    print(f'DB tables:')
    display(PG_DB.tables_df)
    
    @interact(
        table_name=[None] + sorted(PG_DB.tables_df['table_name'].tolist())
    )
    def show_columns(table_name):
        global cur_table, columns_df, df
        cur_table = table_name
        if cur_table is None:
            print('No table selected')
            return
        print('Table columns:')
        columns_df = PG_DB.get_columns(table_name)
        display(columns_df)
        
        print(f'Table "{table_name}" preview:')
        @interact_manual
        def get_examples(N=10):
            global df
            df = PG_DB.table_head(table_name, N)
            display(df)
else:
    print(f'No postgres server');


# In[ ]:





# ## Execute a query

# In[ ]:


if PG_DB is not None:
    
    @interact_manual
    def execute_query(query=widgets.Textarea(
        description='Query:',
        value=query_string,
        placeholder=f'SELECT * FROM {cur_table} LIMIT 10',
        layout=widgets.Layout(width='600px', hieght='250px')
    )):
        global query_string, df
        if query:
            df = PG_DB.query(query)
            query_string=query
            display(df)
        else:
            print('Enter the query to execute.')
            
else:
    print(f'No postgres server')


# In[ ]:





# In[ ]:


nb_utils.fix_display()

