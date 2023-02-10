#!/usr/bin/env python
# coding: utf-8

# # Elasticsearch Exploration
# 
# Explore elasticsearch by indexing and querying WordNet glosses.
# 
# 
# ## Prerequisite Setup: Start Elasticsearch server
# 
# For convenience, the following will start an elasticsearch server through docker:
# 
# ```sh
# % bin/start_elasticsearch.sh
# ```
# 
# ### Requirements:
# 
# Create directories/files with user:group permissions of 1000:1000
#   * $HOME/data/docker_es
#      * config -- with configuration files
#      * data
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
import dataknobs.utils.elasticsearch_utils as es_utils
import dataknobs.utils.stanza_utils as stanza_utils
import dataknobs.utils.wordnet_utils as wn_utils


# ## Initializations

# In[ ]:


DATADIR = resource_utils.active_datadir()

##
## Derived environmental initializations
##
## NOTE: Double-check this cell's output in case the auto-derivation fails
##       and you will rely on one of these variables.
##

ELASTICSEARCH_IP = nb_utils.get_subnet_ip('elasticsearch', verbose=True)

ES_INDEX = None

if ELASTICSEARCH_IP is not None:
    ES_INDEX = es_utils.ElasticsearchIndex(
        requests_utils.RequestHelper(
            ELASTICSEARCH_IP, 9200,
        ),
        wn_utils.ELASTICSEARCH_TABLE_SETTINGS,
    )
else:
    print(f'***WARNING: ElasticsearchIndex not initialized because elasticsearch server is not up.')

    
WORDNET_FILEPATH = os.path.join(DATADIR, 'dev/wordnet')


##
## Global notebook variables
##

cur_text = ''    # Latest text input
batch_filename = 'es_bulk.nltk.sgloss.jsonl'
ts_builder = None
resp = None
cur_query = None
cur_syntax = True
cur_match_all = False
cur_slop = 0
cur_gloss_field = 'egloss'


# In[ ]:





# ## Make sure elasticsearch is accessible and up

# In[ ]:


if ES_INDEX is not None:
    resp = ES_INDEX.get_cluster_health(verbose=True)
    if not resp.succeeded:
        print(f'FAILED');
    else:
        print('OK');
else:
    print(f'No elasticsearch server');


# In[ ]:





# ## Build a batch file of records for bulk indexing

# In[ ]:


@interact_manual
def build_batch(
    batch_fname=batch_filename, start_id=0, append=False,
    sgloss=True, prev_sgloss_batch='es_bulk.nltk.sgloss.jsonl',
):
    global batch_filename, ts_builder
    batch_filename = batch_fname
    if prev_sgloss_batch:
        prev_fpath = os.path.join(WORDNET_FILEPATH, prev_sgloss_batch)
        ts_builder = wn_utils.build_token_string_builder(prev_fpath)
        print(f'NOTE: Loaded ts_builder with sgloss from "{prev_fpath}"')
    fpath = os.path.join(WORDNET_FILEPATH, batch_filename)
    if os.path.exists(fpath) and not append:
        print(f'Batch file "{fpath}" already exists. Skipping build.')
    else:
        mode = 'a' if append else 'w'
        wn_utils.build_elasticsearch_batch(
            fpath, mode=mode, start_id=start_id,
            include_sgloss=sgloss,
            token_string_builder=ts_builder,
        )
        print(f'Wrote batch data to: "{fpath}"')


# In[ ]:





# ## Purge existing data

# In[ ]:


print(f'Run interact to delete currently indexed data')

@interact_manual
def purge_data():
    global resp
    if ES_INDEX is None:
        print(f'ERROR: Elasticsearch server is not accessible.')
    else:
        resp = ES_INDEX.inspect_indices(verbose=False)
        print(f'({resp.status})\n{resp.result}')
        print(f'\nRun interact to confirm purging indexed data')

        @interact_manual
        def do_delete():
            resp = ES_INDEX.purge(verbose=True)
            if resp.succeeded:
                print(f'Deleted data')
            else:
                print(f'Failed to delete data')


# In[ ]:





# ## Bulk load data to the index

# In[ ]:


print(f'Run interact to upload data to the index')

@interact_manual
def upload_data():
    global resp
    if ES_INDEX is None:
        print(f'ERROR: Elasticsearch server is not accessible.')
    else:
        print(f'Current indices:')
        resp = ES_INDEX.inspect_indices(verbose=False)
        print(f'({resp.status})\n{resp.result}')
        
        print(f'\nRun interact to bulk load "{batch_filename}" to the elasticsearch index')

        @interact_manual
        def do_upload(verbose=True):
            global resp
            fpath = os.path.join(WORDNET_FILEPATH, batch_filename)
            if os.path.exists(fpath):
                # using cell magic:
                fpvar = "@" + fpath
                urlvar = ES_INDEX.request_helper.build_url('_bulk')
                if verbose:
                    print(f'{datetime.now()}: Uploading "{fpath}" to index via "{urlvar}"')
                upload_output = get_ipython().getoutput('curl -s -H "Content-Type: application/x-ndjson" -XPOST $urlvar --data-binary $fpvar')
                if verbose:
                    print(f'{datetime.now()}: Done uploading "{fpath}" ({len(upload_output[0])})')

                # using code (currently BROKEN)
                #resp = ES_INDEX.bulk_load(fpath, verbose=verbose)
                #print(f'Done uploading (success={resp.succeeded})')

                resp = ES_INDEX.inspect_indices(verbose=False)
                if verbose:
                    print(f'({resp.status})\n{resp.result}')
            else:
                print(f'Failed. "{fpath}" does not exist')


# In[ ]:





# ## Re-examine table details

# In[ ]:


print('Click "Run Interact" to re-examine table details:')

@interact_manual
def reexamine_tables():
    resp = ES_INDEX.inspect_indices(verbose=False)
    print(f'({resp.status})\n{resp.result}')


# In[ ]:





# ## Lookup word by text

# In[ ]:


@interact_manual
def lookup_word_by_text(word_text='', verbose=False):
    global resp
    if ES_INDEX is None:
        print(f'ERROR: Elasticsearch server is not accessible.')
    else:
        resp = ES_INDEX.search(
            es_utils.build_field_query_dict('word', word_text),
            verbose=verbose,
        )
        if not resp.succeeded:
            print(f'Search request failed:\n{resp}')
        elif not 'hits_df' in resp.extra:
            print(f'No results')
        else:
            display(resp.extra['hits_df'])


# In[ ]:





# ## Lookup word by ID

# In[ ]:


@interact_manual
def lookup_word_by_id(word_id='', verbose=False):
    global resp
    if ES_INDEX is None:
        print(f'ERROR: Elasticsearch server is not accessible.')
    else:
        resp = ES_INDEX.search(
            es_utils.build_field_query_dict('id', word_id),
            verbose=verbose,
        )
        if not resp.succeeded:
            print(f'Search request failed:\n{resp}')
        elif not 'hits_df' in resp.extra:
            print(f'No results')
        else:
            display(resp.extra['hits_df'])


# In[ ]:





# ## Search index

# In[ ]:


@interact_manual(
    gloss_field=['egloss', 'gloss', 'raw_gloss'],
)
def search_for_phrase(
    phrase=cur_text,
    syntax=cur_syntax,
    gloss_field=cur_gloss_field,
    match_all=cur_match_all,
    slop=widgets.IntText(value=cur_slop),
    verbose=False
):
    global resp, cur_query, cur_text, cur_syntax, cur_match_all, cur_slop, ts_builder
    cur_text = phrase
    cur_syntax = syntax
    cur_match_all = match_all
    cur_slop = slop
    cur_gloss_field = gloss_field
    if ES_INDEX is None:
        print(f'ERROR: Elasticsearch server is not accessible.')
    else:
        field = cur_gloss_field
        if cur_syntax:
            field = 'sgloss'
            if ts_builder is None:
                ts_builder = stanza_utils.TokenStringBuilder()
            phrase = ts_builder.build_string(
                phrase,
                ignore_utags_override=stanza_utils.DEFAULT_IGNORE_QUERY_UTAGS,
            )
        if match_all:
            cur_query = es_utils.build_phrase_query_dict(
                field,
                phrase,
                slop=cur_slop,
            )
        else:
            cur_query = es_utils.build_field_query_dict(
                field, phrase, operator='OR',
            )
        resp = ES_INDEX.search(
            cur_query,
            verbose=verbose,
        )
        if not resp.succeeded:
            print(f'Search request failed:\n{resp}')
        elif not 'hits_df' in resp.extra:
            print(f'No results')
        else:
            print(f'Results for "{phrase}" in "{field}":')
            display(resp.extra['hits_df'])


# ## Analyze text (analyzer exploration)

# In[ ]:


@interact_manual
def analyze_text(analyzer='standard', text=''):
    global resp
    if ES_INDEX is None:
        print(f'No elasticsearch index.')
    else:
        resp = ES_INDEX.analyze(text, analyzer, verbose=True)


# In[ ]:


nb_utils.fix_display()

