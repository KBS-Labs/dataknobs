#!/usr/bin/env python
# coding: utf-8

# # Stanford NLP Stanza Exploration
# 
# ## Prerequisite setup
# 
# ### Add dependency for "stanza" package:
# 
# ```sh
# % poetry add stanza
# ```
# 
# ### Download stanza resources:
# 
# ```nb
# import stanza
# stanza.download('en', model_dir='/data/opt/stanza_resources')
# ```
# 
# ### (Optional) Start CoreNLP server:
# 
# To take advantage of extra functionality, e.g., coreference resolution, etc,
# start the CoreNLP server (from outside this notebook's environment/container.)
# 
# ```sh
# % bin/start_corenlp.sh
# ```

# In[ ]:


import notebooks.util.nbloader
import notebooks.nb.utils as nb_utils

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

import json
import numpy as np
import os
import pandas as pd
import stanza
from datetime import datetime
from stanza.server import CoreNLPClient

import dataknobs.utils.pandas_utils as pd_utils
import dataknobs.utils.stanza_utils as stanza_utils
import notebooks.util.widget_utils as widget_utils


# ## Initializations

# In[ ]:


##
## Derived environmental initializations
##
## NOTE: Double-check this cell's output in case the auto-derivation fails
##       and you will rely on one of these variables.
##

CORENLP_IP = nb_utils.get_subnet_ip('corenlp', verbose=True)

if CORENLP_IP is None:
    print(f'***WARNING: CoreNLP server is not up.')

##
## Global notebook variables
##

cur_text = ''    # Latest text input
ann = None       # CoreNLP annotations
doc = None       # Stanza document
doc_df = None    # Stanza document DataFrame
en_nlp = None    # Stanza english NLP
coref_data = None
parse_data = None
stanza_doc_path = ''
corenlp_ann_path = ''
corenlp_results = None
stanza_processors = list(stanza_utils.STANZA_PROCESSORS)
corenlp_annotators = list(stanza_utils.CORENLP_ANNOTATORS)
cur_max_rows = 100

proc_checkboxes = widget_utils.MultiCheckbox(
    list(stanza_utils.STANZA_PROCESSORS),
    stanza_processors,
)

annotator_checkboxes = widget_utils.MultiCheckbox(
    list(stanza_utils.CORENLP_ANNOTATORS),
    stanza_processors,
    num_cols=6,
)


# In[ ]:





# ## Stanza Exploration

# In[ ]:


display(proc_checkboxes.ui)

@interact_manual
def stanza_exploration(
    text=cur_text,
):
    global cur_text, stanza_processors, en_nlp, doc, doc_df
    cur_text = text
    changed = (stanza_processors != proc_checkboxes.selected)
    stanza_processors = proc_checkboxes.selected
    if en_nlp is None or changed:
        en_nlp = stanza.Pipeline(
            'en',
            model_dir=stanza_utils.STANZA_RESOURCES_DIR,
            processors=stanza_processors,
            download_method=stanza.DownloadMethod.REUSE_RESOURCES,
        )
    doc = en_nlp(text)
    doc_df = stanza_utils.doc2df(doc)
    print(f'Stanza document for "{text}":')
    
    @interact
    def show_dataframe(max_rows=widgets.IntText(cur_max_rows)):
        global cur_max_rows
        cur_max_rows = max_rows
        nb_utils.display_df(doc_df, max_rows=max_rows, max_cols=None)


# In[ ]:





# ## CoreNLP Exploration
# 
# ### (If the CoreNLP server is up and accessible)

# In[ ]:


display(annotator_checkboxes.ui)

@interact_manual
def corenlp_exploration(
    text=cur_text,
):
    global cur_text, ann, corenlp_annotators, coref_data, parse_data, corenlp_results
    if CORENLP_IP:
        cur_text = text
        ann = None
        corenlp_annotators = annotator_checkboxes.selected
        with CoreNLPClient(
            endpoint=f'http://{CORENLP_IP}:9000',
            annotators=corenlp_annotators,
            start_server=stanza.server.StartServer.DONT_START,
        ) as client:
            ann = client.annotate(
                text,
                properties='english',
                output_format='json'
            )
            corenlp_results = stanza_utils.CoreNLPResults(ann)
        print(f'Annotations of "{text}":')
        @interact(
            annotation_type=list(ann.keys())
        )
        def show_atype(annotation_type):
            global coref_data, parse_data
            annotation_data = ann[annotation_type]
            if isinstance(annotation_data, list):
                # List of sentence data...
                @interact(
                    sentence_num=list(range(len(annotation_data)))
                )
                def show_sdata(sentence_num):
                    global parse_data
                    sent_data = annotation_data[sentence_num]
                    print(f'Sentence #{sentence_num} (index={sent_data.get("index", sentence_num)})')
                    keys = set(sent_data.keys())
                    keys.remove('index')
                    @interact(
                        parse_type=list(keys)
                    )
                    def show_parse(parse_type):
                        global parse_data
                        parse_data = sent_data[parse_type]
                        if isinstance(parse_data, str):
                            print(parse_data)
                        else:
                            display(pd.DataFrame.from_records(parse_data))
            elif isinstance(annotation_data, dict) and len(annotation_data) > 0:
                # Dict of coref data...
                @interact(
                    coref=list(annotation_data.keys())
                )
                def show_cdata(coref):
                    global coref_data
                    coref_data = annotation_data[coref]
                    display(pd_utils.dicts2df(coref_data, rename={'id': 'word_id'}, item_id=None))
    else:
        print(f'Please start the CoreNLP server and set the CORENLP_IP variable to interact with this cell.')


# In[ ]:





# ## Save current stanza doc

# In[ ]:


@interact_manual
def save_stanza_doc(path=stanza_doc_path):
    global stanza_doc_path
    stanza_doc_path = path
    if path and doc is not None:
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                print(json.dumps(doc.to_dict(), indent=2), file=f)
            print(f'Wrote stanza doc to "{path}"')
        else:
            print(f'ERROR: Path "{path}" already exists. Not written.')


# In[ ]:





# ## Restore stanza doc from saved file

# In[ ]:


@interact_manual
def restore_stanza_doc(path=stanza_doc_path):
    global doc, doc_df, stanza_doc_path
    stanza_doc_path = path
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
            doc = stanza.models.common.doc.Document(data)
            doc_df = stanza_utils.doc2df(doc)
            print(f'Stanza document from "{path}":')
            display(doc_df)


# In[ ]:





# ## Save current CoreNLP annotations

# In[ ]:


@interact_manual
def save_corenlp_annotations(path=corenlp_ann_path):
    global corenlp_ann_path
    corenlp_ann_path = path
    if path and ann is not None:
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                print(json.dumps(ann, indent=2), file=f)
            print(f'Wrote corenlp annotations to "{path}"')
        else:
            print(f'ERROR: Path "{path}" already exists. Not written.')


# In[ ]:





# ## Restore CoreNLP annotations from saved file

# In[ ]:


@interact_manual
def restore_corenlp_annotations(path=corenlp_ann_path):
    global ann, corenlp_results, corenlp_ann_path
    corenlp_ann_path = path
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            ann = json.loads(f.read())
            corenlp_results = stanza_utils.CoreNLPResults(ann)
    @interact(
        dtype=['dependencies', 'tokens', 'correfs', 'parse']
    )
    def show_corenlp_results(dtype='dependencies'):
        if dtype == 'dependencies':
            display(corenlp_results.dependencies_df)
        elif dtype == 'tokens':
            display(corenlp_results.tokens_df)
        elif dtype == 'correfs':
            display(corenlp_results.corefs_df)
        elif dtype == 'parse':
            print(corenlp_results.parse_tree)


# In[ ]:




