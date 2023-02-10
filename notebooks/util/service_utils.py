'''
Helper methods for working with the current dev env.
'''
import notebooks.nb.utils as nb_utils

import os
import pandas as pd

import dataknobs.utils.requests_utils as requests_utils
import dataknobs.utils.sql_utils as sql_utils
import dataknobs.utils.elasticsearch_utils as es_utils
import dataknobs.utils.stanza_utils as stanza_utils
import dataknobs.utils.wordnet_utils as wn_utils


VERBOSE = True


VECTORS_DATADIR = '/data/dev/wordnet'
ELASTICSEARCH_BATCHFILE = f'{VECTORS_DATADIR}/es_bulk.nltk.sgloss.jsonl'
VECTORS_FILEPATHS = {
    'egloss': f'{VECTORS_DATADIR}/wn_vs.multi-qa-MiniLM-L6-cos-v1.egloss.npy',
    'word': f'{VECTORS_DATADIR}/wn_vs.multi-qa-MiniLM-L6-cos-v1.word.npy',
}


KNOWN_SERVICES = [
    'stanza',
    'corenlp',
    'pgdb',
    'elasticsearch',
    'defsemb',
    'termemb',
]


# cache -- service_name : handler
_SERVICES = dict()


def get_service_handler(service_name, rebuild=False, verbose=VERBOSE):
    if service_name not in KNOWN_SERVICES:
        print(f'ERROR: Unknown service "{service_name}"')
        return None

    handler = _SERVICES.get(service_name, None)

    if service_name not in _SERVICES or rebuild:
        if service_name == 'stanza':
            handler = build_stanza_processor(verbose=verbose)
        elif service_name == 'corenlp':
            handler = build_corenlp_annotator(verbose=verbose)
        elif service_name == 'pgdb':
            handler = build_postgres_db(verbose=verbose)
        elif service_name == 'elasticsearch':
            handler = build_elasticsearch_index(verbose=verbose)
        elif service_name == 'defsemb':
            handler = build_embedding_store('egloss', verbose=verbose)
        elif service_name == 'termemb':
            handler = build_embedding_store('word', verbose=verbose)

        _SERVICES[service_name] = handler

    return handler


def build_stanza_processor(verbose=False):
    stanza = None
    if os.path.exists(stanza_utils.STANZA_RESOURCES_DIR):
        stanza = stanza_utils.StanzaProcessor(
            stanza_utils.STANZA_RESOURCES_DIR,
            stanza_utils.STANZA_PROCESSORS,
        )
    return stanza


def build_corenlp_annotator(verbose=False):
    corenlp = None
    corenlp_ip = nb_utils.get_subnet_ip('corenlp', verbose=verbose)
    if corenlp_ip is not None:
        corenlp = stanza_utils.CoreNLPAnnotator(
            corenlp_ip,
            stanza_utils.CORENLP_ANNOTATORS,
        )
    return corenlp


def build_postgres_db(verbose=False):
    db = None
    db_ip = nb_utils.get_subnet_ip('postgres', verbose=verbose)
    if db_ip is not None:
        db = sql_utils.PostgresDB(
            host=db_ip
        )
    return db


def build_elasticsearch_index(verbose=False):
    es_index = None
    es_ip = nb_utils.get_subnet_ip('elasticsearch', verbose=verbose)
    if es_ip is not None:
        try:
            es_index = es_utils.ElasticsearchIndex(
                requests_utils.RequestHelper(
                    es_ip, 9200,
                ),
                wn_utils.ELASTICSEARCH_TABLE_SETTINGS,
            )
        except Exception as ex:
            if verbose:
                print(f'WARNING: Failed to connect to elasticsearch index at {es_ip}: {ex}')
            es_index = None
    return es_index


def build_embedding_store(field_name, verbose=False):
    estore = None
    vectors_filepath = VECTORS_FILEPATHS[field_name]
    if os.path.exists(vectors_filepath):
        estore = wn_utils.build_simple_embedding_store(
            vectors_filepath,
            ELASTICSEARCH_BATCHFILE,
            text_field_name=field_name,
        )
    return estore


def load_available_services(rebuild=False, verbose=False):
    return pd.DataFrame([
        {
            'service_name': service_name,
            'available': get_service_handler(
                service_name, rebuild=rebuild, verbose=verbose,
            ) is not None,
        }
        for service_name in KNOWN_SERVICES
    ])
