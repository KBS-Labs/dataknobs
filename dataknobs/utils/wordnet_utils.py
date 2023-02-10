import re
import dataknobs.utils.resource_utils as resource_utils
import dataknobs.utils.sql_utils as sql_utils
import dataknobs.utils.elasticsearch_utils as es_utils
import dataknobs.utils.embedding_utils as embedding_utils
import dataknobs.utils.stanza_utils as stanza_utils


# Synset Pattern: 1:word, 2:POS, 3:sense_num
WN_SYNSET_PATTERN = re.compile(r'^(.*)\.([a-z]+)\.(\d+)$')


# Lemma Pattern: 1:word, 2:POS, 3:sense_num, 4:synset_word
WN_LEMMA_PATTERN = re.compile(r'^(.*)\.([a-z]+)\.(\d+).([^.]+)$')


def split_synset_name(synset_name):
    '''
    Split the components from the synset_name

    :param synset_name: The name of the synset
    :return: [word, POS, sense_num]
    '''
    m = WN_SYNSET_PATTERN.match(synset_name)
    return [m.group(x) for x in range(1,4)]


def split_lemma_name(lemma_name):
    '''
    Split the components from the lemma_name

    :param lemma_name: The name of the synset
    :return: [word, POS, sense_num, synset_word]
    '''
    m = WN_LEMMA_PATTERN.match(lemma_name)
    return [m.group(x) for x in range(1,5)]


wn_data_settings = {
    "settings": {
        "analysis": {
            "analyzer": {
                "std_stop": {
                    "type": "standard",
                    "stopwords": "_english_",
                    "filter": [
                        "lowercase",
                        "english_snow",
                    ]
                }
            },
            "filter": {
                "english_snow": {
                    "type": "snowball",
                    "language": "English",
                }
            },
        }
    }
}

wn_data_mapping = {
    "properties": {
        "id": {
            "type": "long",
        },
        "word": {
            "type": "text",
            "analyzer": "simple",
        },
        "pos": {
            "type": "text",
            "analyzer": "keyword",
        },
        "sense_num": {
            "type": "text",
            "analyzer": "keyword",
        },
        "raw_gloss": {
            "type": "text",
            "analyzer": "std_stop",
        },
        "gloss": {
            "type": "text",
            "analyzer": "std_stop",
        },
        "synset_name": {
            "type": "text",
            "analyzer": "keyword",
        },
        "egloss": {
            "type": "text",
            "analyzer": "std_stop",
        },
        "synsets": {  # space-delimitted synset name's from lemmas
            "type": "text",
            "analyzer": "whitespace",
        },
        "sgloss": {  # space-delimited syntaxnet tokens
            "type": "text",
            "analyzer": "whitespace",
        }
    }
}


wn_data_table_name = 'data'

ELASTICSEARCH_TABLE_SETTINGS = [
    es_utils.TableSettings(
        wn_data_table_name, wn_data_settings, wn_data_mapping
    ),
]


def word_info_generator(wn=None):
    '''
    Generate a record for each WordNet word (lemma) with fields:
      - synset -- the name of the synset (of form: <word>.<pos>.<sense_num>)
      - id -- an ID for the synset (0-based generator order)
      - word -- the "word" portion of the synset name (with underscores
            replaced by spaces for multi-term words.)
      - pos -- the POS (part of speech) of the word
      - sense_num -- the sense number of the word
      - gloss -- the definition of the word
      - lemmas -- a list of lemma names pertaining to this word
      - examples -- a list of example sentences for this word

    :param wn: The wordnet object to use.
    :yield: A word record (dictionary)
    '''
    wn = resource_utils.get_nltk_wordnet() if wn is None else wn
    for idx, synset in enumerate(wn.all_synsets()):
        word, pos, sense_num = split_synset_name(synset.name())
        yield {
            'id': idx,
            'word': word.replace('_', ' '),
            'pos': pos,
            'sense_num': sense_num,
            'gloss': synset.definition(),
            'synset_name': synset.name(),
            'lemmas': synset.lemma_names(),
            'examples': synset.examples(),
        }


def elasticsearch_record_generator(
        include_sgloss=True,
        start_id=0,
        token_string_builder=None,
        wn=None,
):
    '''
    Generate records for elasticsearch indexing.

    :param include_sgloss: True to include the stanza gloss field
        (WARNING: Generating this data may be time consuming.)
    :param start_id: The synset id (0-based) from which to start emitting data.
        (Useful for restarts that preserve prior batch data.)
    :param token_string_builder: The stanza_utils.TokenStringBuilder to use
        for the sgloss (or the default if None).
    :param wn: The wordnet object to use.
    :yield: The next record for indexing
    '''
    builder = (
        token_string_builder
        if token_string_builder is not None
        else stanza_utils.TokenStringBuilder()
    )
    for word_info in word_info_generator(wn=wn):
        word_id = word_info['id']
        if word_id >= start_id:
            word = word_info['word']
            prefix = 'to ' if word_info['pos'] == 'v' else ''
            raw_gloss = word_info['gloss']
            word_info['raw_gloss'] = raw_gloss
            word_and_gloss = f'{prefix}{word}: {raw_gloss}'
            word_info['gloss'] = word_and_gloss
            examples = word_info.pop('examples')

            # egloss: Enhanced gloss with examples, etc.
            word_info['egloss'] = ' '.join([word_and_gloss] + examples)

            # synsets: synset names for this word
            word_info['synsets'] = ' '.join(word_info.pop('lemmas'))
            # sgloss: stanza token string
            if include_sgloss:
                word_info['sgloss'] = builder.build_string(raw_gloss)
            yield word_info


def build_elasticsearch_batch(
        batchfile,
        idx_name=wn_data_table_name,
        mode='a',
        start_id=0,
        include_sgloss=True,
        token_string_builder=None,
        wn=None,
):
    '''
    Build wordnet batch data for elasticsearch bulk indexing.

    :param batchfile: The file to which to write the batch data.
    :param idx_name: The elasticsearch index name
    :param mode: The file mode for opening the batchfile, e.g.,
        'a' for appending to the file,
        'w' for overwriting the file
    :param start_id: The word info record id from which to start emitting
        data. (Useful for restarts that preserve prior batch data.)
    :param include_sgloss: True to include the stanza gloss field
        (WARNING: Generating this data may be time consuming.)
    :param token_string_builder: The stanza_utils.TokenStringBuilder to use
        for the sgloss (or the default if None).
    :param wn: The wordnet object to use.
    '''
    with open (batchfile, mode, encoding='utf-8') as bf:
        es_utils.add_batch_data(
            bf,
            elasticsearch_record_generator(
                include_sgloss=include_sgloss,
                start_id=start_id,
                token_string_builder=token_string_builder,
                wn=wn,
            ),
            idx_name,
            source_id_fieldname="id",
            cur_id=start_id
        )


def build_simple_record_fetcher(es_batchfile, id_field_name='id'):
    '''
    Build a simple record fetcher for wordnet data records based on the
    elasticsearch batch file.

    :param es_batchfile: Path to the elasticsearch batchfile (as generated
        by build_elasticsearch_batch)
    :param id_field_name: The name of the id field in the elasticsearch
        records.
    :return: A RecordFetcher for retrieving elasticsearch records
    '''
    return sql_utils.DataFrameRecordFetcher(
        es_utils.collect_batchfile_records(es_batchfile),
        id_field_name=id_field_name,
        one_based_ids=True,
    )


def build_simple_embedding_store(
        vectors_filepath,
        es_batchfile,
        text_field_name='gloss',
        record_fetcher=None,
):
    '''
    Build a simple embedding store for the wordnet glosses.

    :param vectors_filepath: The path to the embedding vectors file
    :param es_batchfile: An elasticsearch batchfile for retrieving record texts
    :param text_field_name: The elasticsearch record text field for which to
        generate embeddings
    :param record_fetcher: A pre-built record fetcher
    :return: An EmbeddingStore
    '''
    if record_fetcher is None:
        record_fetcher = build_simple_record_fetcher(es_batchfile, id_field_name='id')
    return embedding_utils.HF_Numpy_Embeddings(
        texts=es_utils.collect_batchfile_values(es_batchfile, text_field_name),
        vectors_fpath=vectors_filepath,
        record_fetcher=record_fetcher,
    )


def build_token_string_builder(
        es_batchfile,
        gloss_field_names=('raw_gloss', 'gloss'),
        sgloss_field_name='sgloss',
        stanza_nlp=None,
        ignore_utags=None,
):
    '''
    Build a stanza_utils.TokenStringBuilder that leverages the existing
    sgloss computation in the given batchfile.

    :param es_batchfile: Path to the elasticsearch batchfile
    :param gloss_field_names: The name(s) of the gloss (text) field in the records.
        Note: These are tried in order until one exists.
    :param sgloss_field_name: The name of the sgloss field in the records
    :param stanza_nlp: The stanza_nlp pipeline to use
    :param ignore_utags: The utags to ignore (only for newly built)
    :return: a stanza_utils.TokenStringBuilder
    '''
    cache = dict()
    for record in es_utils.batchfile_record_generator(
            es_batchfile
    ):
        text = None
        for gloss_field_name in gloss_field_names:
            if gloss_field_name in record:
                text = record[gloss_field_name]
                break
        if text is not None:
            sgloss = record.get(sgloss_field_name, None)
            if sgloss is not None:
                cache[text] = sgloss
    if len(cache) == 0:
        cache = None
    return stanza_utils.TokenStringBuilder(
        stanza_nlp=stanza_nlp,
        ignore_utags=ignore_utags,
        text2tokenstring=cache
    )
