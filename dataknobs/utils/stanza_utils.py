import os
import pandas as pd
import stanza
import dataknobs.structures.tree as dk_tree
import dataknobs.utils.pandas_utils as pd_utils
from stanza.server import CoreNLPClient


DATADIR =  os.environ.get(
    'DATADIR',
    os.path.join(os.environ['HOME'], 'data')
)
STANZA_RESOURCES_DIR = os.path.join(DATADIR, 'opt/stanza_resources')
STANZA_PROCESSORS = (
    'tokenize', 'pos', 'lemma', 'depparse', 'sentiment', 'constituency', 'ner',
)
CORENLP_ANNOTATORS = (
    'ssplit', 'tokenize', 'lemma', 'pos', 'ner', 'parse', 'natlog', 'openie', 'coref',
)


DEFAULT_IGNORE_STORE_UTAGS = None
DEFAULT_IGNORE_QUERY_UTAGS = (
    'PUNCT', 'DET', 'CCONJ', 'SCONJ', 'CD',
)


_DEFAULT_STANZA_PIPELINE = None


def build_stanza_pipeline(
        language='en',
        processors=STANZA_PROCESSORS,
        model_dir=STANZA_RESOURCES_DIR,
        reuse_resources=True,
        stanza_pipeline_builder=stanza.Pipeline,
):
    '''
    Build a stanza pipeline with the given params.
    '''
    download_method = (
        stanza.DownloadMethod.REUSE_RESOURCES
        if reuse_resources else None
    )
    stanza_nlp = stanza_pipeline_builder(
        language,
        model_dir=model_dir,
        processors=processors,
        download_method=download_method,
        #use_gpu=True,  #NOTE: This isn't necessary. GPU is used when detected.
    )
    return stanza_nlp


def default_stanza_pipeline(stanza_pipeline_builder=stanza.Pipeline):
    '''
    Get the default stanza pipeline.
    '''
    global _DEFAULT_STANZA_PIPELINE  # pylint: disable-msg=W0603
    if _DEFAULT_STANZA_PIPELINE is None:
        _DEFAULT_STANZA_PIPELINE = build_stanza_pipeline(
            stanza_pipeline_builder=stanza_pipeline_builder
        )
        #_DEFAULT_STANZA_PIPELINE.use_gpu = True
    return _DEFAULT_STANZA_PIPELINE


def doc2df(doc):
    '''
    Create a dataframe from a stanza doc
    :param doc: A stanza doc
    :return: A dataframe
    '''
    return pd_utils.dicts2df([
        s.to_dict()
        for s in doc.sentences
    ], rename={'id': 'word_id'}, item_id='sent_id')


class TokenStringBuilder:
    '''
    Build a space-delimitted string from the given text with each token of
    the form:

        <lemma.lower()>_<xpos>

    For tokens without a <upos> value in the given ignore_utags.
    '''

    def __init__(
            self,
            stanza_nlp=None,
            ignore_utags=None,
            text2tokenstring=None
    ):
        '''
        Initialize instance for building a space-delimitted string from the
        given text with each token of the form:
    
            <lemma.lower()>_<xpos>
    
        For tokens without a <upos> value in the given ignore_utags.
    
        :param stanza_nlp: The stanza pipeline to use for parsing, using the
            default if None
        :param ignore_utags: The utags of tokens to drop
        :param text2tokenstring: A cache of token strings for input text
        :return: The resulting space-delimited token string
        '''
        self._stanza_nlp = stanza_nlp
        self.ignore_utags = ignore_utags
        self._text2docdf = dict()
        self._text2tokenstring = (
            text2tokenstring
            if text2tokenstring is not None
            else dict()
        )

    @property
    def stanza_nlp(self):
        '''
        Get the stanza_nlp pipeline.
        '''
        if self._stanza_nlp is None:
            self._stanza_nlp = default_stanza_pipeline()
        return self._stanza_nlp

    def build_string(self, text, ignore_utags_override=None):
        '''
        Build a space-delimitted string from the given text with each token of
        the form:
    
            <lemma.lower()>_<xpos>
    
        For tokens without a <upos> value in the instance's ignore_utags.
    
        :param text: The text to process
        :param ignore_utags_override: Override for instance value
        :return: The resulting space-delimited token string
        '''
        tokens = (
            self._text2tokenstring.get(text, None)
            if ignore_utags_override is None and self.ignore_utags is None
            else None
        )
        if tokens is None:
            doc_df = self._text2docdf.get(text, None)
            if doc_df is None:
                doc = self.stanza_nlp(text)
                doc_df = doc2df(doc)
                self._text2docdf[text] = doc_df
            tokens = self._do_build_string(
                doc_df,
                ignore_utags_override=ignore_utags_override
            )
            if ignore_utags_override is None and self.ignore_utags is None:
                # NOTE: Only cache token strings without ignored utags
                self._text2tokenstring[text] = tokens
        return tokens

    def _do_build_string(self, doc_df, ignore_utags_override=None):
        tokens = ''
        ignore_utags = ignore_utags_override
        if ignore_utags is None:
            ignore_utags = self.ignore_utags
        if ignore_utags is not None and len(ignore_utags) > 0:
            doc_df = doc_df[
                ~doc_df['upos'].isin(ignore_utags)
            ]
        tokens = ' '.join(doc_df[['lemma', 'xpos']].apply(
            lambda row: '_'.join([
                row['lemma'].lower() if not pd.isna(row['lemma']) else '',
                row['xpos']
            ]), axis=1))
        return tokens


class StanzaResults:
    def __init__(self, pipeline, text, doc):
        self.pipeline = pipeline
        self.text = text
        self.doc = doc
        self._df = None

    @property
    def df(self):
        if self._df is None:
            self._df = doc2df(self.doc)
        return self._df


class StanzaProcessor:
    def __init__(self, model_dir, stanza_processors, lang='en', nlp=None):
        self.lang = lang
        self.model_dir = model_dir
        self.stanza_processors = stanza_processors
        self._nlp = nlp
        self._tsb = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = stanza.Pipeline(
                self.lang,
                model_dir=self.model_dir,
                processors=self.stanza_processors,
                download_method=stanza.DownloadMethod.REUSE_RESOURCES
            )
        return self._nlp

    def process(self, text):
        return StanzaResults(self.nlp, text, self.nlp(text))

    @property
    def token_string_builder(self):
        if self._tsb is None:
            self._tsb = TokenStringBuilder(self.nlp)
        return self._tsb


class CoreNLPResults:
    def __init__(self, annotations, deptype=None):
        self.ann = annotations
        self._sentences_df = {}
        self._parse_tree = None
        self._corefs_df = None
        self._deptype = deptype

    @property
    def deptype(self):
        if self._deptype is None:
            sent_data = self.get_sentence(0)
            if sent_data is not None:
                for dtype in [
                    'enhancedPlusPlusDependencies',
                    'enhancedDependencies',
                    'basicDependencies',
                ]:
                    if dtype in sent_data:
                        self._deptype = dtype
                        break
        return self._deptype

    @deptype.setter
    def deptype(self, new_deptype):
        self._deptype = new_deptype

    def get_annotation_types(self):
        return list(self.ann.keys())

    def get_data(self, annotation_type, missing_value=None):
        return self.ann.get(annotation_type, missing_value)

    @property
    def sentences(self):
        return self.get_data('sentences', missing_value=[])

    @property
    def sentences_count(self):
        return len(self.sentences)

    @property
    def corefs(self):
        return self.get_data('corefs', missing_value={})

    @property
    def corefs_count(self):
        return len(self.corefs)

    @property
    def coref_keys(self):
        return list(self.corefs.keys())

    def _build_sentences_df(self, data_type):
        return pd_utils.dicts2df(
            [
                sent_data.get(data_type, [])
                for sent_data in self.sentences
            ],
            rename={'index': 'word_id'},
            item_id='sent_id',
        )

    def get_sentences_df(self, data_type):
        if data_type and data_type not in self._sentences_df:
            self._sentences_df[data_type] = self._build_sentences_df(data_type)
        return self._sentences_df.get(data_type, None)

    @property
    def dependencies_df(self):
        return self.get_sentences_df(self.deptype)

    @property
    def tokens_df(self):
        return self.get_sentences_df('tokens')

    @property
    def parse_tree(self):
        if self._parse_tree is None:
            self._parse_tree = dk_tree.build_tree_from_string(
                '(PARAGRAPH ' +
                ' '.join(
                    str(sent_data['parse'])
                    for sent_data in self.sentences
                ) +
                ')'
            )
        return self._parse_tree

    @property
    def corefs_df(self):
        if self._corefs_df is None:
            self._corefs_df = pd.concat([
                self.get_corefs_df(coref_key)
                for coref_key in self.coref_keys
            ]).reset_index(names='coref_idx')
        return self._corefs_df

    def get_sentence(self, sentence_idx=0):
        sent_data = None
        adata = self.sentences
        if len(adata) > sentence_idx:
            sent_data = adata[sentence_idx]
        return sent_data

    def get_sentence_data(self, sentence_idx, data_type, asdf=False):
        data = None
        sent_data = self.get_sentence(sentence_idx)
        if sent_data is not None:
            data = sent_data.get(data_type, None)
            if data is not None and asdf:
                data = pd.DataFrame.from_records(data)
        return data

    def get_tokens_df(self, sentence_idx=0):
        return self.get_sentence_data(sentence_idx, 'tokens', asdf=True)

    def get_dependencies_df(self, sentence_idx=0, deptype=None):
        if deptype == None:
            deptype = self.deptype
        return self.get_sentence_data(sentence_idx, deptype, asdf=True)

    def get_parse_tree(self, sentence_idx=0):
        parse_tree = None
        parse_data = self.get_sentence_data(sentence_idx, 'parse', asdf=False)
        if parse_data:
            parse_tree = dk_tree.build_tree_from_string(parse_data)
        return parse_tree

    def get_corefs_df(self, coref_key):
        df = None
        coref_data = self.corefs.get(coref_key, None)
        if coref_data is not None:
            df = pd_utils.dicts2df(
                coref_data,
                rename={'id': 'word_id'},
                item_id=None
            )
            df['coref_key'] = coref_key
        return df


class CoreNLPAnnotator:
    def __init__(
            self,
            corenlp_ip,
            corenlp_annotators,
            port=9000,
            lang='english',
            corenlp_client_builder=CoreNLPClient,
    ):
        self.ip = corenlp_ip
        self.annotators = corenlp_annotators
        self.port = port
        self.lang = lang
        self.client_builder = corenlp_client_builder

    def annotate(self, text, annotators=None):
        if annotators is None:
            annotators = self.annotators
        ann = None
        if annotators:
            with self.client_builder(
                    endpoint=f'http://{self.ip}:{self.port}',
                    annotators=annotators,
                    start_server=stanza.server.StartServer.DONT_START,
            ) as client:
                ann = client.annotate(
                    text,
                    properties=self.lang,
                    output_format='json',
                )
        return CoreNLPResults(ann) if ann else None
