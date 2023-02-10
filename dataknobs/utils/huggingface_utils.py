import numpy as np
import os
import pandas as pd
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer  #, util
from typing import List
from dataknobs.util.sql_utils import RecordFetcher


DEFAULT_HF_EMBEDDING_MODEL = 'multi-qa-MiniLM-L6-cos-v1'

VECTOR_INDEX_COLNAME = 'idx'
DOT_SCORE_COLNAME = 'dot_score'
SCALED_SCORE_COLNAME = 'scaled_score'


class DotCompare:
    '''
    Wrapper around dot product comparison results.
    '''

    def __init__(self, dot_product, record_fetcher=None):
        self.record_fetcher = record_fetcher
        self.dot_product = dot_product
        self.ordered = np.argsort(self.dot_product, axis=0)[::-1]
        self.values = self.dot_product[self.ordered]

    def get_dataframe(self, vec_idx=0, pov='similar', top_n=50, raw_results=False):
        '''
        Get a dataframe representing the results for the identified vector.

        :param vec_idx:  For multivector comparisons, the index of the (comparison) vector
            of interest.
        :param pov: Comparison point of view: 'similar', 'dissimilar', 'orthogonal'
        :param top_n: If non-zero, limit the comparison results to the top N vectors
        :param raw_results: If True or if no record_fetcher has been provided,
            then only the raw scored results are returned; otherwise, the
            dataframe will also include full record information.
        :return: A pandas DataFrame with columns: ['idx', 'dot_score', 'scaled_score']
            indicating:
                the vector (store) index (for the identified comparison store index)
                sorted from most to least applicable to the requested point of view.
                with a scaled_score representing degree of match wrt the pov
        '''
        df = None

        # Construct dataframe
        if len(self.dot_product.shape) == 1:
            df = pd.DataFrame(
                zip(self.ordered, self.values),
                columns=[VECTOR_INDEX_COLNAME, DOT_SCORE_COLNAME],
            )
        else:
            df = pd.DataFrame(
                zip(self.ordered[:,vec_idx], self.values[:,vec_idx][:,vec_idx]),
                columns=[VECTOR_INDEX_COLNAME, DOT_SCORE_COLNAME],
            )
        if pov != 'similar':  # already sorted by similarity
            key = (lambda x: x.abs()) if pov == 'orthogonal' else None
            df.sort_values(
                by=DOT_SCORE_COLNAME, axis=0, ascending=True, key=key, inplace=True
            )

        # Trim to size
        if top_n > 0:
            df = df[:top_n]

        # Add scaled score
        if pov == 'orthogonal':
            df[SCALED_SCORE_COLNAME] = 1.0 - df[DOT_SCORE_COLNAME].abs()
        else:
            scaled_score = 0.5 * (df[DOT_SCORE_COLNAME] + 1)
            if pov == 'similar':
                df[SCALED_SCORE_COLNAME] = scaled_score
            else:  # 'dissimilar'
                df[SCALED_SCORE_COLNAME] = 1.0 - scaled_score

        # Merge with full records
        if self.record_fetcher is not None and not raw_results:
            df = self._merge_records(df)

        return df

    def _merge_records(self, df):
        # NOTE: df -- 0-based embedding vectors
        if self.record_fetcher is not None:
            recs_df = self.record_fetcher.get_records(
                df[VECTOR_INDEX_COLNAME].to_list(),
                one_based=False,  # Vector indices are always 0-based
            ).copy()  # copy to not munge original records

            # Create vector index column in records (copy)
            offset = -1 if self.record_fetcher.one_based else 0
            recs_df.loc[:,VECTOR_INDEX_COLNAME] = recs_df[self.record_fetcher.id_field_name] + offset

            # Merge on common vector index column
            df = recs_df.merge(
                df, how='left',
                on=VECTOR_INDEX_COLNAME,
            ).sort_values(
                SCALED_SCORE_COLNAME, ascending=False,
            )
        return df


class EmbeddingStore(ABC):
    '''
    Abstract base class for an embedding store
    '''
    def __init__(
            self,
            is_normalized: bool = False,
            record_fetcher: RecordFetcher = None,
    ):
        '''
        Create an embedding store.

        :param is_normalized: True if the embeddings are normalized (to a length of 1)
        :param record_fetcher: (Optional) A record_fetcher for access to
            the "record" context for each embedding.
        '''
        self.is_normalized = is_normalized
        self.record_fetcher = record_fetcher

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        ...


class NumpyEmbeddingStore(EmbeddingStore):
    '''
    Base class for an in-memory embedding store as a NumPy array.
    '''

    def __init__(
            self,
            is_normalized: bool = False,
            record_fetcher: RecordFetcher = None,
            vectors: np.ndarray = None,
            texts: List[str] = None,
            vectors_fpath: str = None,
    ):
        '''
        Create an embedding store, initializing with the given vectors or texts, or
        serialized vectors.

        :param is_normalized: True if the vectors are normalized (to a length of 1)
        :param record_fetcher: (Optional) A record_fetcher for access to
            the "record" context for each embedding.
        :param vectors: The vectors to store in this instance.
        :param texts: The texts from which to generate and store vectors.
        :param vectors_fpath: The file path to serialized vectors (.npy file).

        NOTE: Instance vectors are initialized with the first of:
            vectors, vectors_fpath, and texts.
        '''
        super().__init__(is_normalized=is_normalized, record_fetcher=record_fetcher)
        
        self._vectors = vectors
        self.texts = texts
        self.vectors_fpath = vectors_fpath

    def _init_vectors(self):
        if self._vectors is None:
            if self.vectors_fpath is not None:
                vfpath = self.vectors_fpath
                if not vfpath.endswith('.npy') and not os.path.exists(vfpath):
                    vfpath = vfpath + '.npy'
                if os.path.exists(vfpath):
                    self._vectors = np.load(vfpath)
        if self._vectors is None and self.texts is not None:
            #NOTE: This could take a long time!
            self._vectors = self.encode(self.texts)
            if self.vectors_fpath is not None:
                np.save(self.vectors_fpath, self._vectors)

    @property
    def vectors(self):
        if self._vectors is None:
            #NOTE: This could take a long time ... TODO: add progress logging
            self._init_vectors()
        return self._vectors

    def dot_compare(self, vector):
        d = np.dot(self.vectors, vector.T)
        return DotCompare(d, self.record_fetcher)


class HF_Numpy_Embeddings(NumpyEmbeddingStore):
    '''
    A Numpy-based embedding store using HuggingFace SentenceTransformer models.
    '''

    def __init__(
            self,
            model_name: str = DEFAULT_HF_EMBEDDING_MODEL,
            model: SentenceTransformer = None,
            vectors: np.ndarray = None,
            texts: List[str] = None,
            vectors_fpath: str = None,
            record_fetcher: RecordFetcher = None,
    ):
        '''
        Create an embedding store, initializing with the given vectors, texts, or
        serialized vectors.

        :param model_name: The HuggingFace SentenceTransformer model name.
        :param vectors: The vectors to store in this instance.
        :param texts: The texts from which to generate and store vectors.
        :param vectors_fpath: The file path to serialized vectors.
        :param record_fetcher: (Optional) A record_fetcher for access to
            the "record" context for each embedding.
        '''
        super().__init__(
            is_normalized=('-cos-' in model_name or '-dot-' in model_name),
            record_fetcher=record_fetcher,
            vectors=vectors,
            texts=texts,
            vectors_fpath=vectors_fpath,
        )
        self.model_name = model_name
        if model is not None:
            self.model = model
            self.model_name = None
        else:
            self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)
