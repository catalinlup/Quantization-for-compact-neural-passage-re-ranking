import abc
import time
import pickle
import logging
from enum import Enum
from pathlib import Path
from queue import PriorityQueue
from collections import OrderedDict, defaultdict
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Set, Tuple, Union
import h5py
import faiss
from faiss.loader import *
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine

from fast_forward.ranking import Ranking
from fast_forward.encoder import QueryEncoder
from .util import interpolate, cpointer_to_array
import math



LOGGER = logging.getLogger(__name__)


class Mode(Enum):
    """Enum used to set the retrieval mode of an index."""

    PASSAGE = 1
    MAXP = 2
    FIRSTP = 3
    AVEP = 4


class Index(abc.ABC):
    """Abstract base class for Fast-Forward indexes."""

    def __init__(
        self,
        encoder: QueryEncoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
        socoring_function: Union['dot', 'max_sim'] = 'dot'
    ) -> None:
        """Constructor.

        Args:
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.
        """
        super().__init__()

        if socoring_function not in {'dot', 'max_sim'}:
            raise Exception('Invalid scoring function')

        self.encoder = encoder
        self.mode = mode
        self._encoder_batch_size = encoder_batch_size
        self.scoring_function = socoring_function


    def encode(self, queries: Sequence[str]) -> List[np.ndarray]:
        """Encode queries.

        Args:
            queries (Sequence[str]): The queries to encode.

        Raises:
            RuntimeError: When no query encoder exists.

        Returns:
            List[np.ndarray]: The query representations.
        """
        if self._encoder is None:
            raise RuntimeError("This index does not have a query encoder.")

        result = []
        for i in range(0, len(queries), self._encoder_batch_size):
            batch = queries[i : i + self._encoder_batch_size]
            result.extend(self._encoder.encode(batch))
        return result

    @property
    def encoder(self) -> QueryEncoder:
        """Return the query encoder.

        Returns:
            QueryEncoder: The encoder.
        """
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: QueryEncoder) -> None:
        """Set the query encoder.

        Args:
            encoder (QueryEncoder): The encoder.
        """
        self._encoder = encoder

    @property
    def mode(self) -> Mode:
        """Return the indexing mode.

        Returns:
            Mode: The mode.
        """
        return self._mode

    @mode.setter
    def mode(self, mode: Mode) -> None:
        """Set the indexing mode.

        Args:
            mode (Mode): The indexing mode.
        """
        assert isinstance(mode, Mode)
        self._mode = mode

    @property
    def doc_ids(self) -> Set[str]:
        """Return all unique document IDs.

        Returns:
            Set[str]: The document IDs.
        """
        return self._get_doc_ids()

    @abc.abstractmethod
    def _get_doc_ids(self) -> Set[str]:
        """Return all unique document IDs.

        Returns:
            Set[str]: The document IDs.
        """
        pass

    @property
    def psg_ids(self) -> Set[str]:
        """Return all unique passage IDs.

        Returns:
            Set[str]: The passage IDs.
        """
        return self._get_psg_ids()

    @abc.abstractmethod
    def _get_psg_ids(self) -> Set[str]:
        """Return all unique passage IDs.

        Returns:
            Set[str]: The passage IDs.
        """
        pass

    @abc.abstractmethod
    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[Union[str, None]],
        psg_ids: Sequence[Union[str, None]],
    ) -> None:
        """Add vector representations and corresponding IDs to the index. Each vector is guaranteed to
        have either a document or passage ID associated.

        Args:
            vectors (np.ndarray): The representations, shape (num_vectors, dim).
            doc_ids (Sequence[Union[str, None]]): The corresponding document IDs (may be duplicate).
            psg_ids (Sequence[Union[str, None]]): The corresponding passage IDs (must be unique).
        """
        pass

    def add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[str] = None,
        psg_ids: Sequence[str] = None,
    ) -> None:
        """Add vector representations and corresponding IDs to the index. Only one of "doc_ids" and "psg_ids"
        may be None. For performance reasons, this function should not be called frequently with few items.

        Args:
            vectors (np.ndarray): The representations, shape (num_vectors, dim).
            doc_id (Sequence[str], optional): The corresponding document IDs (may be duplicate). Defaults to None.
            psg_id (Sequence[str], optional): The corresponding passage IDs (must be unique). Defaults to None.

        Raises:
            ValueError: When there are no document IDs and no passage IDs.
        """
        if doc_ids is None and psg_ids is None:
            raise ValueError(
                'At least one of "doc_ids" and "psg_ids" must be provided.'
            )

        num_vectors = vectors.shape[0]
        if num_vectors < 100:
            LOGGER.warning(
                'calling "Index.add()" repeatedly with few vectors may be slow'
            )
        if doc_ids is None:
            doc_ids = [None] * num_vectors
        if psg_ids is None:
            psg_ids = [None] * num_vectors

        assert num_vectors == len(doc_ids) == len(psg_ids)
        self._add(vectors, doc_ids, psg_ids)

    @abc.abstractmethod
    def _get_vectors(
        self, ids: Iterable[str], mode: Mode
    ) -> Tuple[np.ndarray, List[Union[List[int], int, None]]]:
        """Return:
            * A single array containing all vectors necessary to compute the scores for each document/passage.
            * For each document/passage (in the same order as the IDs), either
                * a list of integers (MAXP, AVEP),
                * a single integer (FIRSTP, PASSAGE),
                * None (the document/passage is not indexed and has no vector)

        The integers will be used to get the corresponding representations from the array.
        The output of this function depends on the current mode.

        Args:
            ids (Iterable[str]): The document/passage IDs to get the representations for.
            mode (Mode): The index mode.

        Returns:
            Tuple[np.ndarray, List[Union[List[int], int, None]]]: The vectors and corresponding indices.
        """
        pass
    
    
    def _max_sim(self, Q: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Implements the max_sim operator, which computes 
        """
        scores_per_query = []

        if len(Q.shape) == 2:
            Q = Q.reshape(1, Q.shape[0], Q.shape[1])

        for i in range(Q.shape[0]):
            scores = D @ np.transpose(Q[i, :, :].reshape(1, Q.shape[1], Q.shape[2]), axes=(0, 2, 1))
            scores = np.transpose(scores, axes=(0, 2, 1))
            scores_per_query.append(np.sum(np.max(scores, axis=1), axis=1))

        if len(scores_per_query) == 1:
            return scores_per_query[0]
            
        return np.stack(scores_per_query)
    

    def _compute_scores(self, q_rep: np.ndarray, ids: Iterable[str]) -> Iterator[float]:
        """Compute scores based on the current mode.

        Args:
            q_rep (np.ndarray): Query representation.
            ids (Iterable[str]): Document/passage IDs.

        Yields:
            float: The scores, preserving the order of the IDs.
        """
        vectors, id_indices = self._get_vectors(ids, self.mode)

        if self.scoring_function == 'dot':
            all_scores = np.dot(q_rep, vectors.T)
        else:   
            all_scores = self._max_sim(q_rep, vectors)


        for ind in id_indices:
            if ind is None:
                yield None
            else:
                if self.mode == Mode.MAXP:
                    yield np.max(all_scores[ind])
                elif self.mode == Mode.AVEP:
                    yield np.average(all_scores[ind])
                elif self.mode in (Mode.FIRSTP, Mode.PASSAGE):
                    yield all_scores[ind]

    def _early_stopping(
        self,
        ids: Iterable[str],
        dense_scores: Iterable[float],
        sparse_scores: Iterable[float],
        alpha: float,
        cutoff: int,
    ) -> Dict[str, float]:
        """Interpolate scores with early stopping.

        Args:
            ids (Iterable[str]): Document/passage IDs.
            dense_scores (Iterable[float]): Corresponding dense scores.
            sparse_scores (Iterable[float]): Corresponding sparse scores.
            alpha (float): Interpolation parameter.
            cutoff (int): Cut-off depth.

        Returns:
            Dict[str, float]: Document/passage IDs mapped to scores.
        """
        result = {}
        relevant_scores = PriorityQueue(cutoff)
        min_relevant_score = float("-inf")
        max_dense_score = float("-inf")
        for id, dense_score, sparse_score in zip(ids, dense_scores, sparse_scores):
            if relevant_scores.qsize() >= cutoff:

                # check if approximated max possible score is too low to make a difference
                min_relevant_score = relevant_scores.get_nowait()
                max_possible_score = (
                    alpha * sparse_score + (1 - alpha) * max_dense_score
                )

                # early stopping
                if max_possible_score <= min_relevant_score:
                    break

            if dense_score is None:
                LOGGER.warning(f"{id} not indexed, skipping")
                continue

            max_dense_score = max(max_dense_score, dense_score)
            score = alpha * sparse_score + (1 - alpha) * dense_score
            result[id] = score

            # the new score might be ranked higher than the one we removed
            relevant_scores.put_nowait(max(score, min_relevant_score))
        return result

    def get_scores(
        self,
        ranking: Ranking,
        queries: Dict[str, str],
        alpha: Union[float, Iterable[float]] = 0.0,
        cutoff: int = None,
        early_stopping: bool = False,
    ) -> Dict[float, Ranking]:
        """Compute corresponding dense scores for a ranking and interpolate.

        Args:
            ranking (Ranking): The ranking to compute scores for and interpolate with.
            queries (Dict[str, str]): Query IDs mapped to queries.
            alpha (Union[float, Iterable[float]], optional): Interpolation weight(s). Defaults to 0.0.
            cutoff (int, optional): Cut-off depth (documents/passages per query). Defaults to None.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.

        Raises:
            ValueError: When the cut-off depth is missing for early stopping.

        Returns:
            Dict[float, Ranking]: Alpha mapped to interpolated scores.
        """
        if isinstance(alpha, float):
            alpha = [alpha]

        if early_stopping and cutoff is None:
            raise ValueError("A cut-off depth is required for early stopping.")

        t0 = time.time()

        # batch encode queries
        q_id_list = list(ranking)
        q_reps = self.encode([queries[q_id] for q_id in q_id_list])

        result = {}
        if not early_stopping:
            # here we can simply compute the dense scores once and interpolate for each alpha
            dense_run = defaultdict(OrderedDict)
            for q_id, q_rep in zip(tqdm(q_id_list), q_reps):
                ids = list(ranking[q_id].keys())
                for id, score in zip(ids, self._compute_scores(q_rep, ids)):
                    if score is None:
                        LOGGER.warning(f"{id} not indexed, skipping")
                    else:
                        dense_run[q_id][id] = score
            for a in alpha:
                result[a] = interpolate(
                    ranking, Ranking(dense_run, sort=False), a, sort=True
                )
                if cutoff is not None:
                    result[a].cut(cutoff)
        else:
            # early stopping requries the ranking to be sorted
            # this should normally be the case anyway
            if not ranking.is_sorted:
                LOGGER.warning("input ranking not sorted. sorting...")
                ranking.sort()

            # since early stopping depends on alpha, we have to run the algorithm more than once
            for a in alpha:
                run = defaultdict(OrderedDict)
                for q_id, q_rep in zip(tqdm(q_id_list), q_reps):
                    ids, sparse_scores = zip(*ranking[q_id].items())
                    dense_scores = self._compute_scores(q_rep, ids)
                    scores = self._early_stopping(
                        ids, dense_scores, sparse_scores, a, cutoff
                    )
                    for id, score in scores.items():
                        run[q_id][id] = score
                result[a] = Ranking(run, sort=True, copy=False)
                result[a].cut(cutoff)

        LOGGER.info(f"computed scores in {time.time() - t0}s")
        return result





class H5Index(Index):
    """Fast-Forward index that is held in memory."""

    def __init__(
        self,
        encoder: QueryEncoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
        socoring_function: Union['dot', 'max_sim'] = 'dot'

    ) -> None:
        """Constructor.

        Args:
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.
        """
        
        if mode != Mode.PASSAGE:
            raise Exception('Only passage mode is supported!')

        self._index_file = None
        self._doc_ids = []
        self._psg_ids = []
        self._doc_id_to_idx = defaultdict(list)
        self._psg_id_to_idx = {}
        super().__init__(encoder, mode, encoder_batch_size, socoring_function)

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[Union[str, None]],
        psg_ids: Sequence[Union[str, None]],
    ) -> None:

        raise Exception('This is a static index. New vectors cannot be added.')

    def _get_doc_ids(self) -> Set[str]:
        return set(self._doc_id_to_idx.keys())

    def _get_psg_ids(self) -> Set[str]:
        return set(self._psg_id_to_idx.keys())

    def _get_vectors(
        self, ids: Iterable[str], mode: Mode
    ) -> Tuple[np.ndarray, List[Union[List[int], int, None]]]:
        # a list of all vectors to take from the main vector array

        if mode != Mode.PASSAGE:
            raise Exception('Only passage mode is supported!')

        vector_indices = []

        # for each ID, keep a list of indices to get the corresponding vectors from "vector_indices"
        id_indices = []

        for i, id in enumerate(ids):
            vector_indices.append(int(self._psg_id_to_idx[id]))
            id_indices.append(i)

        vector_indices = np.array(vector_indices)

        # original order -> sorted order
        original_order_to_sorted_order = np.argsort(vector_indices)

        sorted_order_to_original_order = np.zeros_like(original_order_to_sorted_order)

        for i, v in enumerate(original_order_to_sorted_order):
            sorted_order_to_original_order[v] = i



        with h5py.File(self._index_file) as fp:
            vectors_sorted = fp['vectors'][vector_indices[original_order_to_sorted_order], :]

            vectors_original_order = vectors_sorted[sorted_order_to_original_order]


            
            return vectors_original_order, id_indices

    def save(self, target: Path) -> None:
        """Save the index in a file on disk.

        Args:
            target (Path): Target file to create.
        """
        raise Exception('This is a static index, it cannot be saved!')

    @classmethod
    def from_disk(
        cls,
        index_file: Path,
        encoder: QueryEncoder = None,
        encoder_batch_size: int = 32,
    ) -> "InMemoryIndex":
        """Read an index from disk.

        Args:
            index_file (Path): The index file.
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.

        Returns:
            InMemoryIndex: The index.
        """
        LOGGER.info(f"reading {index_file}")
        with h5py.File(index_file) as fp:
            ids = list(fp["docids"][:])

        # print(ids[:100])
        index = cls(encoder, Mode.PASSAGE, encoder_batch_size)

        index._index_file = index_file
        index._doc_ids = list(map(lambda x: str(x), ids))
        index._psg_ids = index._doc_ids
        index._doc_id_to_idx = {str(doc_id): i for i, doc_id in enumerate(index._doc_ids)}
        index._psg_id_to_idx = {str(psg_id): i for i, psg_id in enumerate(index._psg_ids)}
        


        index.mode = Mode.PASSAGE
        return index




class ColBertH5Index(H5Index):

    def _get_vectors(
        self, ids: Iterable[str], mode: Mode
    ) -> Tuple[np.ndarray, List[Union[List[int], int, None]]]:
        # a list of all vectors to take from the main vector array

        if mode != Mode.PASSAGE:
            raise Exception('Only passage mode is supported!')

        flat_vector_indices = []
        

        # for each ID, keep a list of indices to get the corresponding vectors from "vector_indices"
        id_indices = []

        # keep track of the largest number of vectors per passage
        max_vectors_per_passage = 0

        # print(self._psg_id_to_idx)

        for i, id in enumerate(ids):
            flat_vector_indices += self._psg_id_to_idx[id]
            max_vectors_per_passage = max(max_vectors_per_passage, len(self._psg_id_to_idx[id]))
            id_indices.append(i)

        flat_vector_indices = np.array(flat_vector_indices)
        

        # # original order -> sorted order
        original_order_to_sorted_order = np.argsort(flat_vector_indices)

        sorted_order_to_original_order = np.zeros_like(original_order_to_sorted_order)

        for i, v in enumerate(original_order_to_sorted_order):
            sorted_order_to_original_order[v] = i


        # read the vectors from the index file
        with h5py.File(self._index_file) as fp:
            vectors_sorted = fp['vectors'][flat_vector_indices[original_order_to_sorted_order], :]
            psgids_sorted = fp['docids'][flat_vector_indices[original_order_to_sorted_order]]

            vectors_original_order = vectors_sorted[sorted_order_to_original_order]
            psgids_sorted_original_order = psgids_sorted[sorted_order_to_original_order]


        vectors_original_order_grouped = []
        for i, id in enumerate(ids):
            indices_for_group =  self._psg_id_to_idx[id]
            tensor_for_group = np.zeros((max_vectors_per_passage, vectors_original_order.shape[1]))
            tensor_for_group[:len(indices_for_group)] = vectors_original_order[np.where(psgids_sorted_original_order == int(id))]

            vectors_original_order_grouped.append(tensor_for_group)

       
        vectors_original_order_grouped = np.stack(vectors_original_order_grouped)

        return vectors_original_order_grouped, id_indices

    @classmethod
    def from_disk(
        cls,
        index_file: Path,
        encoder: QueryEncoder = None,
        encoder_batch_size: int = 32,
    ) -> "InMemoryIndex":
        """Read an index from disk.

        Args:
            index_file (Path): The index file.
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.

        Returns:
            InMemoryIndex: The index.
        """
        LOGGER.info(f"reading {index_file}")

        DATASET_SIZE = 8841823
        ids = list(range(0, DATASET_SIZE))
        index = cls(encoder, Mode.PASSAGE, encoder_batch_size, 'max_sim')

        index._index_file = index_file
        index._doc_ids = list(map(lambda x: str(x), ids))
        index._psg_ids = index._doc_ids

        index._psg_id_to_idx =  defaultdict(lambda: [])
        index._doc_id_to_idx = defaultdict(lambda: [])

        for i, id in enumerate(ids):
            index._psg_id_to_idx[str(id)].append(i)
            index._doc_id_to_idx[str(id)].append(i)
        

        index.mode = Mode.PASSAGE
        return index


class FaissIVFPQIndex(H5Index):
    """
    FaissIVFPQ index that is held in memory.
    """

    def __init__(self, 
                 encoder: QueryEncoder = None, 
                 mode: Mode = Mode.PASSAGE, 
                 encoder_batch_size: int = 32) -> None:
        """Constructor

        Args:
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.
        """

        if mode != Mode.PASSAGE:
            raise Exception('Only passage mode is supported!')
        
        self._index_file = None # stores the path of the h5 index file
        self._faiss_ivfpq_index: faiss.IndexIVFPQ = None # stores the IVFPQ index (used in IVFPQ augmentation)
        self._num_cells = 0
        self._cell_to_doc_idxs = dict()
        self._doc_idx_to_cell = dict()
        self._doc_ids = []
        self._psg_ids = []
        self._doc_id_to_idx = defaultdict(list)
        self._psg_id_to_idx = {}
        super().__init__(encoder, mode, encoder_batch_size)


    

    def _ivf_search(self, target_doc_ids: List[str]) -> List[str]:
        """
        Perform an _ivf_augmentation search base don the provided doc_ids
        """

        doc_idxs = [self._doc_id_to_idx[doc_id] for doc_id in target_doc_ids]
        target_cell_ids = list(set([self._doc_idx_to_cell[doc_idx] for doc_idx in doc_idxs]))

        # fetch the idxs of the documents that fell into the same voronoi cells as the target_doc_ids
        aug_doc_idxs = []
        for target_cell_id in target_cell_ids:
            aug_doc_idxs += self._cell_to_doc_idxs[target_cell_id]

        aug_doc_idxs = list(set(aug_doc_idxs))

        # convert the idxs to doc_ids

        aug_doc_ids = [self._psg_ids[idx] for idx in aug_doc_idxs]

        return aug_doc_ids




    def _ivfpq_retrieval_set_augmentation(self, ids: List[str], top_k: int) -> List[str]:
        """
        Augments the retrievel set using the IVFPQ index for improved performance.

        Args:
            ids: the ids of the vectors in the retrieval set.
            top_k: how many additional vectors to add, for each vector in the retrieval set


        Returns an augmented id set, containing additional ids returned via IVFPQ augmentation.
        """

        if self._faiss_ivfpq_index == None:
            raise Exception('Faiss IVFPQ index not intialized')
        
        # if top_k is 0 (or None), skip augmentation 
        if top_k == 0 or top_k == None:
            return ids
        

        vector_ids: np.ndarray = self._ivf_search(ids)

        # remove the ids that already exost in the original set
        vector_ids_no_originals = list(set(vector_ids).difference(set(ids)))

        # print(vector_ids_no_originals)

        # take a sample of the vector_ids_no_originals
        vector_ids_sample = list(np.random.choice(np.array(vector_ids_no_originals), size=top_k, replace=False))

        # print('Augmented')
        # print(vector_ids_sample)
        
        # print(vector_ids)

        # print(len(ids))
        # print(len(vector_ids_sample))

        # add the new vector ids to the original id set
        augmented_ids =  vector_ids_sample

        return augmented_ids
    

    def get_scores(
        self,
        ranking: Ranking,
        queries: Dict[str, str],
        alpha: Union[float, Iterable[float]] = 0.0,
        top_k: Union[float, Iterable[int]] = 0,
        cutoff: int = None,
    ) -> Dict[float, Ranking]:
        """Compute corresponding dense scores for a ranking and interpolate.

        Args:
            ranking (Ranking): The ranking to compute scores for and interpolate with.
            queries (Dict[str, str]): Query IDs mapped to queries.
            alpha (Union[float, Iterable[float]], optional): Interpolation weight(s). Defaults to 0.0.
            top_k (Union[float, Iterable[int]], optional): The number of additional documents to add when using IVFPQ augmentation
            cutoff (int, optional): Cut-off depth (documents/passages per query). Defaults to None.
        Raises:
            ValueError: When the cut-off depth is missing for early stopping.

        Returns:
            Dict[float, Ranking]: Alpha mapped to interpolated scores.
        """
        if isinstance(alpha, float):
            alpha = [alpha]

        if isinstance(top_k, int):
            top_k = [top_k]

        t0 = time.time()

        # batch encode queries
        q_id_list = list(ranking)
        q_reps = self.encode([queries[q_id] for q_id in q_id_list])

        result = {}
        # here we can simply compute the dense scores once and interpolate for each alpha
        print(top_k)
        for tk in top_k:
            result[tk] = {}
            dense_run = defaultdict(OrderedDict)

            
            for q_id, q_rep in zip(tqdm(q_id_list), q_reps):
                ids = list(ranking[q_id].keys())


                augmented_ids = self._ivfpq_retrieval_set_augmentation(ids, tk)

                for id, score in zip(augmented_ids, self._compute_scores(q_rep, augmented_ids)):
                    if score is None:
                        LOGGER.warning(f"{id} not indexed, skipping")
                    else:
                        # print(score)
                        dense_run[q_id][id] = score
                
                

            for a in alpha:
                result[tk][a] = interpolate(
                    ranking, Ranking(dense_run, sort=False), a, sort=True
                )                
        

        LOGGER.info(f"computed scores in {time.time() - t0}s")
        return result


    @classmethod
    def from_disk(
        cls,
        index_file: Path,
        ivfpq_index_file: Path,
        encoder: QueryEncoder = None,
        encoder_batch_size: int = 32
    ) -> "FaissIVFPQIndex":
        """Read an index from disk.

        Args:
            index_file (Path): The path to the H5 index
            ivfpq_index_file: The path to the IVFPQ index
            encoder : the query encoders
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.

        Returns:
            InMemoryIndex: The index
        """

        LOGGER.info(f"reading {index_file}")
        with h5py.File(index_file) as fp:
            ids = list(fp["docids"][:])

        index = cls(encoder, Mode.PASSAGE, encoder_batch_size)

        index._index_file = index_file
        index._faiss_ivfpq_index = faiss.read_index(str(ivfpq_index_file))

        # map the cell id to docs ids and each doc id to its cell.  
        index._num_cells = index._faiss_ivfpq_index.invlists.nlist
        index._cell_to_doc_idxs = {cell_id: list(cpointer_to_array(index._faiss_ivfpq_index.invlists.get_ids(cell_id), index._faiss_ivfpq_index.invlists.list_size(cell_id), np.int64)) for cell_id in range(index._num_cells)}

        index._doc_idx_to_cell = dict()
        for cell_id, doc_ids in index._cell_to_doc_idxs.items():
            for doc_id in doc_ids:
                index._doc_idx_to_cell[doc_id] = cell_id

        index._doc_ids = list(map(lambda x: str(x), ids))
        index._psg_ids = index._doc_ids
        index._doc_id_to_idx = {str(doc_id): i for i, doc_id in enumerate(index._doc_ids)}
        index._psg_id_to_idx = {str(psg_id): i for i, psg_id in enumerate(index._psg_ids)}

        index.mode = Mode.PASSAGE
        return index



class InMemoryIndex(Index):
    """Fast-Forward index that is held in memory."""

    def __init__(
        self,
        encoder: QueryEncoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
    ) -> None:
        """Constructor.

        Args:
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.
        """
        self._vectors = None
        self._doc_ids = []
        self._psg_ids = []
        self._doc_id_to_idx = defaultdict(list)
        self._psg_id_to_idx = {}
        super().__init__(encoder, mode, encoder_batch_size)

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[Union[str, None]],
        psg_ids: Sequence[Union[str, None]],
    ) -> None:
        if self._vectors is None:
            idx = 0
            self._vectors = vectors.copy()
        else:
            idx = self._vectors.shape[0]
            self._vectors = np.append(self._vectors, vectors, axis=0)

        for doc_id, psg_id in zip(doc_ids, psg_ids):
            if doc_id is not None:
                self._doc_id_to_idx[doc_id].append(idx)
            if psg_id is not None:
                assert psg_id not in self._psg_id_to_idx
                self._psg_id_to_idx[psg_id] = idx
            idx += 1

        self._doc_ids.extend(doc_ids)
        self._psg_ids.extend(psg_ids)

    def _get_doc_ids(self) -> Set[str]:
        return set(self._doc_id_to_idx.keys())

    def _get_psg_ids(self) -> Set[str]:
        return set(self._psg_id_to_idx.keys())

    def _get_vectors(
        self, ids: Iterable[str], mode: Mode
    ) -> Tuple[np.ndarray, List[Union[List[int], int, None]]]:
        # a list of all vectors to take from the main vector array
        vector_indices = []

        # for each ID, keep a list of indices to get the corresponding vectors from "vector_indices"
        id_indices = []
        i = 0

        if mode in (Mode.MAXP, Mode.AVEP):
            for id in ids:
                if id in self._doc_id_to_idx:
                    doc_indices = self._doc_id_to_idx[id]
                    vector_indices.extend(doc_indices)
                    id_indices.append(list(range(i, i + len(doc_indices))))
                    i += len(doc_indices)
                else:
                    id_indices.append(None)
        elif mode == Mode.FIRSTP:
            for id in ids:
                if id in self._doc_id_to_idx:
                    vector_indices.append(self._doc_id_to_idx[id][0])
                    id_indices.append(i)
                    i += 1
                else:
                    id_indices.append(None)
        elif mode == Mode.PASSAGE:
            for id in ids:
                if id in self._psg_id_to_idx:
                    vector_indices.append(self._psg_id_to_idx[id])
                    id_indices.append(i)
                    i += 1
                else:
                    id_indices.append(None)
        else:
            LOGGER.error(f"invalid mode: {mode}")
        return self._vectors[vector_indices], id_indices

    def save(self, target: Path) -> None:
        """Save the index in a file on disk.

        Args:
            target (Path): Target file to create.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"writing {target}")
        with open(target, "wb") as fp:
            pickle.dump((self._vectors, self._doc_ids, self._psg_ids), fp)

    @classmethod
    def from_disk(
        cls,
        index_file: Path,
        encoder: QueryEncoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
    ) -> "InMemoryIndex":
        """Read an index from disk.

        Args:
            index_file (Path): The index file.
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.

        Returns:
            InMemoryIndex: The index.
        """
        LOGGER.info(f"reading {index_file}")
        with open(index_file, "rb") as fp:
            vectors, doc_ids, psg_ids = pickle.load(fp)

        index = cls(encoder, mode, encoder_batch_size)
        if vectors is not None:
            index.add(vectors, doc_ids, psg_ids)
        index.mode = mode
        return index


def create_coalesced_index(
    source_index: Index,
    target_index: Index,
    delta: float,
    distance: Callable[[np.ndarray, np.ndarray], float] = cosine,
    buffer_size: int = None,
) -> None:
    """Create a compressed index using sequential coalescing.

    Args:
        source_index (Index): The source index. Should contain multiple vectors for each document.
        target_index (Index): The target index. Must be empty.
        delta (float): The coalescing threshold.
        distance (Callable[[np.ndarray, np.ndarray], float]): The distance function. Defaults to cosine.
        buffer_size (int, optional): Use a buffer instead of adding all vectors at the end. Defaults to None.
    """
    assert len(target_index.doc_ids) == 0
    buffer_size = buffer_size or len(source_index.doc_ids)

    def _coalesce(P):
        P_new = []
        A = []
        A_avg = None
        first_iteration = True
        for v in P:
            if first_iteration:
                first_iteration = False
            elif distance(v, A_avg) >= delta:
                P_new.append(A_avg)
                A = []
            A.append(v)
            A_avg = np.mean(A, axis=0)
        P_new.append(A_avg)
        return P_new

    vectors, doc_ids = [], []
    for doc_id in tqdm(source_index.doc_ids):

        # check if buffer is full
        if len(vectors) == buffer_size:
            target_index.add(np.array(vectors), doc_ids=doc_ids)
            vectors, doc_ids = [], []

        v_old, _ = source_index._get_vectors([doc_id], Mode.MAXP)
        v_new = _coalesce(v_old)
        vectors.extend(v_new)
        doc_ids.extend([doc_id] * len(v_new))

    if len(vectors) > 0:
        target_index.add(np.array(vectors), doc_ids=doc_ids)

    assert source_index.doc_ids == target_index.doc_ids




class FaissPQIndex(Index):
    """
    FaissPQ index held on disk.
    """

    def __init__(
        self,
        encoder: QueryEncoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
    ) -> None:
        """Constructor.

        Args:
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.
        """
        self._doc_ids = []
        self._psg_ids = []
        self._doc_id_to_idx = defaultdict(list)
        self._psg_id_to_idx = {}

        # stores the data used in quantization

        self.codebook = None # the codebook
        self.quantized_index = None # the quantized index
        self.M = None # the number of subspace divisions
        self.K = None # the number of centroids per dimension
        self.N = None # the number of vectors in the index

        super().__init__(encoder, mode, encoder_batch_size)

    
    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[Union[str, None]],
        psg_ids: Sequence[Union[str, None]],
    ) -> None:
        raise Exception('This index is static. Adding vectors dynamically is not supported.')
    
    def save(self, target: Path) -> None:
        raise Exception('This index is static. Saving it is not supported.')
    

    def _get_doc_ids(self) -> Set[str]:
        return set(self._doc_id_to_idx.keys())

    def _get_psg_ids(self) -> Set[str]:
        return set(self._psg_id_to_idx.keys())
    

    

    @classmethod
    def from_disk(
        cls,
        index_file: Path,
        encoder: QueryEncoder = None,
        encoder_batch_size: int = 32,
    ) -> "FaissPQIndex":
        """
        Reads a FaissPQIndex from disk. The index is static and it only works in PASSAGE MODE.

        Args:
            N -- the number of vectors in the index.
            codebook_fike_path -- the filpath for the codebook.
            quantized_index_file_path -- the filepath to the quantized index.
            M -- the number of subspace division
            K -- the number of centroids per cluster
            ids -- the ids used as identifiers for the documents / passages
            encode -- the query encoded to be used
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.

        Returns:
            InMemoryIndex: The index.
        """
        LOGGER.info(f"reading {index_file}")
        with open(index_file, "rb") as fp:
            index_obj = pickle.load(fp)
            codebook = index_obj.get('codebook')
            quantized_index = index_obj.get('quantized_index')
            ids = index_obj.get('doc_ids')
            M = index_obj.get('M')
            K = index_obj.get('K')
            N = index_obj.get('vector_size')

      

        index = cls(encoder, Mode.PASSAGE, encoder_batch_size)
        index.codebook = codebook
        index.quantized_index = quantized_index
        index.M = M
        index.K = K
        index.N = N

        index._doc_ids = list(map(lambda x: str(x), ids))
        index._psg_ids = index._doc_ids
        index._doc_id_to_idx = {str(doc_id): i for i, doc_id in enumerate(index._doc_ids)}
        index._psg_id_to_idx = {str(psg_id): i for i, psg_id in enumerate(index._psg_ids)}


        return index


    def _reconstruct_vector(self, vector_index) -> np.ndarray:
        """
        Reconstructs a vector based on a FaissPQ quantized index.
        """

        codebook = self.codebook
        quantized_index = self.quantized_index
        K = self.K
        M = self.M

        bits_per_vector = int(math.ceil(math.log2(K)))

        offset_ending = 8 * quantized_index.shape[1] - bits_per_vector * M


        bytes = quantized_index[vector_index, :]
        bit_strings = ['{0:08b}'.format(x)[::-1] for x in bytes]
        bit_string = ''.join(bit_strings)

        bit_string_with_offsets = bit_string[: len(bit_string) - offset_ending]

        sub_vecs = []

        for i in range(M):
            bit_string_for_index = bit_string_with_offsets[i * bits_per_vector : (i + 1) * bits_per_vector]
            index = int(bit_string_for_index[::-1], 2)
            sub_vecs.append(codebook[i, index, :])

        return np.concatenate(sub_vecs)
    
    def _get_vectors(self, ids: Iterable[str], mode: Mode) -> Tuple[np.ndarray, List[Union[List[int], int, None]]]:
        """
        Returns the reconstructed vectors of the provided PASSAGE IDs
        """

        # print(mode)

        if mode != Mode.PASSAGE:
            raise Exception('Index only works for passage mode!')
        
        vector_indices = []

        # for each ID, keep a list of indices to get the corresponding vectors from "vector_indices"
        id_indices = []
        i = 0

        for i, id in enumerate(ids):
            vector_indices.append(int(self._psg_id_to_idx[id]))
            id_indices.append(i)
          
        
        # reconstruct the vectors based on the stored PQ compressed index
        vectors = [self._reconstruct_vector(i) for i in vector_indices]

        vectors = np.vstack(vectors)

        return vectors, id_indices

        

class WolfPQIndex(FaissPQIndex):

    def _reconstruct_vector(self, vector_index) -> np.ndarray:
        """
        Reconstruct the vector for a WolfPQIndex.
        """
        codebook = self.codebook
        quantized_index = self.quantized_index
        K = self.K
        M = self.M

        sub_vecs = []

        for i in range(M):
            sub_vecs.append(codebook[i][quantized_index[vector_index][i]])

        return np.concatenate(sub_vecs)