import logging
from collections import defaultdict
from fast_forward.ranking import Ranking
from faiss.loader import *
import numpy as np


LOGGER = logging.getLogger(__name__)


def cpointer_to_array(v, size, tp):
    """ convert a C pointer to a numpy array """
    dtype = np.dtype(tp)
    a = np.empty(size, dtype=dtype)
    if size > 0:
        memcpy(swig_ptr(a), v, a.nbytes)
    return a


def interpolate(
    r1: Ranking, r2: Ranking, alpha: float, name: str = None, sort: bool = True
) -> Ranking:
    """Interpolate scores. For each query-doc pair:
        * If the pair has only one score, ignore it.
        * If the pair has two scores, interpolate: r1 * alpha + r2 * (1 - alpha).

    Args:
        r1 (Ranking): Scores from the first retriever.
        r2 (Ranking): Scores from the second retriever.
        alpha (float): Interpolation weight.
        name (str, optional): Ranking name. Defaults to None.
        sort (bool, optional): Whether to sort the documents by score. Defaults to True.

    Returns:
        Ranking: Interpolated ranking.
    """
    assert r1.q_ids == r2.q_ids
    results = defaultdict(dict)
    for q_id in r1:
        # docs_ids = set(r1[q_id].keys()).union(set(r2[q_id].keys()))
        docs_ids = set(r2[q_id].keys())

        print('DocIds Length', len(docs_ids))
        
        for doc_id in docs_ids:

            # if the sparse score is not available, consider a sparse score of 0
            results[q_id][doc_id] = (
                alpha * r1[q_id].get(doc_id, 0) + (1 - alpha) * r2[q_id][doc_id]
            )
    return Ranking(results, name=name, sort=sort, copy=False)
