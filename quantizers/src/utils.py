from typing import Union
import h5py
import numpy as np
from tqdm import trange
import math

def sample_from_dataset_vectors(dataset: h5py.Dataset, sample_size) -> np.ndarray:
    """
    Takes a random sample from the provided h5 dataset. If the sample size is none, raturns the entire dataset.
    """

    SAMPLE_BATCH_SIZE = 1000

    if sample_size == None:
        return dataset[:, :]
    

    random_ids = np.random.choice(dataset.shape[0], size=sample_size, replace=False)
    random_ids.sort()

    num_batches = math.ceil(random_ids.shape[0] / SAMPLE_BATCH_SIZE)

    random_samples = []

    for bi in trange(num_batches):

        index_start = bi * SAMPLE_BATCH_SIZE
        index_end = min((bi + 1) * SAMPLE_BATCH_SIZE, random_ids.shape[0])

        random_id_batch = random_ids[index_start:index_end]
        random_samples.append(dataset[random_id_batch])

    return np.concatenate(random_samples)