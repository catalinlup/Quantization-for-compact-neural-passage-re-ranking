import numpy as np
import h5py
import faiss
from .Quantizer import Quantizer
from utils import sample_from_dataset_vectors
import time


class FaissPQ(Quantizer):
    """
    Wrapper around a faiss quantizer.
    """

    def _is_power_of_2(self, x: int) -> bool:
        if type(x) != int:
            return False
        
        log2_x = int(np.log2(x))

        v = int(2 ** log2_x)


        if v != int(x):
            return False

        return True
    


    def __init__(self, vector_size: int, m: int, k: int) -> None:
        """
        Initializes the FaissPQ
        """

        super().__init__()

        if not self._is_power_of_2(k):
            raise Exception('K needs to be a power of 2')

        n_bits = int(np.log2(k))

        self.pq = faiss.ProductQuantizer(vector_size, m, n_bits)
        self.is_quantizer_trained = False
        self.codebook = None
        self.vector_size = vector_size
        self.m = m
        self.k = k

    


    def train(self, dataset: h5py.Dataset, sample_size: int = None) -> None:
        """
        Trains the quantizer on the provided dataset, by taking a sample of the provided size.
        If no sample size is provided, it will train the quantizer on the entire dataset.
        """
        print('Trainining started', flush=True)
        time_st = time.time()

        sample: np.ndarray = sample_from_dataset_vectors(dataset, sample_size)
        self.pq.train(sample)
        self.is_quantizer_trained = True
        self.codebook = faiss.vector_to_array(self.pq.centroids).reshape(self.m, self.pq.ksub, self.pq.dsub)

        time_end = time.time()

        print(f'Training ended. Ellapsed time (s): {time_end - time_st}', flush=True)


    def quantize(self, dataset: h5py.Dataset, batch_size: int) -> np.ndarray:
        """
        Quantizes the provided dataset, returning an array representing the codes.

        Keyword arguments:
            dataset -- the dataset to be quantized
            batch_size -- the size of the batch to be loaded in memory
        """

        if not self.is_trained():
            raise Exception('You need to first train the quantizer')

        vector_count = int(dataset.shape[0])
        num_batches = vector_count // batch_size + (1 if vector_count % batch_size > 0 else 0)

        code_batches = []

        for bi in range(0, num_batches):
            print(f'{bi + 1} / {num_batches}', flush=True)

            start_ms = time.time()

            index_start = bi * batch_size
            index_end = min((bi + 1) * batch_size, vector_count)

            vector_batch = np.array(dataset[index_start:index_end, :])


            code_batch = self.pq.compute_codes(vector_batch)


            code_batches.append(code_batch)

            end_ms = time.time()

            print(f'Elleapsed time(s) {end_ms - start_ms}', flush=True)

        
        return np.concatenate(code_batches, axis=0)

    def is_trained(self) -> bool:
        """
        Return true if the quantizer is trained, false otherwise.
        """

        return self.is_quantizer_trained
    
    def get_codebook(self) -> np.ndarray:
        """
        Return the code book
        """

        return self.codebook
