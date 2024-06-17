import h5py
import numpy as np

class Quantizer:
    """
    Interface for a generic quantizer.
    """

    def train(self, dataset: h5py.Dataset, sample_size: int = None) -> None:
        """
        Trains the quantizer on the provided dataset, by taking a sample of the provided size.
        If no sample size is provided, it will train the quantizer on the entire dataset.
        """
        raise Exception('Not implemented')
    
    def quantize(self, dataset: h5py.Dataset, batch_size: int) -> np.ndarray:
        """
        Quantizes the provided dataset, returning an array representing the codes.

        Keyword arguments:
            dataset -- the dataset to be quantized
            batch_size -- the size of the batch to be loaded in memory
        """

        raise Exception('Not implemented')
    

    def is_trained(self) -> bool:
        """
        Return true if the quantizer is trained, false otherwise.
        """

        raise Exception('Not implemented')

    def get_codebook(self) -> np.ndarray:
        """
        Return the code book
        """

        raise Exception('Not implemented')
