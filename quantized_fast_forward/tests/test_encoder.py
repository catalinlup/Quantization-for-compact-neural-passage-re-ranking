import unittest

import numpy as np

from fast_forward.encoder import TCTColBERTQueryEncoder, LambdaQueryEncoder


class TestTCTColBERTQueryEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.encoder = TCTColBERTQueryEncoder(
            "castorini/tct_colbert-msmarco", device="cpu"
        )

    def test_encode(self):
        q_enc = self.encoder.encode(["test query 1", "test query 2"])
        self.assertEqual(q_enc.shape, (2, 768))


class TestLambdaQueryEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.encoder = LambdaQueryEncoder(lambda q: np.zeros(shape=(768,)))

    def test_encode(self):
        q_enc = self.encoder.encode(["test query 1", "test query 2"])
        self.assertTrue(np.array_equal(q_enc, np.zeros(shape=(2, 768))))


if __name__ == "__main__":
    unittest.main()
