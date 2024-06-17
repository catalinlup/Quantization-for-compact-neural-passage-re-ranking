import sys
sys.path.append('../')
from src.models.WolfPQ import WolfPQEncoder
import unittest
import torch
import numpy as np
import random

class WolfPQTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    
    def test_wolf_pq_encoder_batch(self):
        encoder = WolfPQEncoder(768, 16, 256)
        input = torch.randn((5, 768))
        output = encoder(input)


        self.assertEqual(output.shape[0], 5)
        self.assertEqual(output.shape[1], 16)
        self.assertEqual(output.shape[2], 256)

        for i in range(5):
            for j in range(16):
                self.assertAlmostEqual(output[i][j].sum().item(), 1.0)

    
    def test_wolf_pq_encoder_single(self):
        encoder = WolfPQEncoder(768, 16, 256)
        input = torch.randn(768)
        output = encoder(input)


        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 16)
        self.assertEqual(output.shape[2], 256)

        for i in range(1):
            for j in range(16):
                self.assertAlmostEqual(output[i][j].sum().item(), 1.0)

    
    def test_wolf_pq_encoder_indexing(self):
        encoder = WolfPQEncoder(768, 16, 256)
        input = torch.randn(3, 768)
        output = encoder(input)
        codebook = torch.randn(16, 256, 768 // 16)

        print(codebook.shape)
        print(output[0].shape)

        res = codebook * output[0].reshape(16, 256, 1)

        print(res)
        print(res.shape)

        res1 = res.sum(dim=1)
        print(res1)
        print(res1.shape)

        res2 = res1.reshape(1, -1)

        print(res2)
        print(res2.shape)
        
        # codebook * output[0]

    def test_wolf_pq_encoder_indexing_multi(self):
        encoder = WolfPQEncoder(768, 16, 256)
        input = torch.randn(3, 768)
        output = encoder(input)
        codebook = torch.randn(16, 256, 768 // 16)


        res = codebook.reshape(-1, *codebook.shape) * output.reshape(*output.shape, -1)
        res1 = res.sum(dim=2)
        res2 = res1.reshape(res1.shape[0], -1)

        for bi in range(3):
            indices = torch.argmax(output[bi], dim=-1)

            sub_vecs = []

            for mi in range(16):
                sub_vecs.append(codebook[mi, indices[mi]])

           
            vec = torch.concat(sub_vecs)
            diff = torch.sum((vec - res2[bi]) ** 2).item()
            self.assertAlmostEqual(diff, 0)

            print(vec)
            print(res2[bi])

            print('##')