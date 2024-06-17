import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


class WolfPQRotateEncoder(nn.Module):

    def __init__(self, dim, M, K) -> None:
        super(WolfPQRotateEncoder, self).__init__()
        self.dim = dim
        self.M = M
        self.K = K
        self.rotation_matrix = nn.Parameter(torch.eye(dim))
        self.layer1 = nn.Linear(dim, M * K // 2)
        self.layer2 = nn.Linear(M * K // 2, M * K)

    
    def forward(self, x):

        if x.shape[-1] != self.dim:
            raise Exception(f'Expected embedding if size {self.dim}')

        # apply rotation before propagating through the network.
        x_rot = x @ self.rotation_matrix.T
        h = F.tanh(self.layer1(x_rot))
        a = F.relu(self.layer2(h))

        a_reshaped = a.reshape(-1, self.M, self.K)

        s = F.gumbel_softmax(a_reshaped, hard=True, dim=-1)

        return s



class WolfPQRotate(nn.Module):

    def __init__(self, dim, M, K, pq_index_path: str) -> None:
        super(WolfPQRotate, self).__init__()

        self.dim = dim
        self.M = M
        self.K = K

        self.encoder = WolfPQRotateEncoder(dim, M, K)

        if pq_index_path == None:
            self.codebook = nn.Parameter(torch.randn((M, K, dim // M)))

        else:
            pq_index = pickle.load(open(pq_index_path, 'rb'))
            initial_codebook = pq_index['codebook']
            self.codebook = nn.Parameter(torch.from_numpy(initial_codebook))
  
    
    def forward(self, x):

        if x.shape[-1] != self.dim:
            raise Exception(f'Expected embedding if size {self.dim}')
        
        s = self.encoder(x)

        # select from the codebook using the gumbel indices
        res = self.codebook.reshape(-1, *self.codebook.shape) * s.reshape(*s.shape, -1)
        res1 = res.sum(dim=2)
        res2 = res1.reshape(res1.shape[0], -1)

        return res2, s