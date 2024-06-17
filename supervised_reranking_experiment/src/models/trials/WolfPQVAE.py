import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim: int, mid_dim: int, output_dim: int):
        super(VariationalEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(input_dim, mid_dim)
        self.linear2 = nn.Linear(mid_dim, output_dim)
        self.linear3 = nn.Linear(mid_dim, output_dim)

        self.N = torch.distributions.Normal(0, 1)

        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))

        # sample from
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return z



    

class WolfPQVAEEncoder(nn.Module):

    def __init__(self, dim, M, K) -> None:
        super(WolfPQVAEEncoder, self).__init__()
        self.dim = dim
        self.M = M
        self.K = K
        self.norm = nn.LayerNorm(dim)
        self.rotation_matrix = nn.Parameter(torch.eye(M * K // 2))
        self.layer1 = VariationalEncoder(dim, int(0.75 * M * K), M * K // 2)
        self.layer2 = nn.Linear(M * K // 2, M * K)

    
    def forward(self, x):

        if x.shape[-1] != self.dim:
            raise Exception(f'Expected embedding if size {self.dim}')

        # apply rotation before propagating through the network.
        
        h = F.tanh(self.layer1(self.norm(x)))

        h_rot = h @ self.rotation_matrix.T
        a = F.relu(self.layer2(h_rot))

        a_reshaped = a.reshape(-1, self.M, self.K)

        s = F.gumbel_softmax(a_reshaped, hard=True, dim=-1)

        return s



class WolfPQVAE(nn.Module):

    def __init__(self, dim, M, K, pq_index_path: str) -> None:
        super(WolfPQVAE, self).__init__()

        self.dim = dim
        self.M = M
        self.K = K

        self.encoder = WolfPQVAEEncoder(dim, M, K)

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