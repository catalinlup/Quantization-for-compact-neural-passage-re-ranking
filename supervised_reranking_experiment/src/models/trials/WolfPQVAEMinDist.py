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





class WolfPQVAEMinDistEncoder(nn.Module):

    def __init__(self, dim, M, K, min_dist_factor) -> None:
        super(WolfPQVAEMinDistEncoder, self).__init__()
        self.dim = dim
        self.M = M
        self.K = K
        self.min_dist_factor = min_dist_factor
        self.norm = nn.LayerNorm(dim)
        self.rotation_matrix1 = nn.Parameter(torch.eye(dim))
        self.rotation_matrix2 = nn.Parameter(torch.eye(M * K // 2))
        self.layer1 = VariationalEncoder(dim, int(0.75 * M * K), M * K // 2)
        self.layer2 = nn.Linear(M * K // 2, M * K)
        # self.layer_interpolation = nn.Linear(M * K * 2, 1)
        self.interp_param = nn.Parameter(torch.randn(1))


    def _compute_distance(self, x, codebook):
      x_reshaped = x.reshape(x.shape[0], self.M, 1, self.dim // self.M)
      codebook_reshaped = codebook.reshape(1, self.M, self.K, self.dim // self.M)

      dist = torch.sum((codebook_reshaped - x_reshaped) ** 2, dim=-1)


      return dist


    def forward(self, x, codebook):

        if x.shape[-1] != self.dim:
            raise Exception(f'Expected embedding if size {self.dim}')

        x_rot = x @ self.rotation_matrix1.T

        h = F.tanh(self.layer1(self.norm(x_rot)))

        # apply rotation before propagating through the network.
        h_rot = h @ self.rotation_matrix2.T
        a = F.relu(self.layer2(h_rot))

        a_reshaped = a.reshape(-1, self.M, self.K)

        # compute the soft distmin between the vectors and the codebook
        dist = self._compute_distance(x_rot, codebook)
        min_dist = F.one_hot(torch.argmin(dist, dim=-1), self.K).to(dtype=torch.float)


        # compute the interpolation paramterer
        # a_dist_combination = torch.concat([a, dist_linear], dim=-1)
        interp = F.sigmoid(self.interp_param)
        # interp = interp.reshape(interp.shape[0], 1, 1)

        s = F.gumbel_softmax((1.0 - interp ) * a_reshaped + interp * self.min_dist_factor * min_dist, hard=True, dim=-1)

        return s



class WolfPQVAEMinDist(nn.Module):

    def __init__(self, dim, M, K, min_dist_factor, pq_index_path: str) -> None:
        super(WolfPQVAEMinDist, self).__init__()

        self.dim = dim
        self.M = M
        self.K = K
        self.min_dist_factor = min_dist_factor

        self.encoder = WolfPQVAEMinDistEncoder(dim, M, K, min_dist_factor)

        if pq_index_path == None:
            self.codebook = nn.Parameter(torch.randn((M, K, dim // M)))

        else:
            pq_index = pickle.load(open(pq_index_path, 'rb'))
            initial_codebook = pq_index['codebook']
            self.codebook = nn.Parameter(torch.from_numpy(initial_codebook))


    def forward(self, x):

        if x.shape[-1] != self.dim:
            raise Exception(f'Expected embedding if size {self.dim}')

        s = self.encoder(x, self.codebook)

        # select from the codebook using the gumbel indices
        res = self.codebook.reshape(-1, *self.codebook.shape) * s.reshape(*s.shape, -1)
        res1 = res.sum(dim=2)
        res2 = res1.reshape(res1.shape[0], -1)

        return res2, s