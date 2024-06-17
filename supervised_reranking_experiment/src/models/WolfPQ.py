import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


class WolfPQEncoder(nn.Module):

    def __init__(self, dim, M, K, min_dist_factor, semantic_sampling) -> None:
        super(WolfPQEncoder, self).__init__()
        self.dim = dim
        self.M = M
        self.K = K
        self.min_dist_factor = min_dist_factor
        self.norm = nn.LayerNorm(dim)
        self.layer1 = nn.Linear(dim, M * K // 2)
        self.layer2 = nn.Linear(M * K // 2, M * K)
        self.interp_param = nn.Parameter(torch.randn(1))


        self.semantic_sampling = semantic_sampling

        if not self.semantic_sampling:
          self.layer1.requires_grad_(False)
          self.layer2.requires_grad_(False)
          self.interp_param.requires_grad_(False)


    def _compute_distance(self, x, codebook):
      """
      Computes the expected quantizer indices based on the minimum distance
      """

      x_reshaped = x.reshape(x.shape[0], self.M, 1, self.dim // self.M)
      codebook_reshaped = codebook.reshape(1, self.M, self.K, self.dim // self.M)

      dist = torch.sum((codebook_reshaped - x_reshaped) ** 2, dim=-1)

      return dist


    def forward(self, x, codebook):

        if x.shape[-1] != self.dim:
            raise Exception(f'Expected embedding if size {self.dim}')


        dist = self._compute_distance(x, codebook)
        min_dist = F.one_hot(torch.argmin(dist, dim=-1), self.K).to(dtype=torch.float)

        if self.semantic_sampling:

          # propoage vector throught he feed forward neural network
          h = F.tanh(self.layer1(self.norm(x)))
          a = F.relu(self.layer2(h))
          a_reshaped = a.reshape(-1, self.M, self.K)

          # compute the interpolation patameter
          interp = F.sigmoid(self.interp_param)

          # sample from gumbels softmax distribution based on the miniumum distance corrected by the semantic information derived from the vector
          s = F.gumbel_softmax((1.0 - interp ) * a_reshaped + interp * self.min_dist_factor * min_dist, hard=True, dim=-1)

        else:

          # sample base don minimum distance only
          s = F.gumbel_softmax(self.min_dist_factor * min_dist, hard=True, dim=-1)

        return s



class WolfPQ(nn.Module):

    def __init__(self, dim, M, K, min_dist_factor, semantic_sampling: bool, pq_index_path: str) -> None:
        super(WolfPQ, self).__init__()

        self.dim = dim
        self.M = M
        self.K = K
        self.min_dist_factor = min_dist_factor
        self.rotation_matrix = nn.Parameter(torch.stack([torch.eye(dim // M) for _ in range(M)]))

        self.encoder = WolfPQEncoder(dim, M, K, min_dist_factor, semantic_sampling)

        if pq_index_path == None:
            self.codebook = nn.Parameter(torch.randn((M, K, dim // M)))

        else:
            pq_index = pickle.load(open(pq_index_path, 'rb'))
            initial_codebook = pq_index['codebook']
            self.codebook = nn.Parameter(torch.from_numpy(initial_codebook))




    def forward(self, x):

        if x.shape[-1] != self.dim:
            raise Exception(f'Expected embedding if size {self.dim}')

        # apply the rotational matrix to the input
        codebook_rot = self.codebook @ self.rotation_matrix

        # get the indices into the codebook based on the vector
        s = self.encoder(x, codebook_rot)

        # construct the quantized vector to be used during training
        res = codebook_rot.reshape(-1, *codebook_rot.shape) * s.reshape(*s.shape, -1)
        res1 = res.sum(dim=2)
        res2 = res1.reshape(res1.shape[0], -1)

        return res2, s