import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


class WolfPQMinDistEncoder(nn.Module):

    def __init__(self, dim, M, K, min_dist_factor, semantic_training) -> None:
        super(WolfPQMinDistEncoder, self).__init__()
        self.dim = dim
        self.M = M
        self.K = K
        self.min_dist_factor = min_dist_factor
        self.norm = nn.LayerNorm(dim)
        self.layer1 = nn.Linear(dim, M * K // 2)
        self.layer2 = nn.Linear(M * K // 2, M * K)
        self.interp_param = nn.Parameter(torch.randn(1))


        self.semantic_training = semantic_training

        if not self.semantic_training:
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
        # print(torch.sum(dist_filtered, dim=-1))


        # min_dist = min_dist * (min_dist > max_vals)
        # print(torch.max(min_dist, keepdim=True, dim=-1).values)

        # only apply interpolation when training for the reranking experiment
        if self.semantic_training:
          h = F.tanh(self.layer1(self.norm(x)))
          # apply rotation before propagating through the network.
          a = F.relu(self.layer2(h))
          a_reshaped = a.reshape(-1, self.M, self.K)

          # compute the interpolation paramterer
          # a_dist_combination = torch.concat([a, dist_linear], dim=-1)
          interp = F.sigmoid(self.interp_param)
          # interp = interp.reshape(interp.shape[0], 1, 1)

          s = F.gumbel_softmax((1.0 - interp ) * a_reshaped + interp * self.min_dist_factor * min_dist, hard=True, dim=-1)

        else:
          # when performing MSE training, always take the minium distance
          # 

          s = F.gumbel_softmax(self.min_dist_factor * min_dist, hard=True, dim=-1)

        return s



class WolfPQMinDist(nn.Module):

    def __init__(self, dim, M, K, min_dist_factor, semantic_training: bool, pq_index_path: str) -> None:
        super(WolfPQMinDist, self).__init__()

        self.dim = dim
        self.M = M
        self.K = K
        self.min_dist_factor = min_dist_factor
        self.rotation_matrix = nn.Parameter(torch.stack([torch.eye(dim // M) for _ in range(M)]))

        self.encoder = WolfPQMinDistEncoder(dim, M, K, min_dist_factor, semantic_training)

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

        s = self.encoder(x, codebook_rot)
        # expected_min_dist = self.compute_expected_min_distance(x_rot, self.codebook)

        # rotated codebook
        # codebook_rot =
        # select from the codebook using the gumbel indices
        res = codebook_rot.reshape(-1, *codebook_rot.shape) * s.reshape(*s.shape, -1)
        res1 = res.sum(dim=2)
        res2 = res1.reshape(res1.shape[0], -1)

        return res2, s