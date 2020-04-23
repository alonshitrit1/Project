from itertools import chain
from typing import List

from torch import nn, Tensor

from models.neural_forcaster import Forecaster


class HyperLinear(Forecaster):

    def __init__(self, hidden_dims: List[int], seq_length: int, horizon: int):
        super(HyperLinear, self).__init__()

        hidden_dims.insert(0, seq_length)

        models = [(nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1]),
                   # nn.BatchNorm1d(num_features=hidden_dims[i+1]),
                   # nn.LeakyReLU(negative_slope=1e-1)
                   nn.Tanh()
                   # nn.Sigmoid()
                   )
                  for i in range(len(hidden_dims) - 1)]

        self.sequential = nn.Sequential(*chain(*models))

        self.coeff = nn.Linear(in_features=hidden_dims[-1], out_features=seq_length * horizon)
        self.bias = nn.Linear(in_features=hidden_dims[-1], out_features=horizon)

    def forward(self, tensor: Tensor):
        batch_size, seq_length, n_features = tensor.shape
        shaped_x = tensor.view(batch_size, -1)
        base = self.sequential(shaped_x)
        coeff = self.coeff(base).view(batch_size, seq_length, -1)
        bias = self.bias(base)
        result = (coeff * tensor).sum(dim=1) + bias
        return result
