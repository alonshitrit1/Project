from itertools import chain
from typing import List

from torch import Tensor
from torch.nn import  Linear, Sequential, ReLU

from models.neural_forcaster import Forecaster


class LinearAutoregressive(Forecaster):

    def __init__(self, hidden_dims: List[int], seq_length: int, horizon: int):
        super(LinearAutoregressive, self).__init__()

        hidden_dims.insert(0, seq_length)
        models = [(Linear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1]),
                   ReLU()
                   )
                  for i in range(len(hidden_dims) - 1)]

        self.sequential = Sequential(*chain(*models),
                                     Linear(in_features=hidden_dims[-1], out_features=horizon)
                                     )

    def forward(self, tensor: Tensor):
        batch_size, seq_length, n_features = tensor.shape
        shaped_x = tensor.view(batch_size, -1)
        result = self.sequential(shaped_x)
        return result