import torch
from torch import nn

from src.models.neural_forcaster import Forecaster


class HyperLSTM(Forecaster):
    def __init__(self,
                 n_features: int,
                 seq_length: int,
                 num_layers: int = 1,
                 hidden_layer_size: int = 100,
                 horizon: int = 24
                 ):
        super(HyperLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.seq_length = seq_length
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_layer_size,
                            num_layers=num_layers,
                            batch_first=True
                            )

        self.linear = nn.Linear(hidden_layer_size, horizon)

        self.hidden_cell = None

    def init_hidden(self, n_features: int, batch_size: int, hidden_layer_size: int):
        self.hidden_cell = (
            torch.zeros(self.num_layers * n_features, batch_size, hidden_layer_size).double().to(device=self.device),
            torch.zeros(self.num_layers * n_features, batch_size, hidden_layer_size).double().to(device=self.device)
        )

    def forward(self, input_seq):
        batch_size, seq_length, n_features = input_seq.shape
        self.init_hidden(n_features=n_features,
                         batch_size=batch_size,
                         hidden_layer_size=self.hidden_layer_size)

        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)

        coeff = self.linear(lstm_out.contiguous().view(-1, self.hidden_layer_size)).view(batch_size, seq_length, -1)

        predictions = (input_seq * coeff).sum(dim=1)
        return predictions
