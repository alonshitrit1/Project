from typing import Dict, Tuple

import argparse
import torch

from torch import nn
from sklearn.metrics import mean_squared_error
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler

from src.models.autoregressive import LinearAutoregressive
from src.models.hyper_lstm import HyperLSTM
from src.models.lstm import VanillaLSTM
from src.models.neural_forcaster import Forecaster
from src.utils.preprocess import get_stock_data, get_electricity_data, get_traffic_data
from src.models.hyper_linear import HyperLinear
from src.utils.visualize import *

use_cuda = torch.cuda.is_available()
lstm_device = torch.device("cuda:0" if use_cuda else "cpu")
linear_device = torch.device('cpu')


class TrainInput(NamedTuple):
    model: Forecaster
    optimizer: Optimizer
    scheduler: _LRScheduler


class TrainOutput(NamedTuple):
    model: Forecaster
    look_ahead_context: np.ndarray
    look_ahead: np.ndarray


def train_models(models: Dict[str, TrainInput],
                 train_data_loader: DataLoader,
                 validation_data_loader: DataLoader,
                 loss_function: _Loss,
                 num_epochs: int,
                 look_ahead_context: Tuple[torch.Tensor, torch.Tensor]):
    result = {}

    for model_name, input_info in models.items():
        model, optimizer, scheduler = input_info
        model.fit(train_data_loader=train_data_loader,
                  validation_data_loader=validation_data_loader,
                  loss_function=loss_function,
                  optimizer=optimizer,
                  lr_scheduler=scheduler,
                  num_epochs=num_epochs)

        result[model_name] = TrainOutput(model=model,
                                         look_ahead_context=look_ahead_context,
                                         look_ahead=model(look_ahead_context[0].unsqueeze(0)))

    return result


def get_unscaled_loss(batch_size, labels, predictions, scaler):
    np_labels = torch.cat(labels.detach().unbind()).cpu().numpy().reshape(batch_size, -1)
    np_predictions = torch.cat(predictions.detach().unbind()).cpu().numpy().reshape(batch_size, -1)

    scaled_labels, scaled_predictions = scaler.inverse_transform(np_labels), scaler.inverse_transform(np_predictions)

    return mean_squared_error(scaled_labels, scaled_predictions)


def main():
    parser = argparse.ArgumentParser(description='Run Time Series Forecasting')

    parser.add_argument('--seq_length', type=int, default=168,
                        help='The Time Series Sequence Length')

    parser.add_argument('--horizon', type=int, default=24,
                        help='How Many DataPoint In The Future To Predict')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='The Training Batch Size')

    parser.add_argument('--epochs', type=int, default=10,
                        help='The Number Of Epochs To Run')

    args = parser.parse_args()

    ######################################### Static #########################################

    seq_length: int = args.seq_length
    batch_size: int = args.batch_size
    num_epochs: int = 2  # args.epochs
    horizon: int = args.horizon

    output_directory = 'stocks'

    ######################################### Organizing Data #########################################

    prerocessed_data = get_traffic_data(scale=False, seq_length=seq_length, test_ratio=0.25, horizon=horizon)

    train_dataset, validation_dataset, scaler = prerocessed_data.train_dataset, prerocessed_data.validation_dataset, prerocessed_data.scaler

    n_samples, n_features = train_dataset.data.shape

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

    look_ahead_context = validation_dataset[len(validation_dataset) - 1]

    # initial_look_ahead_input = torch.from_numpy(look_ahead_context[:seq_length].reshape((1, seq_length, n_features)))

    loss_function = nn.MSELoss(reduction='sum')

    ######################################### Models #########################################

    # Linear
    hidden_dims_auto = [8, 16, 32, 64, 128]
    linear = LinearAutoregressive(hidden_dims=hidden_dims_auto,
                                  seq_length=seq_length, horizon=horizon).double()

    linear.device = linear_device

    linear_optimizer = torch.optim.Adam(linear.parameters(), lr=1e-2)
    linear_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=linear_optimizer, gamma=0.99)

    # Hyper Linear
    hidden_dims = [8, 16, 32, 64, 128]

    hyper_linear = HyperLinear(hidden_dims=hidden_dims,
                               seq_length=seq_length, horizon=horizon).double()

    hyper_linear.device = linear_device

    hyper_linear_optimizer = torch.optim.Adam(hyper_linear.parameters(), lr=1e-4)
    hyper_linear_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=hyper_linear_optimizer, gamma=0.99)

    # LSTM
    lstm = VanillaLSTM(n_features=n_features,
                       seq_length=seq_length,
                       horizon=horizon).double()

    lstm.device = lstm_device
    lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    lstm_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=lstm_optimizer, gamma=0.99)

    # Hyper LSTM
    hyper_lstm = HyperLSTM(n_features=n_features,
                           seq_length=seq_length,
                           horizon=horizon).double()

    hyper_lstm.device = lstm_device

    hyper_lstm_optimizer = torch.optim.Adam(hyper_lstm.parameters(), lr=1e-4)
    hyper_lstm_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=hyper_lstm_optimizer, gamma=0.99)

    ######################################### Train #########################################

    models = {"Linear": TrainInput(linear, linear_optimizer, linear_scheduler),
              "Hyper Linear": TrainInput(hyper_linear, hyper_linear_optimizer, hyper_linear_scheduler),
              "LSTM": TrainInput(lstm, lstm_optimizer, lstm_scheduler),
              "Hyper LSTM": TrainInput(hyper_lstm, hyper_lstm_optimizer, hyper_lstm_scheduler)
              }

    results = train_models(models=models,
                           train_data_loader=train_data_loader,
                           validation_data_loader=validation_data_loader,
                           loss_function=loss_function,
                           num_epochs=num_epochs,
                           look_ahead_context=look_ahead_context
                           )

    ######################################### Plots #########################################

    losses_info = [
        TrainLossInfo(model_name=model_name, train_loss=model.train_loss, validation_loss=model.validation_loss)
        for model_name, (model, look_ahead) in results.items()
    ]

    plot_losses(losses_info=losses_info, output_dir=output_directory)

    forecast_info = [ForecastsInfo(model_name=model_name,
                                   look_ahead_forecasts=look_ahead) for
                     model_name, (model, look_ahead_context, look_ahead) in
                     results.items()]

    plot_forecasts(train=look_ahead_context[0],
                   validation=look_ahead_context[1],
                   forecasts=forecast_info,
                   output_dir=output_directory)


if __name__ == '__main__':
    main()
    exit(0)
