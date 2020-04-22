from typing import Tuple

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
from src.utils.common import get_logger
from src.utils.preprocess import load_dataset
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
    avg_test_loss: float
    look_ahead: np.ndarray


def train_models(models: Dict[str, TrainInput],
                 train_data_loader: DataLoader,
                 validation_data_loader: DataLoader,
                 test_data_loader: DataLoader,
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

        avg_test_loss = model.predict(test_data_loader=test_data_loader, loss_function=loss_function)
        look_ahead = model(look_ahead_context[0].unsqueeze(0).to(model.device)).detach().numpy().squeeze()

        result[model_name] = TrainOutput(model=model,
                                         avg_test_loss=avg_test_loss,
                                         look_ahead=look_ahead)

    return result


def run_experiment(n_features: int,
                   seq_length: int,
                   horizon: int,
                   train_data_loader: DataLoader,
                   validation_data_loader: DataLoader,
                   test_data_loader: DataLoader,
                   loss_function: _Loss,
                   num_epochs: int,
                   look_ahead_context: Tuple[torch.Tensor, torch.Tensor]):

    learning_rate = 1e-4
    # Linear
    hidden_dims_auto = [8, 16, 32, 64, 128]
    linear = LinearAutoregressive(hidden_dims=hidden_dims_auto,
                                  seq_length=seq_length, horizon=horizon).double()

    linear.device = linear_device

    linear_optimizer = torch.optim.Adam(linear.parameters(), lr=learning_rate)
    linear_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=linear_optimizer, gamma=0.99)

    # Hyper Linear
    hidden_dims = [8, 16, 32, 64, 128]

    hyper_linear = HyperLinear(hidden_dims=hidden_dims,
                               seq_length=seq_length, horizon=horizon).double()

    hyper_linear.device = linear_device

    hyper_linear_optimizer = torch.optim.Adam(hyper_linear.parameters(), lr=learning_rate)
    hyper_linear_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=hyper_linear_optimizer, gamma=0.99)

    # LSTM
    lstm = VanillaLSTM(n_features=n_features,
                       seq_length=seq_length,
                       horizon=horizon).double()

    lstm.device = lstm_device
    lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    lstm_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=lstm_optimizer, gamma=0.99)

    # Hyper LSTM
    hyper_lstm = HyperLSTM(n_features=n_features,
                           seq_length=seq_length,
                           horizon=horizon).double()

    hyper_lstm.device = lstm_device

    hyper_lstm_optimizer = torch.optim.Adam(hyper_lstm.parameters(), lr=learning_rate)
    hyper_lstm_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=hyper_lstm_optimizer, gamma=0.99)

    ######################################### Train #########################################

    models = {"Linear": TrainInput(linear, linear_optimizer, linear_scheduler),
              "Hyper Linear": TrainInput(hyper_linear, hyper_linear_optimizer, hyper_linear_scheduler),
              "LSTM": TrainInput(lstm, lstm_optimizer, lstm_scheduler),
              "Hyper LSTM": TrainInput(hyper_lstm, hyper_lstm_optimizer, hyper_lstm_scheduler)
              }

    experiment_result = train_models(models=models,
                                     train_data_loader=train_data_loader,
                                     validation_data_loader=validation_data_loader,
                                     test_data_loader=test_data_loader,
                                     loss_function=loss_function,
                                     num_epochs=num_epochs,
                                     look_ahead_context=look_ahead_context
                                     )

    return experiment_result


def get_unscaled_loss(batch_size, labels, predictions, scaler):
    np_labels = torch.cat(labels.detach().unbind()).cpu().numpy().reshape(batch_size, -1)
    np_predictions = torch.cat(predictions.detach().unbind()).cpu().numpy().reshape(batch_size, -1)

    scaled_labels, scaled_predictions = scaler.inverse_transform(np_labels), scaler.inverse_transform(np_predictions)

    return mean_squared_error(scaled_labels, scaled_predictions)

def main():
    parser = argparse.ArgumentParser(description='Run Time Series Forecasting')

    parser.add_argument('--num_experiments', type=int, default=10,
                        help='How many experiments to run')

    parser.add_argument('--seq_length', type=int, default=168,
                        help='The Time Series Sequence Length')

    parser.add_argument('--horizon', type=int, default=24,
                        help='How Many DataPoint In The Future To Predict')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='The Training Batch Size')

    parser.add_argument('--epochs', type=int, default=50,
                        help='The Number Of Epochs To Run')

    parser.add_argument('--dataset', type=str, choices=['stocks', 'traffic'], default='stocks',
                        help='Which Dataset To Load')

    args = parser.parse_args()


    ######################################### Static #########################################
    logger = get_logger('Main')
    num_experiments: int = args.num_experiments
    seq_length: int = args.seq_length
    batch_size: int = args.batch_size
    num_epochs: int = args.epochs
    horizon: int = args.horizon
    dataset: str = args.dataset

    ######################################### Organizing Data #########################################

    # loading data
    train_dataset, validation_dataset, test_dataset, scaler = load_dataset(dataset=dataset,
                                                                           seq_length=seq_length,
                                                                           horizon=horizon)

    n_samples, n_features = train_dataset.data.shape

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    look_ahead_context = test_dataset[len(validation_dataset) - 1]

    loss_function = nn.MSELoss(reduction='sum')

    ######################################### Running Experiments #########################################

    results: Dict[str, List[TrainOutput]] = {}

    for i in range(num_experiments):
        logger.info(f'Running Experiment [{i + 1}/{num_experiments}]')
        experiment_result = run_experiment(n_features=n_features,
                                           seq_length=seq_length,
                                           horizon=horizon,
                                           train_data_loader=train_data_loader,
                                           validation_data_loader=validation_data_loader,
                                           test_data_loader=test_data_loader,
                                           loss_function=loss_function,
                                           num_epochs=num_epochs,
                                           look_ahead_context=look_ahead_context)

        for key, value in experiment_result.items():
            if key not in results:
                results[key] = []
            results[key].append(value)

    best_models = {key: min(results, key=lambda result: result.avg_test_loss) for key, results in results.items()}

    ######################################### Results #########################################

    losses_info = [
        TrainLossInfo(model_name=model_name, train_loss=model.train_loss, validation_loss=model.validation_loss)
        for model_name, (model, avg_loss, look_ahead) in best_models.items()
    ]

    plot_losses(losses_info=losses_info, output_dir=dataset)

    forecast_info = [ForecastsInfo(model_name=model_name,
                                   look_ahead_forecasts=look_ahead) for
                     model_name, (model, avg_test_loss, look_ahead) in
                     best_models.items()]

    plot_forecasts(input_seq=look_ahead_context[0],
                   test=look_ahead_context[1],
                   forecasts=forecast_info,
                   output_dir=dataset)

    save_results(results=results,
                 output_directory=dataset)


if __name__ == '__main__':
    main()
    exit(0)
