import os
from typing import NamedTuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cycler

from src.main import TrainOutput

color_cycler = cycler(color=[list(rgb) for rgb in plt.get_cmap('Set1').colors])
plt.rc('axes', prop_cycle=color_cycler)

PLOTS_PATH = os.path.join('data', 'plots')
RESULTS_PATH = os.path.join('data', 'results')


class TrainLossInfo(NamedTuple):
    model_name: str
    train_loss: np.array
    validation_loss: np.array


class ForecastsInfo(NamedTuple):
    model_name: str
    look_ahead_forecasts: np.array


def plot_losses(losses_info: List[TrainLossInfo], output_dir: str):
    num_models = len(losses_info) // 2
    fig, axes = plt.subplots(nrows=num_models, ncols=2, figsize=(45, 45))

    fig.suptitle('Loss')

    for i, loss_info in enumerate(losses_info):
        epochs = np.arange(1, 1 + len(loss_info.train_loss))
        axes[i // num_models][i % 2].plot(epochs, loss_info.train_loss,
                                          linewidth=2, label=f'{loss_info.model_name} - Train Loss')
        axes[i // num_models][i % 2].plot(epochs, loss_info.validation_loss,
                                          linewidth=2, label=f'{loss_info.model_name} - Validation Loss')

        axes[i // num_models][i % 2].set_title(
            f'{loss_info.model_name}:\nLast Epoch Loss = {loss_info.validation_loss[-1]}')
        axes[i // num_models][i % 2].set_ylabel(ylabel='Loss')
        axes[i // num_models][i % 2].set_xlabel(xlabel='Epoch')
        axes[i // num_models][i % 2].set_xticks(epochs)
        axes[i // num_models][i % 2].legend(loc='upper left')
        axes[i // num_models][i % 2].grid()
        axes[i // num_models][i % 2].margins(x=0, y=0)

    plt.savefig(os.path.join(PLOTS_PATH, output_dir, "losses.png"))


def plot_forecasts(input_seq: np.array,
                   test: np.array,
                   forecasts: List[ForecastsInfo],
                   output_dir: str):
    seq_length, test_size = len(input_seq), len(test)
    input_indices, test_indices = np.arange(seq_length), np.arange(seq_length, seq_length + test_size)

    num_models = len(forecasts) // 2
    fig, axes = plt.subplots(nrows=num_models, ncols=2, figsize=(45, 45))

    fig.suptitle('Forecasts')

    for i, forecast_info in enumerate(forecasts):
        axes[i // num_models][i % 2].plot(input_indices, input_seq, linewidth=2, label='Train')
        axes[i // num_models][i % 2].plot(test_indices, test, linewidth=2, label='Validation')
        axes[i // num_models][i % 2].axvspan(0, seq_length, color='gray', alpha=0.2, fill=True)
        axes[i // num_models][i % 2].axvspan(seq_length, seq_length + test_size, color='gray', alpha=0.5,
                                             fill=True)

        look_ahead_indices = np.arange(seq_length, seq_length + len(forecast_info.look_ahead_forecasts))
        axes[i // num_models][i % 2].plot(look_ahead_indices, forecast_info.look_ahead_forecasts, linewidth=2,
                                          label=f'{forecast_info.model_name} - Look_Ahead Forecasts')

        axes[i // num_models][i % 2].set_title(forecast_info.model_name)
        axes[i // num_models][i % 2].set_ylabel('Value')
        axes[i // num_models][i % 2].set_xlabel('Time')
        axes[i // num_models][i % 2].legend(loc='upper left')
        axes[i // num_models][i % 2].grid()
        axes[i // num_models][i % 2].margins(x=0, y=0)

    plt.savefig(os.path.join(PLOTS_PATH, output_dir, "forecasts.png"))


def save_results(results: Dict[str, List[TrainOutput]], output_directory: str):
    df = pd.DataFrame.from_dict({key: [value.avg_test_loss for value in values] for key, values in results.items()})
    avg, std = df.mean().T, df.std().T
    pd.concat([avg.rename("Avg"), std.rename("Std")], axis=1).to_csv(os.path.join(RESULTS_PATH, output_directory, 'results.csv'))
