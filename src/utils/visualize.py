import os
from typing import NamedTuple, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cycler

color_cycler = cycler(color=[list(rgb) for rgb in plt.get_cmap('Set1').colors])
plt.rc('axes', prop_cycle=color_cycler)

PLOTS_PATH = os.path.join('data', 'plots')


class TrainLossInfo(NamedTuple):
    model_name: str
    train_loss: np.array
    validation_loss: np.array


class ForecastsInfo(NamedTuple):
    model_name: str
    look_ahead_forecasts: np.array


def plot_losses(losses_info: List[TrainLossInfo], output_dir: str):
    fig, axes = plt.subplots(nrows=len(losses_info), figsize=(45, 45))

    for i, loss_info in enumerate(losses_info):
        epochs = np.arange(1, 1 + len(loss_info.train_loss))
        axes[i].plot(epochs, loss_info.train_loss,
                     linewidth=2, label=f'{loss_info.model_name} - Train Loss')
        axes[i].plot(epochs, loss_info.validation_loss,
                     linewidth=2, label=f'{loss_info.model_name} - Validation Loss')

        axes[i].set_title(loss_info.model_name)
        axes[i].set_ylabel(ylabel='Loss')
        axes[i].set_xlabel(xlabel='Epoch')
        axes[i].set_xticks(epochs)
        axes[i].legend()
        axes[i].grid()
        axes[i].margins(x=0, y=0)

    plt.savefig(os.path.join(PLOTS_PATH, output_dir, "losses.png"))


def plot_forecasts(train: np.array,
                   validation: np.array,
                   forecasts: List[ForecastsInfo],
                   output_dir: str):

    train_size, validation_size = len(train), len(validation)
    train_indices, validation_indices = np.arange(train_size), np.arange(train_size, train_size + validation_size)

    fig, axes = plt.subplots(nrows=len(forecasts), figsize=(45, 45))

    for i, forecast_info in enumerate(forecasts):
        axes[i].plot(train_indices, train, linewidth=2, label='Train')
        axes[i].plot(validation_indices, validation, linewidth=2, label='Validation')
        axes[i].axvspan(0, train_size, color='gray', alpha=0.2, fill=True)
        axes[i].axvspan(train_size, train_size + validation_size, color='gray', alpha=0.5, fill=True)

        look_ahead_indices = np.arange(train_size, train_size + len(forecast_info.look_ahead_forecasts))
        axes[i].plot(look_ahead_indices, forecast_info.look_ahead_forecasts, linewidth=2,
                     label=f'{forecast_info.model_name} - Look_Ahead Forecasts')

        axes[i].set_title(forecast_info.model_name)
        axes[i].set_ylabel('Value')
        axes[i].set_xlabel('Time')
        axes[i].legend()
        axes[i].grid()
        axes[i].margins(x=0, y=0)

    plt.savefig(os.path.join(PLOTS_PATH, output_dir, "forecasts.png"))
