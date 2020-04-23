import logging
from typing import List, Callable

from abc import ABC

import numpy as np
import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from utils.common import get_logger


class Forecaster(Module, ABC):

    def __init__(self):
        super(Forecaster, self).__init__()
        self._trained = False
        self._device = None
        self._train_loss = None
        self._validation_loss = None
        self._validation_forecast = None
        self._logger = get_logger(self.__class__.__name__)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def trained(self) -> bool:
        return self._trained

    @trained.setter
    def trained(self, value: bool):
        self._trained = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    @property
    def train_loss(self) -> np.array:
        return self._train_loss

    @train_loss.setter
    def train_loss(self, value: np.array):
        self._train_loss = value

    @property
    def validation_loss(self) -> np.array:
        return self._validation_loss

    @validation_loss.setter
    def validation_loss(self, value: np.array):
        self._validation_loss = value

    # @property
    # def validation_forecast(self) -> np.array:
    #     return self._validation_forecast
    #
    # @validation_forecast.setter
    # def validation_forecast(self, value: np.array):
    #     self._validation_forecast = value

    def fit(self,
            train_data_loader: DataLoader,
            validation_data_loader: DataLoader,
            loss_function: _Loss,
            optimizer: Optimizer,
            lr_scheduler: _LRScheduler,
            num_epochs: int,
            hooks: List[Callable[[], None]] = None):
        """
        Runs experiment with the given parameters and return trained module, train_loss, validation_loss, last_predictions
        :param train_data_loader: Train data.
        :param validation_data_loader: Validation data.
        :param loss_function: Loss functions which will be used along side the optimizer.
        :param optimizer: Optimizer.
        :param lr_scheduler: Learning rate scheduler for the optimizer.
        :param num_epochs: How many epochs to train the model.
        :param hooks: functions to invoke during the training forward pass.
        :return: Trained Forecaster Forecaster
        """
        self.to(device=self.device)
        loss_function = loss_function.to(device=self.device)

        self.train_loss = np.empty(num_epochs)
        self.validation_loss = np.empty(num_epochs)
        # self.validation_forecast = np.empty(len(validation_data_loader.dataset))

        states_dict = None

        avg_validate_loss = float("inf")

        for epoch in range(num_epochs):
            # training
            self.train()
            epoch_train_loss = 0
            for seq, labels in train_data_loader:
                if hooks is not None:
                    for hook in hooks:
                        hook()
                optimizer.zero_grad()

                seq, labels = seq.to(self.device), labels.to(self.device)
                y_pred = self(seq)

                single_loss = loss_function(y_pred.squeeze(), labels.squeeze())
                epoch_train_loss += single_loss.item()
                single_loss.backward()
                optimizer.step()

            # validation
            self.eval()
            epoch_validate_loss = 0
            with torch.no_grad():
                for i, (seq, labels) in enumerate(validation_data_loader):
                    seq, labels = seq.to(self.device), labels.to(self.device)
                    y_pred = self(seq)

                    single_loss = loss_function(y_pred.squeeze(), labels.squeeze())
                    epoch_validate_loss += single_loss.item()

                    # if epoch == num_epochs - 1:
                    #     for j in range(len(seq)):
                    #         self.validation_forecast[i*len(seq) + j] = y_pred[j].item()

            avg_train_loss = epoch_train_loss / len(train_data_loader.dataset)

            new_avg_validate_loss = epoch_validate_loss / len(validation_data_loader.dataset)
            if new_avg_validate_loss < avg_validate_loss:
                avg_validate_loss = new_avg_validate_loss
                states_dict = self.state_dict()

            self.train_loss[epoch] = avg_train_loss
            self.validation_loss[epoch] = new_avg_validate_loss

            self.logger.info(f'Epoch {epoch + 1}/{num_epochs}')
            self.logger.info(f'Avg train loss: {avg_train_loss :10.8f}')
            self.logger.info(f'Avg validation loss: {avg_validate_loss :10.8f}')

            lr_scheduler.step(epoch=epoch)

        self.trained = True
        self.load_state_dict(states_dict)
        return self

    def predict(self,
                test_data_loader: DataLoader,
                loss_function: _Loss
                ) -> float:

        self.to(device=self.device)
        loss_function = loss_function.to(device=self.device)
        self.eval()

        test_loss = 0
        with torch.no_grad():
            for seq, labels in test_data_loader:
                seq, labels = seq.to(self.device), labels.to(self.device)
                y_pred = self(seq)

                single_loss = loss_function(y_pred.squeeze(), labels.squeeze())
                test_loss += single_loss.item()

        avg_test_loss = test_loss / len(test_data_loader.dataset)
        self.device = torch.device('cpu')
        self.cpu()
        return avg_test_loss

    def predict_ahead(self, initial_input: torch.Tensor, horizon: int) -> np.array:
        self.to(self.device)
        current_input = initial_input.to(device=self.device)
        forecasts = np.empty(horizon)
        for i in range(horizon):
            forecast = self(current_input)
            item = forecast.item()
            forecasts[i] = item
            current_input = current_input.roll(-1)
            current_input[0][-1] = item

        return forecasts

    def predict_rolling(self, data_loader: DataLoader) -> np.array:
        self.to(self.device)
        forecasts = np.array(len(data_loader))
        for i, (seq, labels) in enumerate(data_loader):
            seq, labels = seq.to(device=self.device), labels.to(device=self.device)
            forecast = self(seq)
            item = forecast.item()
            forecasts[i] = item

        return forecasts
