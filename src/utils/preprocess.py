import os
from collections import namedtuple
from typing import Tuple

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset

DATA_PATH = os.path.join('data')


class TimeSeriesDataSet(Dataset):
    """
    A utility class used to hold time series data.
    Each column represents a time series.
    """

    def __init__(self, data: np.array, seq_length: int, horizon: int):
        """
        :param data: Time series data, S.T each column represents a univariate time series.
        :param seq_length: Determines the output size (window size X Number of time series)
        :param horizon: Determines how many data points in the future to return as labels.
        """
        self.data = data
        self.window_size = seq_length
        self.horizon = horizon
        self.len = len(data) - seq_length - horizon

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[index:index + self.window_size + self.horizon, :]

        X, y = data[:-self.horizon, :], data[-self.horizon:, :]
        return torch.from_numpy(X), torch.from_numpy(y)


PreProcessedData = namedtuple('PreProcessedData', ['train_dataset', 'validation_dataset', 'test_dataset', 'scaler'])


def get_pre_processed_data(features: np.array,
                           data_path: str,
                           test_ratio: float,
                           validation_ratio: float,
                           seq_length: int,
                           scale: bool,
                           horizon: int) -> PreProcessedData:
    """
    Utility function which loads data and returns train and test data.
    :param features: which stock indices to look at
    :param data_path: data file path.
    :param test_ratio: how much of the data should be used for testing.
    :param validation_ratio: how much of the data should be used for validation.
    :param seq_length: seq length.
    :param scale: whether or not to scale
    :param horizon: how many data points ahead to predict
    :return: train data and test data
    """
    df = pd.read_csv(data_path)[features]
    data = df.to_numpy()

    scaler = MinMaxScaler(feature_range=(0, 1)) if scale else None

    df_length = len(df)
    validation_size, test_size = int(df_length * validation_ratio), int(df_length * test_ratio)
    train_size = df_length - validation_size - test_size

    train, validation, test = np.vsplit(data, [train_size, train_size + validation_size])

    validation = np.vstack([train[-seq_length:], validation])
    test = np.vstack([validation[-seq_length:], test])

    if scaler is not None:
        train = scaler.fit_transform(train)
        validation = scaler.transform(validation)
        test = scaler.transform(test)

    train_dataset = TimeSeriesDataSet(data=train, seq_length=seq_length, horizon=horizon)
    validation_dataset = TimeSeriesDataSet(data=validation, seq_length=seq_length, horizon=horizon)
    test_dataset = TimeSeriesDataSet(data=test, seq_length=seq_length, horizon=horizon)

    return PreProcessedData(train_dataset=train_dataset,
                            validation_dataset=validation_dataset,
                            test_dataset=test_dataset,
                            scaler=scaler)


def get_stock_data(features: np.array,
                   test_ratio: float = 0.2,
                   validation_ratio: float = 0.2,
                   seq_length: int = 10,
                   scale: bool = True,
                   horizon: int = 24) -> PreProcessedData:
    return get_pre_processed_data(features=features,
                                  data_path=os.path.join(DATA_PATH, 'nyse_prices.csv'),
                                  test_ratio=test_ratio,
                                  validation_ratio=validation_ratio,
                                  seq_length=seq_length,
                                  scale=scale,
                                  horizon=horizon)


def get_traffic_data(test_ratio: float = 0.2,
                     validation_ratio: float = 0.2,
                     seq_length: int = 168,
                     scale: bool = False,
                     horizon: int = 24) -> PreProcessedData:
    return get_pre_processed_data(features=np.asarray(['Occupancy']),
                                  data_path=os.path.join(DATA_PATH, 'traffic.csv'),
                                  test_ratio=test_ratio,
                                  validation_ratio=validation_ratio,
                                  seq_length=seq_length,
                                  scale=scale,
                                  horizon=horizon)


def load_dataset(dataset: str,
                 test_ratio: float = 0.2,
                 validation_ratio: float = 0.2,
                 seq_length: int = 168,
                 horizon: int = 24,
                 **kwargs):

    if dataset == 'traffic':
        return get_traffic_data(test_ratio=test_ratio,
                                validation_ratio=validation_ratio,
                                seq_length=seq_length,
                                horizon=horizon,
                                **kwargs)
    elif dataset == 'stocks':
        return get_stock_data(features=np.asarray(['MSFT']),
                              seq_length=seq_length,
                              validation_ratio=validation_ratio,
                              test_ratio=test_ratio,
                              horizon=horizon,
                              **kwargs
                              )
    else:
        raise Exception(f'Dataset should be either traffic or stocks, received {dataset} instead')