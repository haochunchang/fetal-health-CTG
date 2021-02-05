#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Author: Hao Chun Chang
# Contact: changhaochun84@gmail.com
# Description: Data preprocessing and exploration

import os
import math
import pandas as pd
import numpy as np


class DataLoader(object):
    """Loading data and performs data transformation

    Args:
        path (string): System path of the data.
        val_size (float): Proportion of the validation set (default=0.2)
        random_seed (int): seed for splitting train/valid subsets
    """
    def __init__(self, path, val_size=0.2, random_seed=9487):
        assert val_size > 0 and val_size < 1, "validation size must in the range (0, 1)"
        self.path = path
        self.data = None
        self.data = pd.read_csv(self.path)
        self.data["subset"] = "train"

        num_samples = self.data.shape[0]
        np.random.seed(random_seed)
        random_index = np.random.randint(
            low=0, high=num_samples, size=math.floor(num_samples * val_size)
        )
        self.data.loc[random_index, "subset"] = "valid"

    def __str__(self):
        return "This Dataloader loads data from " + self.path

    def load_data(self, subset="train"):
        """Load the data into features and labels.

        Args:
            subset (string): either "train" (default) or "valid".
        Returns:
            data: pd.DataFrame, table with feature columns.
            label: pd.Series, table with label values:
                (Normal=1, Suspect=2, Pathogenic=3)
        """
        assert subset in ["train", "valid"], "subset must be either 'train' or 'valid'"
        label_col = "fetal_health"
        data = self.data.loc[
            self.data["subset"] == subset,
            ~self.data.columns.isin([label_col, "subset"])
        ]
        label = self.data.loc[
            self.data["subset"] == subset,
            label_col
        ]
        return data.to_numpy(), label.to_numpy()

    def standardize_column(self, column_name):
        """Standardize the given column into zero mean and unit standard deviation.

        Args:
            column_name (string): target column to standardize.
        Returns:
            None, only modified the given columns.
        """
        assert self.data is not None
        assert column_name in self.data, "Column name not exists in data."
        col = self.data[column_name]
        col = (col - col.mean()) / (col.std() + 1e-7)
        self.data[column_name] = col


__all__ = ['DataLoader']


if __name__ == "__main__":
    pass
    # data_loader = DataLoader(os.path.join("data", "fetal_health.csv"))
    # data_loader.load()
    # data_loader.standardize_column("baseline value")
    # print(data_loader.data.head())
