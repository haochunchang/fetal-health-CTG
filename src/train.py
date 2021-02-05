#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Author: Hao Chun Chang
# Contact: changhaochun84@gmail.com
# Description: Driver code for training models

import os
import numpy as np
from singa import opt, tensor
from singa.device import get_default_device

from model import create_MLP_model
from preprocess import DataLoader


def train():
    """Start the training procedure 
    """
    num_epochs = 1
    learning_rate = 0.05
    batch_size = 8

    data_loader = DataLoader(os.path.join("data", "fetal_health.csv"))
    data_loader.standardize_column("baseline value")
    x_train, y_train = data_loader.load_data(subset="train")
    x_valid, y_valid = data_loader.load_data(subset="valid")

    num_classes = len(np.unique(y_train))
    num_samples, num_features = x_train.shape

    assert x_train.shape[1] == x_valid.shape[1], "Number of features should be equal!"
    assert x_train.shape[0] == y_train.shape[0], "Number of training samples should be equal!"
    assert x_valid.shape[0] == y_valid.shape[0], "Number of validation samples should be equal!"

    dev = get_default_device()
    tx = tensor.Tensor((num_samples, num_features), dev, tensor.float32)
    ty = tensor.Tensor((num_samples,), dev, tensor.int32)

    sgd = opt.SGD(learning_rate)
    model = create_MLP_model(perceptron_size=10, num_classes=num_classes)
    model.set_optimizer(sgd)
    model.compile([tx], is_train=True, use_graph=True, sequential=False)
    model.train()

    for i in range(num_epochs):
        tx.copy_from_numpy(x_train.astype(np.float32))
        ty.copy_from_numpy(y_train.astype(np.int32))
        out, loss = model(tx, ty, 'fp32', spars=None)

        # TODO: Add metric evaluation on validation data
        if i % 10 == 0:
            print("training loss = {:.3f}".format(tensor.to_numpy(loss)[0]))

    # model.save()


if __name__ == "__main__":
    import argparse
    train()
