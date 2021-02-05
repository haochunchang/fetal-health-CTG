#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Author: Hao Chun Chang
# Contact: changhaochun84@gmail.com
# Description: Model definition

from singa import layer
from singa import model
from singa import tensor


class MLP(model.Model):

    def __init__(self, perceptron_size=100, num_classes=10):
        super(MLP, self).__init__()
        self.num_classes = num_classes

        self.relu = layer.ReLU()
        self.linear1 = layer.Linear(perceptron_size)
        self.linear2 = layer.Linear(num_classes)
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()

    def forward(self, inputs):
        y = self.linear1(inputs)
        y = self.relu(y)
        y = self.linear2(y)
        return y

    def train_one_batch(self, x, y):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)
        self.optimizer(loss)
        return out, loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


def create_MLP_model(pretrained=False, **kwargs):
    """Constructs a MLP model.
    Args:
        pretrained (bool): If True, returns a model pre-trained
    """
    model = MLP(**kwargs)

    return model


__all__ = ['MLP', 'create_MLP_model']
