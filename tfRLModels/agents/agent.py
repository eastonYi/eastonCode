'''
the template of agents
'''
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod

from tfModels.layers import residual, conv_lstm
from tfModels.tensor2tensor.common_layers import layer_norm

class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self, is_train, args, name):
        self.is_train = is_train
        self.args = args
        self.name = name

    def __call__(self):
        return self.forward()

    @abstractmethod
    def forward(self):
        """
        """
