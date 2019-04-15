'''
the template of agents
'''
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod

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

    @staticmethod
    def exploration(logit):
        return tf.distributions.Categorical(logits=tf.ones_like(logit)).sample()

    @staticmethod
    def exploitation(logits=None, prob=None):
        return tf.distributions.Categorical(logits=logits, prob=prob).sample()
