import tensorflow as tf
from abc import ABCMeta, abstractmethod


class Environment(object):
    __metaclass__ = ABCMeta

    def __init__(self, args, name):
        self.discount_rate = args.model.discount_rate
        self.args = args
        self.name = name

    def __call__(self, action):
        return self.step(action)

    @abstractmethod
    def step(self, action):
        '''
        '''
        # return next_state, reward, done, info

    @abstractmethod
    def discounted_rewards(self):
        '''
        '''
