import tensorflow as tf
from abc import ABCMeta, abstractmethod


class Processor():
    __metaclass__ = ABCMeta

    def __init__(self, is_train, args, name):
        self.is_train = is_train
        self.args = args
        self.name = name

    def __call__(self):
        return self.forward()

    @abstractmethod
    def process(self):
        '''
        '''
