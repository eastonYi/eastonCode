import tensorflow as tf
import logging
import sys
from collections import namedtuple
from tensorflow.contrib.layers import fully_connected

from tfTools.gradientTools import average_gradients, handle_gradients
from tfModels.tools import warmup_exponential_decay, choose_device, lr_decay_with_warmup, stepped_down_decay, exponential_decay


class LSTM_Model(object):
    num_Instances = 0
    num_Model = 0
