import tensorflow as tf
import os
import sys
import yaml
from argparse import ArgumentParser

from utils.configReader import AttrDict
from tfRLModels.envs.bandit import Bandit
from tfRLModels.agents.q_estimator import Q_Estimator
from tfRLModels.simpleQLearning import QLearning
import numpy as np


np.set_printoptions(precision=2)

CONFIG_FILE = sys.argv[-1]
args = AttrDict(yaml.load(open(CONFIG_FILE)))

def main(mode):
    q_learning = QLearning(is_train=True, args=args)

    with tf.train.MonitoredTrainingSession() as sess:
        for _ in range(2000):
            reward, q = sess.run(q_learning.list_run)
            print('reward: {:.3f}, Q: {}'.format(sum(reward)/len(reward), q))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    main(param.mode)
