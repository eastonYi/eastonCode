import tensorflow as tf
from abc import ABCMeta, abstractmethod

from .agents.lstm_agent import LSTM_Agent
from tfTools.tfMath import Vander_Monde_matrix


class RLModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, args, name):
        self.args = args
        self.name = name

    @staticmethod
    def discount(discount_rate, rewards):
        '''
        rewards: [b, t]

        Demo:
            rewards = tf.ones([3,4])
            sess.run(discount(0.9, rewards))
            [[3.4390001, 2.71     , 1.9      , 1.       ],
             [3.4390001, 2.71     , 1.9      , 1.       ],
             [3.4390001, 2.71     , 1.9      , 1.       ]]
        '''
        batch_size = tf.shape(rewards)[0]
        time_len = tf.shape(rewards)[1]
        discount_rewards_init = tf.zeros([batch_size, 0])

        def step(i, discount_rewards):
            discount_reward = tf.reduce_sum(rewards[:, i:] * \
                Vander_Monde_matrix(discount_rate, batch_size)[:, :tf.shape(rewards)[0]-i+1], 1)
            discount_rewards = tf.concat([discount_rewards, discount_reward], 1)

            return i+1, discount_rewards

        _, discount_rewards = tf.while_loop(
            cond=lambda i, *_: tf.less(i, time_len),
            body=step,
            loop_vars=[0, discount_rewards_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None])]
            )

        return discount_rewards
