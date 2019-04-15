'''
policy model for language model env random working
'''
import logging
import sys
import tensorflow as tf

from tfModels.tools import choose_device, smoothing_cross_entropy
from .ReinforcementLearning import RL
from .agents.q_estimator import Q_Estimator as Agent
from .envs.bandit import Bandit as ENV

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class QLearning(RL):
    def __init__(self, is_train, args, batch=None, name='bandit_Q_estimate'):
        self.args = args
        self.is_train = is_train
        self.env = ENV(is_train, args)
        self.size = self.env.size
        self.agent = Agent(self.env.size, is_train, args)
        self.env.reset()

        self.average_reward = 0
        self.q_estimation_init = tf.ones(self.size) * 21
        self.action_count_init = tf.zeros([self.size])

        self.learning_rate = tf.constant(args.learning_rate)

        self.sample_averages = args.sample_averages
        self.gradient = args.agent.gradient
        self.gradient_baseline = args.gradient_baseline
        self.list_run = list(self.build_graph())

    def reset(self):
        self.average_reward = 0

    def build_graph(self):
        rewards_init = tf.zeros([0])

        def step(i, rewards, action_count, q_estimation):
            # move
            action = self.agent.forward()

            # reward
            reward = self.env.step(action)

            # reward = tf.Print(reward, [action, reward], message='a-r:', summarize=1000)

            rewards = tf.concat([rewards, reward[None]], 0)

            ## compte average_reward in incremental way
            self.average_reward = tf.to_float((i-1)/i) * self.average_reward + reward/tf.to_float(i)
            action_count += tf.one_hot(action, depth=self.size)

            # i = tf.Print(i, [i, action_count], message='action_count: ', summarize=1000)

            # update estimation
            if self.sample_averages:
                # update estimation using sample averages (Eq 2.3)
                q_estimation[action] += 1.0 / action_count[action] * (reward - q_estimation[action])
            elif self.gradient:
                one_hot = tf.zeros([self.size])
                one_hot[action] = 1

                baseline = self.average_reward if self.gradient_baseline else 0
                self.action_prob = tf.nn.softmax(q_estimation)
                q_estimation += self.learning_rate * (reward - baseline) * (one_hot - self.action_prob)
            else:
                # update estimation with constant step size
                musk_action = tf.one_hot(action, depth=self.size)
                q_estimation += self.learning_rate * (reward - q_estimation) * musk_action

            return i+1, rewards, action_count, q_estimation

        _, rewards, action_count, q_estimation = tf.while_loop(
            cond=lambda i, *_: tf.less(i, self.args.times),
            body=step,
            loop_vars=[0, rewards_init, self.action_count_init, self.q_estimation_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None]),
                              tf.TensorShape([self.size]),
                              tf.TensorShape([self.size])]
            )
        self.agent.action_count = action_count
        self.agent.q_estimation = q_estimation

        return rewards, q_estimation
