import tensorflow as tf
from .env import Environment


class Bandit(Environment):
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, is_train, args, name='k-armed_bandit'):
        self.true_reward = [int(i) for i in args.env.true_rewards.split(',')]
        self.size = len(self.true_reward)
        self.q_true = tf.random.normal([self.size]) + self.true_reward
        self.best_action = tf.argmax(self.q_true)
        super().__init__(is_train, args, name)

    def reset(self):
        '''
        '''
        return

    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = tf.random.normal([]) + self.q_true[action]

        return reward
