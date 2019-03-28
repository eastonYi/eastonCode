import tensorflow as tf

from env import Environment
from tfModels.OptimalDistill import OCD
from tfTools.tfMath import Vander_Monde_matrix


class LM_ENV(Environment):
    def __init__(self, lm_obj, args, name):
        self.lm = lm_obj
        super().__init__(args, name)

    def step(self, action, pre_actions, targets):
        lm_state, preds, prob = self.lm.forward()

        next_state = lm_state
        reward = prob
        done = tf.equal(preds, self.args.eos_idx)
        info = None

        return next_state, reward, done, info

    def discounted_rewards(self, actions, targets, lm_rewards):
        rewards_ac = OCD(
            actions,
            targets)
        rewards_lm = lm_rewards

        rewards = rewards_lm + rewards_ac
        # TODO
        discounted_rewards = rewards * Vander_Monde_matrix(self.discount_rate)

        return discounted_rewards
