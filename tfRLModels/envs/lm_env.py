import tensorflow as tf

from env import Environment
from tfTools.tfMath import Vander_Monde_matrix


class LM_ENV(Environment):
    def __init__(self, lm_obj, args, name):
        self.lm = lm_obj
        self.lm_state = lm_obj.zero_state()
        self.stay_idx = self.args.dim_output-1
        name = name if name else 'lm_env'
        super().__init__(args, name)

    def step(self, action):
        # stay or move
        stay_musk = tf.equal(action, self.stay_idx)

        # move
        prob, pred, lm_state = self.lm.forward(
            input=action,
            state=self.lm_state,
            stop_gradient=True,
            list_state=True)
        lm_state_move = lm_state
        reward_move = prob

        # stay
        lm_state_stay = self.lm_state
        reward_stay = tf.zeros_like(pred)

        # merge
        next_state = tf.where(stay_musk, lm_state_stay, lm_state_move)
        reward = tf.where(stay_musk, reward_stay, reward_move)
        info = None

        return next_state, reward, info
