import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

from .env import Environment
from tfTools.tfMath import Vander_Monde_matrix


class LM_ENV(Environment):
    '''
    memoryless env: step() is memoryless
    '''
    def __init__(self, lm_obj, is_train, args, name='lm_env'):
        self.lm = lm_obj
        self.stay_idx = args.dim_output-1
        super().__init__(is_train, args, name)

    def step(self, action, lm_state_prev):
        # stay or move
        stay_musk = tf.equal(action, self.stay_idx)

        # move
        prob, _, lm_state = self.lm.forward(
            input=action,
            state=lm_state_prev,
            stop_gradient=True,
            list_state=True)
        lm_state_move = lm_state
        size_batch = tf.shape(stay_musk)[0]
        indices_batch = tf.range(size_batch)
        # checked
        reward_move = tf.gather_nd(prob, tf.stack([indices_batch, action], -1))

        # stay
        lm_state_stay = lm_state_prev
        reward_stay = tf.zeros_like(reward_move)

        # merge

        states = [tf.where(stay_musk, x, y) \
            for x, y in zip(self.state2list_tensors(lm_state_stay),
                            self.state2list_tensors(lm_state_move))]
        next_state = self.list_tensors2state(states)
        reward = tf.where(stay_musk, reward_stay, reward_move)
        # reward = tf.Print(reward, [reward], message='reward', summarize=1000)
        info = [reward_stay, reward_move]

        return next_state, reward, info

    def reset(self):
        return

    @staticmethod
    def state2list_tensors(state):
        """
        for lstm cell
        """
        import itertools
        list_tensors = list(itertools.chain.from_iterable(state))

        return list_tensors

    @staticmethod
    def list_tensors2state(list_tensors):
        """
        for lstm cell
        """
        from tensorflow.contrib.rnn import LSTMStateTuple
        cells = []
        for i in range(int(len(list_tensors)/2)):
            cells.append(LSTMStateTuple(list_tensors[2*i], list_tensors[2*i+1]))

        return tuple(cells)
