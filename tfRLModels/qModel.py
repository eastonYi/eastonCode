'''
Q value based model, like DQN
'''
import tensorflow as tf
from tensorflow.python.util import nest

from tfModels.tools import choose_device
from RLModel import RLModel
from .agents.lstm_agent import LSTM_Agent as Agent
from .envs.lm_env import LM_ENV as ENV
from .processors.conv import CONV_Processor as Processor
from tfModels.OptimalDistill import Qvalue


class Q_Model(RLModel):
    def __init__(self, args, name):
        self.processor = Processor(args)
        self.agent = Agent(args)
        self.env = ENV(args)
        self.env.reset()
        super().__init__(args, name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        tf.get_variable_scope().set_initializer(tf.variance_scaling_initializer(
            1.0, mode="fan_avg", distribution="uniform"))
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            batch_size = tf.shape(tensors_input.len_fea_splits[id_gpu])
            state_lm_init = self.env.lm.zero_state(batch_size)
            rewards_lm_init = tf.zeros([batch_size, 0])
            actions_init = tf.zeros([batch_size, 0])

            frames, len_frames = self.processor.process(
                features=tensors_input.feature_splits[id_gpu],
                len_feas=tensors_input.len_fea_splits[id_gpu])

            def step(i, state_lm, rewards_lm, actions):
                # generate env state
                state_ac = frames[:, i, :]
                state = tf.concat([state_ac, state_lm], 1)

                # agent takes action
                action = self.agent.forward(state)

                # env transfers staten and bills rewards
                next_state_lm, reward_lm, info = self.env.step(action)
                rewards_lm = tf.concat([rewards_lm, reward_lm], 1)
                actions = tf.concat([actions, action], 1)

                return i+1, next_state_lm, rewards_lm, actions

            _, _, rewards_lm, actions = tf.while_loop(
                cond=lambda i, *_: tf.less(i, tf.shape(frames)[1]),
                body=step,
                loop_vars=[0, state_lm_init, rewards_lm_init, actions_init],
                shape_invariants=[tf.TensorShape([]),
                                  nest.map_structure(lambda t: tf.TensorShape(t.shape), state_lm_init),
                                  tf.TensorShape([None, None]),
                                  tf.TensorShape([None, None])]
                )

            q_value = Qvalue(actions, tensors_input.target_labels[id_gpu])
            # rewards_ac: the temporal-difference Q value of each step
            rewards_ac = q_value[:, 1:] - q_value[:, :-1]
            rewards = rewards_ac + rewards_lm
            rewards_discounted = self.discount(self.discount_rate, rewards)

            return
