'''
policy model for language model env random working
'''
import logging
import sys
import tensorflow as tf
from tensorflow.python.util import nest
from collections import namedtuple

from tfModels.tools import choose_device, smoothing_cross_entropy
from .ReinforcementLearning import RL
from .agents.lstm_agent import LSTM_Agent as Agent
from .envs.lm_env import LM_ENV as ENV

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class PolicyLearning(RL):
    def __init__(self, tensor_global_step, is_train, args, batch=None, name='policy_model'):
        self.agent = Agent(is_train, args)
        self.env = ENV(args.lm_obj, is_train, args)
        self.env.reset()
        self.choose = tf.random.uniform([], minval=0, maxval=None)
        super().__init__(tensor_global_step, is_train, args, batch, name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        tf.get_variable_scope().set_initializer(tf.variance_scaling_initializer(
            1.0, mode="fan_avg", distribution="uniform"))
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            batch_size = 1200 if self.is_train else 3
            state_agent_init = self.agent.zero_state(batch_size)
            state_lm_init = self.env.lm.zero_state(batch_size)
            rewards_lm_init = tf.zeros([batch_size, 0])
            actions_init = tf.zeros([batch_size, 0], dtype=tf.int32)
            logits_init = tf.zeros([batch_size, 0, self.args.dim_output])

            def step(i, state_agent, state_lm, rewards_lm, actions, logits):
                # generate env state
                state_env = tf.concat(state_lm[-1], -1)

                # agent takes action
                cur_logit, next_state_agent = self.agent.forward(state_env, state_agent)
                policy = tf.nn.softmax(cur_logit, name='actor_prob')

                action = tf.cond(
                    tf.less(self.choose, 0.8),
                    lambda: tf.distributions.Categorical(probs=policy).sample(),
                    lambda: tf.distributions.Categorical(logits=tf.ones_like(cur_logit)).sample())

                logits = tf.concat([logits, cur_logit[:, None, :]], 1)

                # env transfers staten and bills rewards
                next_state_lm, reward_lm, info = self.env.step(action, state_lm)
                rewards_lm = tf.concat([rewards_lm, reward_lm[:, None]], 1)
                actions = tf.concat([actions, action[:, None]], 1)

                return i+1, next_state_agent, next_state_lm, rewards_lm, actions, logits

            _, _, _, rewards_lm, actions, logits = tf.while_loop(
                cond=lambda i, *_: tf.less(i, 20),
                body=step,
                loop_vars=[0, state_agent_init, state_lm_init, rewards_lm_init, actions_init, logits_init],
                shape_invariants=[tf.TensorShape([]),
                                  nest.map_structure(lambda t: tf.TensorShape(t.shape), state_agent_init),
                                  nest.map_structure(lambda t: tf.TensorShape(t.shape), state_lm_init),
                                  tf.TensorShape([None, None]),
                                  tf.TensorShape([None, None]),
                                  tf.TensorShape([None, None, self.args.dim_output])]
                )

            if self.is_train:
                rewards = rewards_lm
                rewards_discounted = self.discount(self.discount_rate, rewards)
                rewards_discounted = tf.stop_gradient(rewards_discounted)

                crossent = smoothing_cross_entropy(
                    logits=logits,
                    labels=actions,
                    vocab_size=self.args.dim_output,
                    confidence=1.0)
                loss = crossent * rewards_discounted

                loss = tf.reduce_mean(loss)
                gradients = self.optimizer.compute_gradients(loss)

            self.__class__.num_Model += 1
            logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
                self.__class__.__name__, name_gpu, self.__class__.num_Model))

            if self.is_train:
                return loss, gradients, \
                [tf.reduce_sum(rewards_lm, -1), actions]
            else:
                return actions, rewards_lm

    def build_infer_graph(self):
        tensors_input = self.build_infer_input()

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            actions, rewards_lm = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

        return actions, tensors_input.shape_batch, rewards_lm

    def build_tf_input(self):
        """
        stand training input
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_fea_splits, len_label_splits, shape_batch')

        # split input data alone batch axis to gpus
        tensors_input.feature_splits = None
        tensors_input.label_splits = None
        tensors_input.len_fea_splits = None
        tensors_input.len_label_splits = None
        tensors_input.shape_batch = tf.constant([900, 9, 10])

        return tensors_input

    def build_infer_input(self):
        return self.build_tf_input()
