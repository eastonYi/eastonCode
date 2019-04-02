import tensorflow as tf
import logging
import sys
from collections import namedtuple
from abc import ABCMeta, abstractmethod

from .agents.lstm_agent import LSTM_Agent
from tfTools.tfMath import Vander_Monde_matrix
from tfTools.gradientTools import average_gradients, handle_gradients

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class RLModel(object):
    __metaclass__ = ABCMeta
    num_Instances = 0
    num_Model = 0

    def __init__(self, tensor_global_step, is_train, args, batch=None, name='rl_model'):
        self.discount_rate = args.model.discount_rate
        self.args = args
        self.name = name
        self.is_train = is_train
        self.global_step = tensor_global_step

        self.num_gpus = args.num_gpus if is_train else 1
        self.list_gpu_devices = args.list_gpus
        self.center_device = "/cpu:0"
        self.learning_rate = None
        self.build_input = self.build_tf_input if batch else self.build_pl_input
        self.batch = batch

        self.list_pl = None

        # Build graph
        self.list_run = list(self.build_graph() if is_train else self.build_infer_graph())

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

    def build_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_input()
        # create optimizer
        self.optimizer = self.build_optimizer()
        if 'horovod' in sys.modules:
            import horovod.tensorflow as hvd
            logging.info('wrap the optimizer with horovod!')
            self.optimizer = hvd.DistributedOptimizer(self.optimizer)

        loss_step = []
        tower_grads = []
        list_debug = []
        # the outer scope is necessary for the where the reuse scope need to be limited whthin
        # or reuse=tf.get_variable_scope().reuse
        for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
            # with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                loss, gradients, debug = self.build_single_graph(id_gpu, name_gpu, tensors_input)
                loss_step.append(loss)
                tower_grads.append(gradients)
                list_debug.append(debug)

        # mean the loss
        loss = tf.reduce_mean(loss_step)
        # merge gradients, update current model
        with tf.device(self.center_device):
            # computation relevant to gradient
            averaged_grads = average_gradients(tower_grads)
            handled_grads = handle_gradients(averaged_grads, self.args)
            # with tf.variable_scope('adam', reuse=False):
            op_optimize = self.optimizer.apply_gradients(handled_grads, self.global_step)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(self.__class__.num_Instances, self.__class__.__name__))

        return loss, tensors_input.shape_batch, op_optimize, [x for x in zip(*list_debug)]

    def build_tf_input(self):
        """
        stand training input
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_fea_splits, len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(self.batch[0], self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(self.batch[1], self.num_gpus, name="label_splits")
                tensors_input.len_fea_splits = tf.split(self.batch[2], self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(self.batch[3], self.num_gpus, name="len_label_splits")
        tensors_input.shape_batch = tf.shape(self.batch[0])

        return tensors_input

    def build_pl_input(self):
        """
        use for training. but recomend to use build_tf_input insted
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_fea_splits, len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                batch_features = tf.placeholder(tf.float32, [None, None, self.args.data.dim_input], name='input_feature')
                batch_labels = tf.placeholder(tf.int32, [None, None], name='input_labels')
                batch_fea_lens = tf.placeholder(tf.int32, [None], name='input_fea_lens')
                batch_label_lens = tf.placeholder(tf.int32, [None], name='input_label_lens')
                self.list_pl = [batch_features, batch_labels, batch_fea_lens, batch_label_lens]
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(batch_labels, self.num_gpus, name="label_splits")
                tensors_input.len_fea_splits = tf.split(batch_fea_lens, self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(batch_label_lens, self.num_gpus, name="len_label_splits")
        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

    def build_infer_input(self):
        """
        used for inference. For inference must use placeholder.
        during the infer, we only get the decoded result and not use label
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, len_fea_splits, label_splits, len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                batch_features = tf.placeholder(tf.float32, [None, None, self.args.data.dim_input], name='input_feature')
                batch_fea_lens = tf.placeholder(tf.int32, [None], name='input_fea_lens')
                self.list_pl = [batch_features, batch_fea_lens]
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.len_fea_splits = tf.split(batch_fea_lens, self.num_gpus, name="len_fea_splits")

        tensors_input.label_splits = None
        tensors_input.len_label_splits = None
        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

    def build_optimizer(self):
        self.learning_rate = tf.convert_to_tensor(self.args.learning_rate)

        if 'horovod' in sys.modules:
            import horovod.tensorflow as hvd
            logging.info('wrap the optimizer with horovod!')
            self.learning_rate = self.learning_rate * hvd.size()

        with tf.name_scope("optimizer"):
            if self.args.optimizer == "adam":
                logging.info("Using ADAM as optimizer")
                optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                                   beta1=0.9,
                                                   beta2=0.98,
                                                   epsilon=1e-9,
                                                   name=self.args.optimizer)
            elif self.args.optimizer == "adagrad":
                logging.info("Using Adagrad as optimizer")
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                logging.info("Using SGD as optimizer")
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate,
                                                              name=self.args.optimizer)
        return optimizer

    def variables(self, scope=None):
        '''get a list of the models's variables'''
        scope = scope if scope else self.name
        scope += '/'
        print('all the variables in the scope:', scope)
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=scope)

        return variables
