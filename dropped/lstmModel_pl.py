#!/usr/bin/python
# coding=utf-8
import tensorflow as tf
import math
import logging
import sys
from dataProcessing import gradientTools as grads

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class LSTM_Model(object):
    """support multi-gpus
    多卡训练范式: ![](https://www.tensorflow.org/images/Parallelism.png)
    args是参数对象, 含有各种参数属性
    - args.size_cell_hidden: lstm cell的隐层单元个数, 也是一个cell的输出向量长度
    - args.num_projs: LSTMcell原本直接将`hidden_state`作为输出. 不过如果指定了`num_projs`, 就对`hidden_state`进行一个线性变换, 结果作为输出.
    - args.keep_prob: cell的dropout. 不加的话是1
    -

    `gup_id`不是不是对象的属性, 而是对象的成员函数在调用时才能确定, 根据所指定的gpu才能确定的. reuse跟随id_gpu

    输入tensor是batch_major.
    LSTM_Model的lstm_forward()结果`lstm output`是与`input_feature`在时间维度(axis=1)等长的tensor.
    为了实现序列标注, 在`lstm_output`的基础上再加一个`lstm_affine`. `lstm_affine`把`lstm_output`的3d tensor reshape成2dtensor,
    一个sequence的一个时刻(也就是cell的一个时刻的输出)维度作为一个维度, 身下的看做是size_batch. affine layer没有加非线性

    在序列标注任务中, 对于模型最终的输出ligits有两个评价方式: 一个是准确率, 也就是类别匹配的正确的占比, 是离散量; 另一个是cross entropy loss, 是连续量, 可用于梯度计算.
    也就是对logits进行评价的时候才用到了`mask_sequen_length`.

    `input_label`和`input_feature`都是padding过的tensor

    版本迁移:
    使用的API:
    """
    
    num_Instances = 0

    def __init__(self, is_training, args, global_step=None):
        # Initialize some parameters
        self.is_training = is_training
        self.num_gpus = args.num_gpus if is_training else 1
        self.list_gpu_devices = args.list_gpus if is_training else args.list_gpus[0:1]
        self.center_device = "/cpu:0"
        self.num_streams = args.train_streams if is_training else args.dev_streams
        self.batch_size = (args.train_streams * self.num_gpus) if is_training else args.dev_streams
        self.lstm_wgt_initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
        self.forget_bias = 0.0
        self.args = args

        # Return for session
        self.list_run = None
        self.placeholder_list = None
        self.global_step = global_step

        # Build graph
        self.build_graph()
        LSTM_Model.num_Instances += 1
        logging.info("built {} LSTM_Model instance(s).".format(LSTM_Model.num_Instances))

    def build_graph(self):
        # input Placeholders
        with tf.device(self.center_device):
            with tf.variable_scope("inputs"):
                feature_placeholder = tf.placeholder(tf.float32, [self.args.num_steps, self.batch_size, self.args.dim_input], name="input_feature")
                label_placeholder = tf.placeholder(tf.int32, [self.args.num_steps, self.batch_size], name="input_label")
                mask_placeholder = tf.placeholder(tf.float32, [self.args.num_steps, self.batch_size], name="input_mask")
                reset_placeholder = tf.placeholder(tf.bool, [self.batch_size], name="input_reset")
                lr_placeholder = tf.placeholder(tf.float32, name="input_lr") if self.is_training else None
            # split input data alone batch axis to gpus
            with tf.name_scope("splits"):
                feature_splits = tf.split(feature_placeholder, self.num_gpus, axis=1, name="feature_splits")
                label_splits = tf.split(label_placeholder, self.num_gpus, axis=1, name="label_splits")
                mask_splits = tf.split(mask_placeholder, self.num_gpus, axis=1, name="mask_splits")
                reset_flag = tf.split(reset_placeholder, self.num_gpus, axis=0, name="reset_flag")

            frm_num_sum = tf.reduce_sum(mask_placeholder, [0, 1])
            loss_sum = tf.convert_to_tensor(0.0)
            frm_acc_sum = tf.convert_to_tensor(0.0)

        # optimizer
        if self.is_training:
            with tf.name_scope("optimizer"):
                if self.args.optimizer == "adam":
                    logging.info("Using ADAM as optimizer")
                    optimizer = tf.train.AdamOptimizer(lr_placeholder,
                                                       name=self.args.optimizer)
                else:
                    logging.info("Using SGD as optimizer")
                    optimizer = tf.train.GradientDescentOptimizer(lr_placeholder,
                                                                  name=self.args.optimizer)

        tower_grads = []
        list_ops_assign_state = []
        with tf.variable_scope('model', reuse=bool(LSTM_Model.num_Instances)):
        # the outer scope is necessary for the where the reuse scope need to be limited whthin
            for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
                # Computation relevant to frame accuracy and loss
                # with tf.name_scope('gpu'+str(id_gpu)+('_train' if self.is_training else '_dev')):
                with tf.device(name_gpu):
                    zero_states, prev_states = self.create_cell_state(name_gpu)

                    def choose_device(op, device):
                        if op.type.startswith('Variable'):
                            device = self.center_device
                        return device
                    with tf.device(lambda op: choose_device(op, name_gpu)):
                        # Construct lstm cell
                        multi_cell = self.build_lstm_cell()
                        # Construct output affine layers
                        w, b = self.build_affine_layer()

                        lstm_output, assign_state = self.lstm_forward(cell=multi_cell,
                                                                      inputs=feature_splits[id_gpu],
                                                                      zero_states=zero_states,
                                                                      prev_states=prev_states,
                                                                      reset_flag=reset_flag[id_gpu])

                    logits = self.affine_forward(lstm_output, w=w, b=b)
                    # Accuracy
                    with tf.name_scope("label_accuracy"):
                        correct = tf.nn.in_top_k(logits, tf.reshape(label_splits[id_gpu], [-1]), 1)
                        correct = tf.multiply(tf.cast(correct, tf.float32), tf.reshape(mask_splits[id_gpu], [-1]))
                        frm_acc = tf.reduce_sum(correct)
                        frm_acc_sum += frm_acc
                    # Cross entropy loss
                    with tf.name_scope("CE_loss"):
                        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.reshape(label_splits[id_gpu], [-1]), logits=logits)
                        cross_entropy = tf.multiply(cross_entropy, tf.reshape(mask_splits[id_gpu], [-1]))
                        cross_entropy_loss = tf.reduce_sum(cross_entropy)
                        loss = cross_entropy_loss
                    loss_sum += loss

                    tf.get_variable_scope().reuse_variables()  # all the tf.get_variable will reuse variable

                    if self.is_training:
                        with tf.name_scope("gradients"):
                            gradients = optimizer.compute_gradients(loss)
                        tower_grads.append(gradients)

                list_ops_assign_state.append(assign_state)
                logging.info('\tbuild model on {} succesfully!'.format(name_gpu))

        self.placeholder_list = [feature_placeholder, label_placeholder, mask_placeholder, reset_placeholder]
        self.list_run = [loss_sum, frm_acc_sum, frm_num_sum, list_ops_assign_state]

        # merge gradients, update current model
        if self.is_training:
            with tf.device(self.center_device):
                # computation relevant to gradient
                averaged_grads = grads.average_gradients(tower_grads)
                handled_grads = grads.handle_gradients(averaged_grads, self.args)
                # with tf.variable_scope('adam', reuse=False):
                op_optimize = optimizer.apply_gradients(handled_grads, self.global_step)

            self.placeholder_list.append(lr_placeholder)
            self.list_run.append(op_optimize)

            if self.args.is_debug:
                with tf.name_scope("summaries"):
                    tf.summary.scalar('loss', loss_sum/frm_num_sum)
                    tf.summary.scalar('frame_accuracy', frm_acc_sum/frm_num_sum)
                    tf.summary.scalar('learning_rate', lr_placeholder)
                    op_summary = tf.summary.merge_all()
                self.list_run.append(op_summary)

    def build_lstm_cell(self):
        # because we are building models, creating variables in the same time , so we have variable scope
        with tf.name_scope("lstm"):
            def fentch_cell():
                return tf.contrib.rnn.LSTMCell(self.args.size_cell_hidden,
                                               use_peepholes=True,
                                               num_proj=self.args.num_projs,
                                               cell_clip=50.0,
                                               initializer=self.lstm_wgt_initializer,
                                               forget_bias=self.forget_bias)
            list_cells = [fentch_cell() for _ in range(self.args.num_lstm_layers)]
            # Add dropout
            if self.is_training and self.args.keep_prob < 1:
                list_cells = [tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=self.args.keep_prob) for c in list_cells]
            # multiple lstm layers
            multi_cell = tf.contrib.rnn.MultiRNNCell(list_cells)

        return multi_cell

    def create_cell_state(self, name_gpu):
        with tf.name_scope('cell_states'):
            total_state_dim = (self.args.size_cell_hidden + self.args.num_projs) * self.args.num_lstm_layers
            prev_states = tf.Variable(tf.zeros([self.num_streams, total_state_dim]),
                                      name="prev_states",
                                      trainable=False)
            zero_states = tf.Variable(tf.zeros([self.num_streams, total_state_dim]),
                                      name="zero_states",
                                      trainable=False)

        return zero_states, prev_states

    def reset_state(self, reset_flag, zero_states, prev_states):
        states_after_reset = []
        for i in range(self.num_streams):
            states_after_reset.append(tf.cond(reset_flag[i],
                                              lambda: tf.slice(zero_states, [i, 0], [1, -1]),
                                              lambda: tf.slice(prev_states, [i, 0], [1, -1])))
        states_after_reset = tf.concat(states_after_reset, axis=0)

        # Wrap previous state after reset
        layer_states = []
        for i in range(self.args.num_lstm_layers):
            layer_states.append(tf.contrib.rnn.LSTMStateTuple(
                tf.slice(states_after_reset, [0, i * (self.args.size_cell_hidden + self.args.num_projs)], [-1, self.args.size_cell_hidden]),
                tf.slice(states_after_reset, [0, i * (self.args.size_cell_hidden + self.args.num_projs) + self.args.size_cell_hidden],
                         [-1, self.args.num_projs])))
        state = tuple(layer_states)

        return state

    def lstm_forward(self, cell, inputs, zero_states, prev_states, reset_flag):
        # we have finishing building the model, the tf.nn.dynamic_rnn is a function, and not create new variable, so no need of variable scope
        # `time_major` default is False
        # the variable created in `tf.nn.dynamic_rnn`, not in cell
        with tf.variable_scope("lstm"):
            lstm_output, state = tf.nn.dynamic_rnn(cell,
                                                   inputs,
                                                   initial_state=self.reset_state(reset_flag, zero_states, prev_states),
                                                   time_major=True,
                                                   dtype=tf.float32)
        # Concatenate state
        concat_states = []  # [c, h, c, h , ... , c, h]
        for i in range(self.args.num_lstm_layers):
            concat_states.append(state[i].c)
            concat_states.append(state[i].h)

        # Assign to variable prev_states
        assign_state = tf.assign(prev_states, tf.concat(axis=1, values=concat_states))

        return lstm_output, assign_state

    def build_affine_layer(self):
        with tf.variable_scope("output_layer"):
            w = tf.get_variable("output_linearity",
                                [self.args.num_projs, self.args.dim_output],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.args.num_projs))))
            b = tf.get_variable("output_bias",
                                [self.args.dim_output],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())
        return w, b

    def affine_forward(self, input_tensor, w, b):
        # reshape the 3d lstm_output into 2d tensor
        with tf.name_scope('affine'):
            logits = tf.matmul(tf.reshape(input_tensor, [-1, self.args.num_projs]), w) + b
        return logits


if __name__ == '__main__':
    from dataProcessing.kaldiModel import KaldiModel, build_kaldi_lstm_layers, build_kaldi_output_affine
    from configs.arguments import args
    import os

    os.chdir('/mnt/lustre/xushuang/easton/projects/mix_model_2.0')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    logging.info('CUDA_VISIBLE_DEVICES : {}'.format(args.gpus))

    logging.info('args.dim_input : {}'.format(args.dim_input))

    # from tensorflow.python.client import device_lib
    # local_device_protos = device_lib.list_local_devices()
    # logging.info(local_device_protos)

    global_step = tf.train.get_or_create_global_step()

    graph_train = LSTM_Model(True, args, global_step)
    logging.info('build training graph successfully!')
    graph_dev = LSTM_Model(False, args)
    logging.info('build dev graph successfully!')

    if args.is_debug:
        list_ops = [op.name+' '+op.device for op in tf.get_default_graph().get_operations()]
        # list_variables_and_devices = [op.name+' '+op.device for op in tf.get_default_graph().get_operations() if op.type.startswith('Variable')]
        list_variables_and_devices = [v for v in tf.trainable_variables() if "lstm" in v.name]
        # logging.info('\n'.join(list_variables_and_devices))
        logging.info(list_variables_and_devices)


    list_kaldi_layers = []
    list_kaldi_layers = build_kaldi_lstm_layers(list_kaldi_layers, args.num_lstm_layers, args.dim_input, args.num_projs)
    list_kaldi_layers = build_kaldi_output_affine(list_kaldi_layers)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.train.MonitoredTrainingSession(config=config) as sess:
        kaldi_model = KaldiModel(list_kaldi_layers)
        kaldi_model.loadModel(sess=sess, model_path=args.model_init)
        kaldi_model.saveModel(sess=sess, model_path=args.model_init, args=args)
