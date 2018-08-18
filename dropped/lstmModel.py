import tensorflow as tf
import math
import logging
import sys
from dataProcessing import gradientTools as grads
from tfModels.tools import warmup_exponential_decay


class LSTM_Model(object):
    '''
    support multi-gpus
    多卡训练范式: ![](https://www.tensorflow.org/images/Parallelism.png)
    args是参数对象, 含有各种参数属性
    - args.size_cell_hidden: lstm cell的隐层单元个数, 也是一个cell的输出向量长度
    - args.num_projs: LSTMcell原本直接将`hidden_state`作为输出. 不过如果指定了`num_projs`, 就对`hidden_state`进行一个线性变换, 结果作为输出.
    - args.keep_prob: cell的dropout. 不加的话是1

    输入tensor是batch_major.
    LSTM_Model的lstm_forward()结果`lstm output`是与`input_feature`在时间维度(axis=1)等长的tensor.
    为了实现序列标注, 在`lstm_output`的基础上再加一个`lstm_affine`. `lstm_affine`把`lstm_output`的3d tensor reshape成2dtensor,
    一个sequence的一个时刻(也就是cell的一个时刻的输出)维度作为一个维度, 身下的看做是size_batch. affine layer没有加非线性

    在序列标注任务中, 对于模型最终的输出ligits有两个评价方式: 一个是准确率, 也就是类别匹配的正确的占比, 是离散量; 另一个是cross entropy loss, 是连续量, 可用于梯度计算.
    也就是对logits进行评价的时候才用到了`mask_sequen_length`.

    `input_label`和`input_feature`都是padding过的tensor

    版本迁移:
    使用的API:

    '''
    num_Instances = 0

    def __init__(self, batch, global_step, is_training, args):
        # Initialize some parameters
        self.is_training = is_training
        self.batch = batch
        self.num_gpus = args.num_gpus if is_training else 1
        self.list_gpu_devices = args.list_gpus if is_training else args.list_gpus[0:1]
        self.center_device = "/cpu:0"
        self.batch_size = (args.train_streams * self.num_gpus) if is_training else args.dev_streams
        self.forget_bias = 0.0
        self.learning_rate = None
        self.args = args

        # Return for session
        self.list_run = None
        self.global_step = global_step

        # Build graph
        self.build_graph()
        LSTM_Model.num_Instances += 1
        logging.info("built {} LSTM_Model instance(s).".format(LSTM_Model.num_Instances))

    def build_graph(self):
        def choose_device(op, device):
            if op.type.startswith('Variable'):
                device = self.center_device
            return device

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                feature_splits = tf.split(self.batch[0], self.num_gpus, name="feature_splits")
                label_splits = tf.split(self.batch[1], self.num_gpus, name="label_splits")
                mask_splits = tf.split(self.batch[2], self.num_gpus, name="mask_splits")
                seq_len = tf.split(self.batch[3], self.num_gpus, name="seq_len")

            frm_num_sum = tf.reduce_sum(mask_splits)
            loss_sum = tf.convert_to_tensor(0.0)
            frm_acc_sum = tf.convert_to_tensor(0.0)
            self.learning_rate = tf.convert_to_tensor(self.args.lr)

        # optimizer
        if self.is_training:
            with tf.name_scope("optimizer"):
                if self.args.optimizer == "adam":
                    logging.info("Using ADAM as optimizer")
                    self.learning_rate = warmup_exponential_decay(self.global_step,
                                                                  warmup_steps=self.args.warmup_steps,
                                                                  peak=self.args.peak,
                                                                  decay_rate=0.5,
                                                                  decay_steps=self.args.decay_steps)
                    optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                                       beta1=0.9,
                                                       beta2=0.98,
                                                       epsilon=1e-9,
                                                       name=self.args.optimizer)
                else:
                    logging.info("Using SGD as optimizer")
                    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate,
                                                                  name=self.args.optimizer)

        tower_grads = []
        with tf.variable_scope('model', reuse=bool(LSTM_Model.num_Instances), initializer=tf.uniform_unit_scaling_initializer()):
            # the outer scope is necessary for the where the reuse scope need to be limited whthin
            for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
                # Computation relevant to frame accuracy and loss
                with tf.device(lambda op: choose_device(op, name_gpu)):
                    # Construct lstm cell
                    multi_cell = self.build_lstm_cell()
                    # Construct output affine layers
                    w, b = self.build_affine_layer()

                    lstm_output = self.lstm_forward(multi_cell,
                                                    feature_splits[id_gpu],
                                                    seq_len)
                    logits = self.affine_forward(lstm_output, w=w, b=b)

                    # Accuracy
                    with tf.name_scope("label_accuracy"):
                        correct = tf.nn.in_top_k(logits, tf.reshape(label_splits[id_gpu], [-1]), 1)
                        correct = tf.multiply(tf.cast(correct, tf.float32), tf.reshape(mask_splits[id_gpu], [-1]))
                        frm_acc = tf.reduce_sum(correct)
                    # Cross entropy loss
                    with tf.name_scope("CE_loss"):
                        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.reshape(label_splits[id_gpu], [-1]), logits=logits)
                        cross_entropy = tf.multiply(cross_entropy, tf.reshape(mask_splits[id_gpu], [-1]))
                        cross_entropy_loss = tf.reduce_sum(cross_entropy)
                        loss = cross_entropy_loss

                    tf.get_variable_scope().reuse_variables()  # all the tf.get_variable will reuse variable

                    # Calculate L2 penalty.
                    l2_penalty = 0
                    for v in tf.trainable_variables():
                        if 'biase' not in v.name:
                            l2_penalty += tf.nn.l2_loss(v)
                    loss += self.args.lamda_l2 * l2_penalty

                    if self.is_training:
                        with tf.name_scope("gradients"):
                            gradients = optimizer.compute_gradients(loss)
                        tower_grads.append(gradients)

                with tf.device(self.center_device):
                    frm_acc_sum += frm_acc
                    loss_sum += loss

                logging.info('\tbuild model on {} succesfully!'.format(name_gpu))

        self.list_run = [loss_sum, frm_acc_sum, frm_num_sum]

        # merge gradients, update current model
        if self.is_training:
            with tf.device(self.center_device):
                # computation relevant to gradient
                averaged_grads = grads.average_gradients(tower_grads)
                handled_grads = grads.handle_gradients(averaged_grads, self.args)
                # with tf.variable_scope('adam', reuse=False):
                op_optimize = optimizer.apply_gradients(handled_grads)

                self.list_run.append(op_optimize)

                if self.args.is_debug:
                    with tf.name_scope("summaries"):
                        tf.summary.scalar('loss', loss_sum/frm_num_sum)
                        tf.summary.scalar('frame_accuracy', frm_acc_sum/frm_num_sum)
                        tf.summary.scalar('learning_rate', self.learning_rate)
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
                                               forget_bias=self.forget_bias)
            list_cells = [fentch_cell() for _ in range(self.args.num_lstm_layers)]
            # Add dropout
            if self.is_training and self.args.keep_prob < 1:
                list_cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.args.keep_prob)
                              for cell in list_cells]
            # multiple lstm layers
            multi_cell = tf.contrib.rnn.MultiRNNCell(list_cells)

        return multi_cell

    def lstm_forward(self, cell, inputs, sequence_length):
        # we have finishing building the model, the tf.nn.dynamic_rnn is a function, and not create new variable, so no need of variable scope
        # the variable created in `tf.nn.dynamic_rnn`, not in cell
        with tf.variable_scope("lstm"):
            lstm_output, _ = tf.nn.dynamic_rnn(cell,
                                               inputs,
                                               sequence_length=sequence_length,
                                               dtype=tf.float32)
        return lstm_output

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


def layer_normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    """Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta

    return outputs


if __name__ == '__main__':
    from dataProcessing.kaldiModel import KaldiModel, build_kaldi_lstm_layers, build_kaldi_output_affine
    from configs.arguments import args
    from dataProcessing import tfRecoderData
    import os

    os.chdir('/mnt/lustre/xushuang2/easton/projects/mix_model_2.0')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    logging.info('CUDA_VISIBLE_DEVICES : {}'.format(args.gpus))

    logging.info('args.dim_input : {}'.format(args.dim_input))

    dataReader_train = tfRecoderData.TFRecordReader(args.dir_train_data, args=args)
    dataReader_dev = tfRecoderData.TFRecordReader(args.dir_dev_data, args=args)

    seq_features, seq_labels = dataReader_train.create_seq_tensor(is_train=False)
    batch_train = dataReader_train.fentch_batch_with_TFbuckets([seq_features, seq_labels], args=args)

    seq_features, seq_labels = dataReader_dev.create_seq_tensor(is_train=False)
    batch_dev = dataReader_dev.fentch_batch_with_TFbuckets([seq_features, seq_labels], args=args)

    tensor_global_step = tf.train.get_or_create_global_step()

    graph_train = LSTM_Model(batch_train, tensor_global_step, True, args)
    logging.info('build training graph successfully!')
    graph_dev = LSTM_Model(batch_dev, tensor_global_step, False, args)
    logging.info('build dev graph successfully!')

    writer = tf.summary.FileWriter(os.path.join(args.model_dir, 'log'), graph=tf.get_default_graph())
    writer.close()
    sys.exit()

    if args.is_debug:
        list_ops = [op.name+' '+op.device for op in tf.get_default_graph().get_operations()]
        list_variables_and_devices = [op.name+' '+op.device for op in tf.get_default_graph().get_operations() if op.type.startswith('Variable')]
        logging.info('\n'.join(list_variables_and_devices))

    list_kaldi_layers = []
    list_kaldi_layers = build_kaldi_lstm_layers(list_kaldi_layers, args.num_lstm_layers, args.dim_input, args.num_projs)
    list_kaldi_layers = build_kaldi_output_affine(list_kaldi_layers)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.train.MonitoredTrainingSession(config=config) as sess:
        kaldi_model = KaldiModel(list_kaldi_layers)
        kaldi_model.loadModel(sess=sess, model_path=args.model_init)
