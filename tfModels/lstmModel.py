import tensorflow as tf
import logging
import sys
from collections import namedtuple
from tensorflow.contrib.layers import fully_connected

from tfTools.gradientTools import average_gradients, handle_gradients
from tfModels.tools import warmup_exponential_decay, choose_device, lr_decay_with_warmup
from tfModels.layers import layer_normalize, build_cell, cell_forward


class LSTM_Model(object):
    num_Instances = 0
    num_Model = 0

    def __init__(self, tensor_global_step, is_train, args, batch=None, name='model'):
        # Initialize some parameters
        self.is_train = is_train
        self.num_gpus = args.num_gpus if is_train else 1
        self.list_gpu_devices = args.list_gpus
        self.center_device = "/cpu:0"
        self.learning_rate = None
        self.args = args
        self.batch = batch
        self.name = name
        self.build_input = self.build_tf_input if batch else self.build_pl_input

        self.list_pl = None

        self.global_step = tensor_global_step

        # Build graph
        self.list_run = list(self.build_graph() if is_train else self.build_infer_graph())

    def build_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_input()
        # create optimizer
        self.optimizer = self.build_optimizer()

        loss_step = []
        tower_grads = []
        # the outer scope is necessary for the where the reuse scope need to be limited whthin
        # or reuse=tf.get_variable_scope().reuse
        for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
            with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
                loss, gradients = self.build_single_graph(id_gpu, name_gpu, tensors_input)
                loss_step.append(loss)
                tower_grads.append(gradients)

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

        return loss, tensors_input.shape_batch, op_optimize

    def build_infer_graph(self):
        """
        reuse=True if build train models above
        reuse=False if in the inder file
        """
        # cerate input tensors in the cpu
        tensors_input = self.build_input()

        # the outer scope is necessary for the where the reuse scope need to be limited whthin
        # or reuse=tf.get_variable_scope().reuse
        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            loss, logits = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

        # TODO havn't checked
        infer = tf.nn.in_top_k(logits, tf.reshape(tensors_input.label_splits[0], [-1]), 1)

        return loss, tensors_input.shape_batch, infer

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

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        """
        be used for build infer model and the train model, conditioned on self.is_train
        """
        # build model in one device
        num_cell_units = self.args.model.num_cell_units
        cell_type = self.args.model.cell_type
        dropout = self.args.model.dropout
        forget_bias = self.args.model.forget_bias
        use_residual = self.args.model.use_residual

        hidden_output = tensors_input.feature_splits[id_gpu]
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            for i in range(self.args.model.num_lstm_layers):
                # build one layer: build block, connect block
                single_cell = build_cell(
                    num_units=num_cell_units,
                    num_layers=1,
                    is_train=self.is_train,
                    cell_type=cell_type,
                    dropout=dropout,
                    forget_bias=forget_bias,
                    use_residual=use_residual)
                hidden_output, _ = cell_forward(
                    cell=single_cell,
                    inputs=hidden_output,
                    index_layer=i)
                hidden_output = fully_connected(
                    inputs=hidden_output,
                    num_outputs=num_cell_units,
                    activation_fn=tf.nn.tanh,
                    scope='wx_b'+str(i))
                if self.args.model.use_layernorm:
                    hidden_output = layer_normalize(hidden_output, i)

            logits = fully_connected(inputs=hidden_output,
                                     num_outputs=self.args.dim_output,
                                     activation_fn=tf.identity,
                                     scope='fully_connected')

            # Accuracy
            with tf.name_scope("label_accuracy"):
                correct = tf.nn.in_top_k(logits, tf.reshape(tensors_input.label_splits[id_gpu], [-1]), 1)
                correct = tf.multiply(tf.cast(correct, tf.float32), tf.reshape(tensors_input.mask_splits[id_gpu], [-1]))
                label_accuracy = tf.reduce_sum(correct)
            # Cross entropy loss
            with tf.name_scope("CE_loss"):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.reshape(tensors_input.label_splits[id_gpu], [-1]),
                    logits=logits)
                cross_entropy = tf.multiply(cross_entropy, tf.reshape(tensors_input.mask_splits[id_gpu], [-1]))
                cross_entropy_loss = tf.reduce_sum(cross_entropy)
                loss = cross_entropy_loss

            if self.is_train:
                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Instances))

        return loss, gradients if self.is_train else logits

    def build_optimizer(self):
        if self.args.learning_rate:
            self.learning_rate = lr_decay_with_warmup(
                self.global_step,
                warmup_steps=self.args.warmup_steps,
                hidden_units=self.args.model.encoder.num_cell_units)
        else:
            self.learning_rate = warmup_exponential_decay(
                self.global_step,
                warmup_steps=self.args.warmup_steps,
                peak=self.args.peak,
                decay_rate=0.5,
                decay_steps=self.args.decay_steps)

        with tf.name_scope("optimizer"):
            if self.args.optimizer == "adam":
                logging.info("Using ADAM as optimizer")

                optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                                   beta1=0.9,
                                                   beta2=0.98,
                                                   epsilon=1e-9,
                                                   name=self.args.optimizer)
            else:
                logging.info("Using SGD as optimizer")
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate,
                                                              name=self.args.optimizer)
        return optimizer

    @property
    def variables(self):
        '''get a list of the models's variables'''
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.name + '/')

        if hasattr(self, 'wrapped'):
            #pylint: disable=E1101
            variables += self.wrapped.variables

        return variables


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
