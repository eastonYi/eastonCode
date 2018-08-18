import logging
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, LayerNormBasicLSTMCell, DropoutWrapper, ResidualWrapper, MultiRNNCell, OutputProjectionWrapper


def ff_hidden(inputs, hidden_size, output_size, activation, use_bias=True, reuse=None, name=None):
    with tf.variable_scope(name, "ff_hidden", reuse=reuse):
        hidden_outputs = dense(inputs, hidden_size, activation, use_bias)
        outputs = dense(hidden_outputs, output_size, tf.identity, use_bias)
        return outputs


def dense(inputs,
          output_size,
          activation=tf.identity,
          use_bias=True,
          kernel=None,
          reuse=None,
          name=None):
    argcount = activation.__code__.co_argcount
    if activation.__defaults__:
        argcount -= len(activation.__defaults__)
    assert argcount in (1, 2)
    with tf.variable_scope(name, "dense", reuse=reuse):
        if argcount == 1:
            input_size = inputs.get_shape().as_list()[-1]
            inputs_shape = tf.unstack(tf.shape(inputs))
            inputs = tf.reshape(inputs, [-1, input_size])
            if kernel is not None:
                assert kernel.get_shape().as_list()[0] == output_size
                w = kernel
            else:
                with tf.variable_scope(tf.get_variable_scope()):
                    w = tf.get_variable("kernel", [output_size, input_size])
            outputs = tf.matmul(inputs, w, transpose_b=True)
            if use_bias:
                b = tf.get_variable("bias", [output_size], initializer=tf.zeros_initializer)
                outputs += b
            outputs = activation(outputs)
            return tf.reshape(outputs, inputs_shape[:-1] + [output_size])
        else:
            arg1 = dense(inputs, output_size, tf.identity, use_bias, name='arg1')
            arg2 = dense(inputs, output_size, tf.identity, use_bias, name='arg2')
            return activation(arg1, arg2)


def residual(inputs, outputs, dropout_rate, index_layer):
    """Residual connection.

    Args:
        inputs: A Tensor.
        outputs: A Tensor.
        dropout_rate: A float range from [0, 1).

    Returns:
        A Tensor.
    """
    outputs = inputs + tf.nn.dropout(outputs, 1 - dropout_rate)
    outputs = layer_normalize(outputs, index_layer)
    return outputs


def layer_normalize(inputs, index_layer, epsilon=1e-8):
    with tf.variable_scope("layer_norm"):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta_"+str(index_layer), params_shape, dtype=tf.float32)
        gamma = tf.get_variable("gamma_"+str(index_layer), params_shape, dtype=tf.float32)
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta

    return outputs


def embedding(x, vocab_size, dense_size, name=None, reuse=None, multiplier=1.0):
    """Embed x of type int64 into dense vectors."""
    with tf.variable_scope(
            name, default_name="embedding", values=[x], reuse=reuse):
        embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
        output = tf.gather(embedding_var, x)
        if multiplier != 1.0:
            output *= multiplier
        return output


def single_cell(num_units, is_train, cell_type,
                dropout=0.0, forget_bias=0.0, dim_project=None):
    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    dropout = dropout if is_train else 0.0

    # Cell Type
    if cell_type == "lstm":
        single_cell = BasicLSTMCell(num_units, forget_bias=forget_bias)
    elif cell_type == "cudnn_lstm":
        single_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units)
    elif cell_type == "gru":
        single_cell = GRUCell(num_units)
    elif cell_type == "layer_norm_lstm":
        single_cell = LayerNormBasicLSTMCell(num_units,
                                             forget_bias=forget_bias,
                                             layer_norm=True)
    else:
        raise ValueError("Unknown unit type %s!" % cell_type)

    if dim_project:
        single_cell = OutputProjectionWrapper(
            cell=single_cell,
            output_size=dim_project)

    if dropout > 0.0:
        single_cell = DropoutWrapper(cell=single_cell,
                                     input_keep_prob=(1.0 - dropout))

    return single_cell


def build_cell(num_units, num_layers, is_train, cell_type,
               dropout=0.0, forget_bias=0.0, use_residual=True, dim_project=None):
    with tf.name_scope(cell_type):
        list_cell = [single_cell(
            num_units=num_units,
            is_train=is_train,
            cell_type=cell_type,
            dropout=dropout,
            forget_bias=forget_bias,
            dim_project=dim_project) for _ in range(num_layers)]
    # Residual
    if use_residual:
        for c in range(1, len(list_cell)):
            list_cell[c] = ResidualWrapper(list_cell[c])

    return MultiRNNCell(list_cell) if num_layers > 1 else list_cell[0]


def cell_forward(cell, inputs, index_layer=0, initial_state=None):
    # the variable created in `tf.nn.dynamic_rnn`, not in cell
    with tf.variable_scope("lstm"):
        # print('index_layer: ', index_layer, 'inputs.get_shape(): ', inputs.get_shape())
        lstm_output, state = tf.nn.dynamic_rnn(cell,
                                               inputs,
                                               initial_state=initial_state,
                                               scope='cell_'+str(index_layer),
                                               dtype=tf.float32)
    return lstm_output, state


def conv_layer(inputs, filter_num, kernel, stride,
               padding='same', use_relu=True, scope='conv', norm_type="ln"):
    """
    if cond1d, set stride=(1,1)
    Demo:
        from tfModels.layers import conv_layer

        with tf.variable_scope('test', reuse=False):
            inputs = tf.ones([10, 20, 40, 1])
            conv_output = conv_layer(
                        inputs,
                        filter_num=8,
                        kernel=(41,11),
                        stride=(2,2))
        sess.run(tf.shape(conv_output))
        # array([10, 10, 20,  8], dtype=int32)
    """
    with tf.variable_scope(scope):
        net = tf.layers.conv2d(inputs, filter_num, kernel, stride, padding, name="conv")
        if norm_type == "bn":
            net = tf.layers.batch_normalization(net, name="bn")
        elif norm_type == "ln":
            net = tf.contrib.layers.layer_norm(net)
        output = tf.nn.relu(net) if use_relu else net

    return output


class SimpleModel(object):
    def __init__(self, reuse, scope=None):
        self.reuse = reuse
        self.scope = scope or type(self).__name__
        self.build(self.scope, self.reuse)

    def build(self, scope, reuse):
        '''
        # create trainable variables

        with tf.variable_scope(scope, reuse=reuse):
            self.variable =
        return
        '''
        raise NotImplementedError('a model must have a build function to create variables')


    def forward(self, inputs_batch, labels, device=None, scope=None):
        '''
        with tf.name_scope(scope):
            with tf.device(device=device if not device else '/cpu:0'):
                ...
            return loss
            '''
        raise NotImplementedError('a model must have a forward function to compute loss')


    @property
    def params(self):
        return tf.trainable_variables(self.scope)


class AffineLayer(SimpleModel):
    '''
    '''
    def __init__(self, dim_input, dim_output, reuse):
        super().__init__(reuse=reuse)
        self.dim_input = dim_input
        self.dim_output = dim_output

    def build(self):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.w = tf.get_variable("output_weight", [self.dim_input, self.dim_output], dtype=tf.float32)
            self.b = tf.get_variable("output_bias", [self.dim_output], dtype=tf.float32)

        return

    def forward(self, input_tensor, scope=None):
        with tf.name_scope(scope):
            return tf.matmul(tf.reshape(input_tensor, [-1, self.dim_input]), self.w) + self.b


class LSTMLayer(SimpleModel):
    def __init__(self, size_cell_hidden, num_lstm_layers, is_training, keep_prob=1.0):
        self.size_cell_hidden = size_cell_hidden
        self.num_lstm_layers = num_lstm_layers
        self.is_training = is_training
        self.keep_prob = keep_prob
        self.multi_cell = self.build_lstm_cell()


    def build_lstm_cell(self):
        # because we are building models, creating variables in the same time , so we have variable scope
        with tf.name_scope("lstm"):
            def fentch_cell():
                return tf.contrib.rnn.LSTMCell(self.size_cell_hidden,
                                               use_peepholes=True,
                                               cell_clip=50.0)
            list_cells = [fentch_cell() for _ in range(self.num_lstm_layers)]
            # Add dropout
            if self.is_training and self.keep_prob < 1:
                list_cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                              for cell in list_cells]
            # multiple lstm layers
            multi_cell = tf.contrib.rnn.MultiRNNCell(list_cells)

        return multi_cell

    def forward(self, inputs):
        # we have finishing building the model, the tf.nn.dynamic_rnn is a function, and not create new variable, so no need of variable scope
        # the variable created in `tf.nn.dynamic_rnn`, not in cell
        with tf.variable_scope("lstm"):
            lstm_output, _ = tf.nn.dynamic_rnn(cell=self.multi_cell,
                                               inputs=inputs,
                                               dtype=tf.float32)
        return lstm_output


class CenterLoss(SimpleModel):
    '''

    loss, batch_center_input = model.forward()

    centerloss = CenterLoss(dim_center, num_class,1.0, False)

    batch_center_input_fake = tf.ones((szie_batch, dim_center))
    label_input_fake = tf.ones(size_batch)

    centerloss.forward(batch_center_input_fake, label_input_fake)
    '''
    def __init__(self, dim_center, num_center, lambda_c, reuse, scope=None):
        self.dim_center = dim_center
        self.num_center = num_center
        self.lambda_c = lambda_c
        super().__init__(reuse=reuse, scope=scope)


    def build(self, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            self.tf_centers = tf.get_variable('centers',
                                              shape=[self.num_center, self.dim_center],
                                              dtype=tf.float32,
                                              initializer=tf.constant_initializer(0),
                                              trainable=True)
        return

    def forward(self, inputs_batch, labels, device=None):

        with tf.device(device=device if not device else '/cpu:0'):
            assert labels.shape == inputs_batch.shape[0]
            centers_batch = tf.gather(self.tf_centers, labels)
            center_loss = tf.reduce_sum(((inputs_batch - centers_batch) ** 2) * self.lambda_c)

        return center_loss
