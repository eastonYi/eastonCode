#!/usr/bin/python
# coding=utf-8

import numpy as np
import logging
import sys
import struct
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class KaldiModel(object):
    def __init__(self, model):
        # lstm
        self.cell_dim_ = 100
        self.clip_gradient_ = 10
        self.lstm_max_norm_ = 0

        # affine
        self.learn_rate_coef_ = 1
        self.bias_learn_rate_coef_ = 1
        self.affine_max_norm_ = 0

        self.model = model

    # Token IO---------------------------------------------------------
    def ReadToken(self, f):
        tok = b''
        ch = f.read(1)
        while ch != b' ':
            tok = tok + ch
            ch = f.read(1)
        return tok.decode('utf-8')

    def WriteToken(self, f, token):

        _len = len(token)
        f.write(struct.pack("<%dss" % _len, token.encode('ascii'), b' '))
        return

    def ExpectToken(self, f, token):
        tk = self.ReadToken(f)
        if token == tk:
            return
        else:
            raise Exception("Expect Token %s with %s" % (token, tk))
        return

    # Base Type IO---------------------------------------------------------
    def ReadInt(self, f):
        l, value = struct.unpack('<bi', f.read(5))
        return value

    def ReadFloat(self, f):
        l, value = struct.unpack('<bf', f.read(5))
        return value

    def WriteInt(self, f, value):
        f.write(struct.pack('<bi', 4, value))
        return

    def WriteFloat(self, f, value):
        f.write(struct.pack('<bf', 4, value))
        return

    def ExpectInt(self, f, value):
        va = self.ReadInt(f)
        if value == va:
            return
        else:
            raise Exception("Expect Int %d with %d" % (value, va))

    # Matrix Vec IO---------------------------------------------------------
    def ReadMatrix(self, f, t=False):
        self.ExpectToken(f, 'FM')
        _rows = self.ReadInt(f)
        _cols = self.ReadInt(f)
        _py = np.frombuffer(f.read(_rows * _cols * 4), dtype=np.float32)
        _py = np.reshape(_py, [_rows, _cols])
        if t:
            _py = np.transpose(_py)

        return _py

    def ReadVector(self, f):
        self.ExpectToken(f, 'FV')
        _size = self.ReadInt(f)
        _py = np.frombuffer(f.read(_size * 4), dtype=np.float32)
        return _py

    def WriteMatrix(self, f, mat, t):
        if t:
            mat = np.transpose(mat)
        _rows, _cols = np.shape(mat)
        self.WriteToken(f, 'FM')
        self.WriteInt(f, _rows)
        self.WriteInt(f, _cols)
        f.write(mat.copy(order='C'))
        return

    def WriteVec(self, f, vec):
        _size, = np.shape(vec)
        self.WriteToken(f, 'FV')
        self.WriteInt(f, _size)
        f.write(vec.copy(order='C'))
        return

    # Layer IO---------------------------------------------------------
    def ReadLayer(self, f, sess, layer):
        self.ExpectToken(f, layer[0])
        self.ExpectInt(f, layer[1]) # dim_output
        self.ExpectInt(f, layer[2]) # dim_input
        if layer[0] == '<AffineTransform>':
            self.ExpectToken(f, '<LearnRateCoef>')
            self.learn_rate_coef_ = self.ReadFloat(f)
            self.ExpectToken(f, '<BiasLearnRateCoef>')
            self.bias_learn_rate_coef_ = self.ReadFloat(f)
            self.ExpectToken(f, '<MaxNorm>')
            self.affine_max_norm_ = self.ReadFloat(f)

            weights = self.ReadMatrix(f, True)
            biases = self.ReadVector(f)
            layer[3].load(weights, sess)
            layer[4].load(biases, sess)

        elif layer[0] == '<LstmProjectedStreams>':
            if len(layer) != 5:
                raise Exception('later num:{}'.format(layer))
            self.ExpectToken(f, '<CellDim>')
            self.cell_dim_ = self.ReadInt(f)
            self.ExpectToken(f, '<ClipGradient>')
            self.clip_gradient_ = self.ReadFloat(f)
            self.ExpectToken(f, '<MaxNorm>')
            self.lstm_max_norm_ = self.ReadFloat(f)

            # lstm_cell/weights
            w_gifo_x = self.ReadMatrix(f, True)
            w_gifo_r = self.ReadMatrix(f, True)
            w_g_x, w_i_x, w_f_x, w_o_x = np.split(w_gifo_x, 4, 1)
            w_g_r, w_i_r, w_f_r, w_o_r = np.split(w_gifo_r, 4, 1)
            w_igfo_x = np.concatenate([w_i_x, w_g_x, w_f_x, w_o_x], 1)
            w_igfo_r = np.concatenate([w_i_r, w_g_r, w_f_r, w_o_r], 1)
            layer[3].load(np.concatenate([w_igfo_x, w_igfo_r], 0), sess)

            # lstm_cell/biases
            bias = self.ReadVector(f)
            bias = np.expand_dims(bias, 0)
            bias_g, bias_i, bias_f, bias_o = np.split(bias, 4, 1)
            layer[4].load(np.squeeze(np.concatenate([bias_i, bias_g, bias_f, bias_o], 1), 0), sess)

            # must read to pop the data
            peephole_i = self.ReadVector(f)
            peephole_f = self.ReadVector(f)
            peephole_o = self.ReadVector(f)
            w_r_m = self.ReadMatrix(f, True)

        elif layer[0] == '<LayerNorm>':
            beta = self.ReadVector(f)
            gamma = self.ReadVector(f)
            layer[3].load(beta, sess)
            layer[4].load(gamma, sess)

        elif layer[0] == '<Sigmoid>':
            pass
        elif layer[0] == '<Relu>':
            pass
        elif layer[0] == '<Softmax>':
            pass
        else:
            raise Exception("No Such Layer %s" % (layer[0]))

        return

    def WriteLayer(self, f, sess, layer):
        self.WriteToken(f, layer[0])
        self.WriteInt(f, layer[1]) # dim_output
        self.WriteInt(f, layer[2]) # dim_input
        if layer[0] == '<AffineTransform>':
            assert len(layer) == 5
            self.WriteToken(f, '<LearnRateCoef>')
            self.WriteFloat(f, self.learn_rate_coef_)
            self.WriteToken(f, '<BiasLearnRateCoef>')
            self.WriteFloat(f, self.bias_learn_rate_coef_)
            self.WriteToken(f, '<MaxNorm>')
            self.WriteFloat(f, self.affine_max_norm_)

            _weights, _biases = sess.run(layer[3:])
            self.WriteMatrix(f, _weights, True)
            self.WriteVec(f, _biases)

        elif layer[0] == '<LstmProjectedStreams>':
            assert len(layer) == 5
            self.WriteToken(f, '<CellDim>')
            self.WriteInt(f, self.cell_dim_)
            self.WriteToken(f, '<ClipGradient>')
            self.WriteFloat(f, self.clip_gradient_)
            self.WriteToken(f, '<MaxNorm>')
            self.WriteFloat(f, self.lstm_max_norm_)

            dim_output, dim_input = layer[1:3]
            # assert dim_output == dim_input
            weights, biases = sess.run(layer[3:])
            if (dim_input+dim_output) != weights.shape[0]:
                raise Exception('dim_input:{}; dim_output:{}; weights.shape[0]:{}'.format(dim_input, dim_output, weights.shape[0]))

            # w_gifo_x  w_gifo_r
            w_igfo_x, w_igfo_r = np.split(weights, [dim_input], 0)
            w_i_x, w_g_x, w_f_x, w_o_x = np.split(w_igfo_x, 4, 1)
            w_i_r, w_g_r, w_f_r, w_o_r = np.split(w_igfo_r, 4, 1)
            w_gifo_x = np.concatenate([w_g_x, w_i_x, w_f_x, w_o_x], 1)
            w_gifo_r = np.concatenate([w_g_r, w_i_r, w_f_r, w_o_r], 1)
            self.WriteMatrix(f, w_gifo_x, True)
            self.WriteMatrix(f, w_gifo_r, True)

            # bias
            bias_i, bias_g, bias_f, bias_o = np.split(np.expand_dims(biases, 0), 4, 1)
            bias = np.concatenate([bias_g, bias_i, bias_f, bias_o], 1)
            self.WriteVec(f, np.squeeze(bias, 0))

            # set peehole to 0
            peephole = np.zeros([dim_output], dtype=np.float32)
            self.WriteVec(f, peephole)
            self.WriteVec(f, peephole)
            self.WriteVec(f, peephole)

            # set weights_projection to eye
            w_r_m = np.eye(dim_output, dtype=np.float32)
            self.WriteMatrix(f, w_r_m, True)

        elif layer[0] == '<LayerNorm>':
            assert len(layer) == 5
            _beta, _gamma = sess.run(layer[3:])
            self.WriteVec(f, _beta)
            self.WriteVec(f, _gamma)

        elif layer[0] == '<Sigmoid>':
            pass
        elif layer[0] == '<Relu>':
            pass
        elif layer[0] == '<Softmax>':
            pass
        else:
            raise Exception("No Such Layer %s" % (layer[0]))

        return

    # File IO---------------------------------------------------------
    def loadModel(self, sess, model_path):
        with open(model_path, 'rb') as f:
            head = struct.unpack('<cc', f.read(2))
            if head[0] != b'\0' or head[1] != b'B':
                raise Exception("Error Format %s" % model_path)

            self.ExpectToken(f, '<Nnet>')
            for layer in self.model:
                self.ReadLayer(f, sess, layer)
            self.ExpectToken(f, '</Nnet>')
        logging.info('load kaldi model {} succesfully!'.format(model_path))
        return

    def saveModel(self, sess, model_path, args):
        self.cell_dim_ = args.size_cell_hidden
        self.lstm_max_norm_ = args.max_norm
        self.clip_gradient_ = args.grad_clip_value

        with open(model_path, 'wb') as f:
            f.write(struct.pack('<cc', b'\0', b'B'))

            self.WriteToken(f, '<Nnet>')
            for layer in self.model:
                self.WriteLayer(f, sess, layer)
            self.WriteToken(f, '</Nnet>')
        logging.info('save kaldi model {} succesfully!'.format(model_path))
        return


#======================================
#compound layer methods
#======================================
def build_kaldi_lstm_layers(model, num_layers, dim_input, dim_output):
    """
    build correspond kaldi model after build lstm TF model
    """
    def get_layers_vars(num_layers):
        '''
        [[<tf.Variable 'model/lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/kernel:0' shape=(929, 3200) dtype=float32_ref>,
        <tf.Variable 'model/lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/bias:0' shape=(3200,) dtype=float32_ref>],
        [<tf.Variable 'model/lstm/rnn/multi_rnn_cell/cell_1/cudnn_compatible_lstm_cell/kernel:0' shape=(1600, 3200) dtype=float32_ref>,
        <tf.Variable 'model/lstm/rnn/multi_rnn_cell/cell_1/cudnn_compatible_lstm_cell/bias:0' shape=(3200,) dtype=float32_ref>],
        [<tf.Variable 'model/lstm/rnn/multi_rnn_cell/cell_2/cudnn_compatible_lstm_cell/kernel:0' shape=(1600, 3200) dtype=float32_ref>,
        <tf.Variable 'model/lstm/rnn/multi_rnn_cell/cell_2/cudnn_compatible_lstm_cell/bias:0' shape=(3200,) dtype=float32_ref>]]
        '''
        list_lstm_variables = [v for v in tf.trainable_variables() if "lstm" in v.name]
        interval = len(list_lstm_variables)//num_layers
        lstm_layers = []
        for i in range(0, num_layers):
            lstm_layers.append(list_lstm_variables[i*interval:(i+1)*interval])

        return lstm_layers

    tf_lstm_layers = get_layers_vars(num_layers)

    for i in range(num_layers):
        if i == 0:
            lstm_layer = kaldi_lstm_layer(tf_lstm_layers[i], dim_input, dim_output)
        else:
            lstm_layer = kaldi_lstm_layer(tf_lstm_layers[i], dim_output, dim_output)
        model.append(lstm_layer)

    return model


def build_kaldi_layernorm(model):
    layer_norm_variables = [v for v in tf.trainable_variables() if "layer_norm" in v.name]
    num_layers = len(model)
    num_norm_layers = len(layer_norm_variables)//2
    if not num_norm_layers == num_layers-2: #the last layer is
        raise Exception('num_norm_layers: {}, num_layers:{}'.format(num_norm_layers, num_layers))
    layer_norm = kaldi_layernorm(layer_norm_variables, num_norm_layers)

    model_layernormed = []
    for layer, norm in zip(model, layer_norm):
        model_layernormed.extend([layer, norm])
    model_layernormed.extend(model[num_norm_layers:])

    return model_layernormed


def build_kaldi_output_affine(model):
    '''
    [<tf.Variable 'output_layer/output_linearity:0' shape=(512, 10217) dtype=float32_ref>,
     <tf.Variable 'output_layer/output_bias:0' shape=(10217,) dtype=float32_ref>]
    '''
    output_variables = [v for v in tf.trainable_variables() if "output_layer" in v.name]
    output_affine = kaldi_lstm_last_affine(output_variables[0], output_variables[1],
                                           output_variables[0].get_shape().as_list()[0],
                                           output_variables[0].get_shape().as_list()[1])
    model.extend(output_affine)

    return model


#======================================
#single layer methods
#======================================
def kaldi_lstm_layer(single_layer_params, dim_input, dim_output):

    lstm_layer = ['<LstmProjectedStreams>', dim_output, dim_input]
    for tensor in single_layer_params:
        lstm_layer.append(tensor)

    return lstm_layer


def kaldi_lstm_last_affine(weights, bias, dim_input, dim_output):
    model = []
    model.append(['<AffineTransform>', dim_output, dim_input, weights, bias])
    model.append(['<Softmax>', dim_output, dim_output])

    return model


def kaldi_layernorm(variables, num_norm_layers):
    """
    [<tf.Variable 'model/layer_norm/beta_0:0' shape=(500,) dtype=float32_ref>, <tf.Variable 'model/layer_norm/gamma_0:0' shape=(500,) dtype=float32_ref>,
     <tf.Variable 'model/layer_norm/beta_1:0' shape=(500,) dtype=float32_ref>, <tf.Variable 'model/layer_norm/gamma_1:0' shape=(500,) dtype=float32_ref>,
     <tf.Variable 'model/layer_norm/beta_2:0' shape=(500,) dtype=float32_ref>, <tf.Variable 'model/layer_norm/gamma_2:0' shape=(500,) dtype=float32_ref>]
    """
    variables = iter(variables)
    model = []
    for i in range(num_norm_layers):
        beta = next(variables)
        gamma = next(variables)
        dim_input = int(beta.get_shape()[0])
        model.append(['<LayerNorm>', dim_input, dim_input, beta, gamma])
    return model


if __name__ == '__main__':
    from argparse import ArgumentParser
    from tfModels.lstmModel_cuDNN import LSTM_Model
    from configs.arguments import args

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

    # def config():
    #     parser = ArgumentParser()
    #     parser.add_argument("--dir_model", type=str,
    #         default='/mnt/lustre/xushuang/easton/projects/mix_model_2.0/models/cudnn_model')
    #     parser.add_argument("--num_lstm_layers", type=int, default=5)
    #     parser.add_argument("--dim_input", type=int, default=87)
    #     parser.add_argument("--num_projs", type=int, default=512)
    #     return parser.parse_args()
    # args = config()
    tensor_global_step = tf.train.get_or_create_global_step()
    tensor_batch_train = tf.placeholder(dtype=tf.int32)
    tensor_batch_dev = tf.placeholder(dtype=tf.int32)

    graph_train = LSTM_Model(tensor_batch_train, tensor_global_step, True, args)
    graph_dev = LSTM_Model(tensor_batch_dev, tensor_global_step, False, args)

    model = []
    model = build_kaldi_lstm_layers(model, args.num_lstm_layers, args.dim_input, args.num_projs)
    model = build_kaldi_output_affine(model)
    model = build_kaldi_layernorm(model)
