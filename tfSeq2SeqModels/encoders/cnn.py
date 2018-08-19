import tensorflow as tf
from functools import reduce
import numpy as np

from .encoder import Encoder

from tfModels.layers import conv_layer
# from tfTools.tfAudioTools import down_sample


class CNN(Encoder):
    def encode(self, features, len_feas):
        # num_conv_layers = self.args.model.encoder.num_conv_layers
        # num_filter = [32, 32, 32, 32, 96]
        # kernal = [(41,11), (21,11), (21,11), (21,11), (21,11)]
        # stride = [(2,2), (2,1), (1,1), (1,1), (1,1)]
        num_filter = [32, 32, 32, 96]
        kernal = [(41,11), (21,11), (21,11), (21,11)]
        stride = [(2,2), (2,1), (1,1), (1,1)]

        hidden_output = features
        size_batch = tf.shape(features)[0]
        len_batch = tf.shape(features)[1]
        size_feat = self.args.data.dim_input

        conv_output = tf.expand_dims(hidden_output, -1)

        for i in range(len(kernal)):
            conv_output = conv_layer(
                conv_output,
                filter_num=num_filter[i],
                kernel=kernal[i],
                stride=stride[i],
                scope='conv_'+str(i))
            # conv_output = down_sample(conv_output, rate=2)
        len_shrink = reduce((lambda x, y: x * y), (i[0] for i in stride))
        feat_shrink = reduce((lambda x, y: x * y), (i[1] for i in stride))

        len_feas = tf.cast(tf.ceil(tf.cast(len_feas,tf.float32)/len_shrink),
                           tf.int32)

        len_batch = tf.cast(tf.ceil(tf.cast(len_batch,tf.float32)/len_shrink),
                           tf.int32)

        size_feat = int(np.ceil(size_feat/feat_shrink))*num_filter[-1]
        hidden_output = tf.reshape(conv_output, [size_batch, len_batch, size_feat])

        return hidden_output, len_feas
