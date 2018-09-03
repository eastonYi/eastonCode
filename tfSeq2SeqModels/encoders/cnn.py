import tensorflow as tf
from functools import reduce

from .encoder import Encoder

from tfModels.layers import conv_layer
# from tfTools.tfAudioTools import down_sample


class CNN(Encoder):
    def encode(self, features, len_sequence):
        """
        features: [size_batch, size_length, size_feat]
        len_sequence: 1d tensor. the length of the corresponding batch
        size_length: the 1-th size of the features
        """
        num_filter = [32, 32, 32, 96]
        kernal = [(41,11), (21,11), (21,11), (21,11)]
        stride = [(2,2), (2,1), (2,1), (1,1)]

        hidden_output = features
        size_feat = self.args.data.dim_input

        conv_output = tf.expand_dims(hidden_output, -1)

        for i in range(len(kernal)):
            conv_output, len_sequence, size_feat = conv_layer(
                inputs=conv_output,
                size_feat=size_feat,
                num_filter=num_filter[i],
                kernel=kernal[i],
                stride=stride[i],
                scope='conv_'+str(i))

        return hidden_output, len_sequence
