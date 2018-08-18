import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def np2tfrecord():
    """
    we treat the array as a string: convert the array to string first
    it is not efficient afterall
    """
    writer = tf.python_io.TFRecordWriter('here')

    fea = np.ones([10, 3], dtype=np.float32)
    label = np.array([1,1,2,4,5], dtype=np.int32)
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'feat': _bytes_feature(fea.tostring()),
                'label': _bytes_feature(label.tostring())
    }))
    writer.write(example.SerializeToString())

def tfrecord2np():

    filename_queue = tf.train.string_input_producer(
            ['here'], num_epochs=1)
    reader_tfRecord = tf.TFRecordReader()
    _, serialized_example = reader_tfRecord.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={'feat': tf.FixedLenFeature([],tf.string),
                  'label': tf.FixedLenFeature([],tf.string)}
    )

    feature = tf.decode_raw(features['feat'], tf.float32)
    feature = tf.reshape(feature, [-1, 3])

    label = tf.decode_raw(features['label'], tf.int32)


    with tf.train.MonitoredTrainingSession() as sess:
        print('here')
        # print(example)
        print(sess.run([feature, label]))


if __name__ == '__main__':
    tfrecord2np()
