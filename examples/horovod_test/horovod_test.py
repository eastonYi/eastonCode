import tensorflow as tf
import horovod.tensorflow as hvd
import logging
import sys

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

hvd.init()

a = tf.Variable(10)
if hvd.rank() == 0:
    op = tf.assign(a, 0)
else:
    op = tf.assign(a, 1)

"""
the broadcast op MUST after the variables that need to be sync!
"""
broadcast = hvd.broadcast_global_variables(0)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())
config.allow_soft_placement = True
config.log_device_placement = False
with tf.train.MonitoredTrainingSession(config=config) as sess:
    if hvd.rank() == 0:
        sess.run(op)
        logging.info('here is rank{}, {}'.format(hvd.rank(), sess.run(a)))
    else:
        sess.run(op)
        logging.info('here is rank{}, {} waiting...'.format(hvd.rank(), sess.run(a)))
    sess.run(broadcast)
    logging.info('After here is rank{}, {}'.format(hvd.rank(), sess.run(a)))
