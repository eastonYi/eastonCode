import time

def config():
    ...
args = config()

# Build Graph
graph_train = Model("train")
graph_dev = Model("dev")

# Create batch iterator
dataReader_train = DataReader('train')
dataReader_dev = DataReader('dev')

# Recording
time_start = time.time()
loss_dev_best = 99999
path_model_best = args.path_model_init

# Start session of training
config_sess =  tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
with tf.Session(config=config_sess) as sess:
    saver = tf.train.Saver(max_to_keep=10)

    # variable init
    try:
        model.saver.restore(sess, tf.train.latest_checkpoint(path_model_init))
        logging.info("Reading model parameters from %s" % path_model_init)
    except Exception:
        # Initialize all variables.
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        logging.info("Created model with fresh parameters.")

    # start epoch
    for cur_epoch in args.num_epochs:

        # batch iterring
        logging.info("\tstart {}th epoch training ...".format(cur_epoch))
        for batch in dataReader_train.fentch_batch():
            Model.forward(sess, graph_train, batch, lr, args)

        logging.info("\tfinish {}th epoch training,  begining to evaluate using dev data...".format(cur_epoch))
        for batch in dataReader_dev.fentch_batch():
            Model.forward(sess, graph_dev, batch, args)

        Model.loss
        logging.info("Finish iter {}, used {}s ".format(cur_epoch, time.time()-time_start))
