import tensorflow as tf
from simpleModelClass import SimpleModel

class MainModelClass(SimpleModel):

    def __init__(self, is_training, args):
        # Initialize some parameters
        self.is_training = is_training
        self.num_gpu = args.get_num_gpu()
        self.gpu_devices = args.get_gpu_list()
        self.default_device = "/cpu:0"
        self.num_streams = args.trn_streams if is_training else args.cv_streams
        self.batch_size = self.num_gpu * (args.trn_streams if is_training else args.cv_streams)
        self.scope = type(self).__name__

        # Return for session
        self.run_list = None
        self.placeholder_list = None

        # Build graph
        # no `reuse parameter` compared with simpleLayerClass, so no using of super()
        self.build(args, scope=self.scope)


    def build(self, args, scope):
        with tf.variabe_scope(scope=scope):
            # input Placeholders
            with tf.variable_scope("inputs"):
                feature_placeholder = tf.placeholder(tf.float32, [None, args.num_steps, args.dim_feature], name="input_feature")
                label_placeholder = tf.placeholder(tf.int32, [None, args.num_steps], name="input_label")
                mask_placeholder = tf.placeholder(tf.float32, [None, args.num_steps], name="input_mask")
            self.placeholder_list = [feature_placeholder, label_placeholder, mask_placeholder]

            # split input data alone batch axis to gpus
            with tf.variable_scope("splits"):
                feature_splits = tf.split(feature_placeholder, self.num_gpu, axis=0, name="feature_splits")
                label_splits = tf.split(label_placeholder, self.num_gpu, axis=0, name="label_splits")
                mask_splits = tf.split(mask_placeholder, self.num_gpu, axis=0, name="mask_splits")

            # Construct lstm cell
            cell = self.build_lstm_cell(args=args)
            # Construct output affine layers
            w, b = self.build_affine_layer(args=args)


    def forward(self, inputs_batch, labels, device=None):
        with tf.name_scope(self.scope, reuse=False):
            # optimizer when is training
            if self.is_training:
                logging.info("Using ADAM as optimizer")
                optimizer = tf.train.AdamOptimizer(lr_placeholder, name=args.optimizer)

            # initialization
            num_frame_sum = tf.reduce_sum(mask_placeholder)
            loss_sum = tf.convert_to_tensor(0.0)
            acc_sum = tf.convert_to_tensor(0.0)
            tower_grads = []
            # build each device's graph
            for gpu_id in range(self.num_gpu):
                # Computation relevant to frame accuracy and loss
                with tf.device(self.gpu_devices[gpu_id]):
                    lstm_output = self.lstm_forward(cell=cell, inputs=feature_splits[gpu_id], gpu_id=gpu_id)
                    logits = self.affine_forward(lstm_output, w=w, b=b, gpu_id=gpu_id)
                    # Cross entropy loss
                    with tf.name_scope("CE_loss"):
                        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.reshape(label_splits[gpu_id], [-1]), logits=logits)
                        cross_entropy = tf.multiply(cross_entropy, tf.reshape(mask_splits[gpu_id], [-1]))
                        cross_entropy_loss = tf.reduce_sum(cross_entropy, name="gpu%d_loss" % gpu_id)
                        loss = cross_entropy_loss
                        # if args.lamda_l2 > 0:
                            # loss += self.apply_l2_regularization(args.lamda_l2)
                        loss_sum += loss

                    # Accuracy
                    with tf.name_scope("label_accuracy"):
                        indicate_correct = tf.nn.in_top_k(logits, tf.reshape(label_splits[gpu_id], [-1]), 1)
                        indicate_correct_masked = tf.multiply(tf.cast(indicate_correct, tf.float32), tf.reshape(mask_splits[gpu_id], [-1]))
                        accuracy = tf.reduce_mean(indicate_correct_masked, name="gpu%d_label_acc" % gpu_id)

                    if self.is_training:
                        # Compute Gradients
                        gradients = optimizer.compute_gradients(loss)
                        tower_grads.append(gradients)

            self.run_list = [loss_sum, accuracy]
            # merge gradients, update current model
            if self.is_training:
                with tf.device(self.default_device):
                    # computation relevant to gradient
                    averaged_grads = grads.average_gradients(tower_grads)
                    handled_grads = grads.handle_gradients(averaged_grads, args)
                    backward = optimizer.apply_gradients(handled_grads)
                self.run_list.append(backward)

                if args.is_debug:
                    with tf.variable_scope("summaries"):
                        tf.summary.scalar('loss', loss_sum / frm_num_sum )
                        tf.summary.scalar('learning_rate', lr_placeholder)
                        summary_op = tf.summary.merge_all()
                    self.run_list.append(summary_op)

        return
