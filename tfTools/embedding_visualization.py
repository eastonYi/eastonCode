import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector


def visualizer(dict_points):
    from utils.tools import AttrDict
    dict_points = AttrDict(dict_points)

    list_locations = []
    with open('./embedding_test/100_vocab.csv', 'w') as fw:
        for name, location in dict_points.items():
            list_locations.append(location)
            fw.write(str(name)+'\n')
    array_locations = np.array(list_locations, dtype=np.float32)

    with tf.device('/cpu:0'):
        tensor_locations = tf.Variable(array_locations)

    # config = tf.ConfigProto()
    # config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True
    # config.log_device_placement = False
    # with tf.Session(config=config) as sess:
    #     # Create summary writer.
    #     writer = tf.summary.FileWriter('./embedding_test', sess.graph)
    #     # Initialize embedding_var
    #     sess.run(tensor_locations.initializer)
    #     # Create Projector config
    #     config = projector.ProjectorConfig()
    #     # Add embedding visualizer
    #     embedding = config.embeddings.add()
    #     # Attache the name 'embedding'
    #     embedding.tensor_name = tensor_locations.name
    #     # Metafile which is described later
    #     embedding.metadata_path = '100_vocab.csv'
    #     # Add writer and config to Projector
    #     projector.visualize_embeddings(writer, config)
    #     # Save the model
    #     saver_embed = tf.train.Saver([tensor_locations])
    #     saver_embed.save(sess, './embedding_test/embedding_test.ckpt', 1)

    # writer.close()
    print('genrate finished!')

if __name__ == '__main__':

    # tensorboard --logdir=embedding_test
    dict_points = {}
    for i in range(100):
        loc = np.random.randn(10)
        dict_points[i] = loc

    visualizer(dict_points)
