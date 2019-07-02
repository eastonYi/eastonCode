#!/usr/bin/env python
from datetime import datetime
from time import time
import sys
import os
import logging
import tensorflow as tf
from arguments import args
from tqdm import tqdm
import numpy as np

from tfTools.tfTools import get_session
from tfModels.tools import create_embedding, size_variables

from dataProcessing.dataHelper import LMDataLoader, PTBDataLoader

from utils.summaryTools import Summary
from utils.textTools import array_idx2char, array2text, array_char2idx
from utils.tools import check_to_stop

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
print('CUDA_VISIBLE_DEVICES: ', args.gpus)


def train():
    dataloader_train = LMDataLoader(args.dataset_train, 999999, args)
    # dataloader_train = PTBDataLoader(args.dataset_train, 80, args)
    tensor_global_step = tf.train.get_or_create_global_step()

    model = args.Model(
        tensor_global_step,
        is_train=True,
        args=args)
    model_infer = args.Model(
        tensor_global_step,
        is_train=False,
        args=args)
    input_pl = tf.placeholder(tf.int32, [None, None])
    len_pl = tf.placeholder(tf.int32, [None])
    score_T, distribution_T = model.score(input_pl, len_pl)

    size_variables()
    start_time = datetime.now()
    checker = check_to_stop()

    saver = tf.train.Saver(max_to_keep=15)
    summary = Summary(str(args.dir_log))

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        if args.dirs.checkpoint_init:
            checkpoint = tf.train.latest_checkpoint(args.dirs.checkpoint_init)
            saver.restore(sess, checkpoint)

        batch_time = time()
        num_processed = 0
        progress = 0
        for x, y, len_x, len_y in dataloader_train:
            global_step, lr = sess.run([tensor_global_step, model.learning_rate])
            feed_dict = {model.list_pl[0]: x,
                         model.list_pl[1]: y,
                         model.list_pl[2]: len_x,
                         model.list_pl[3]: len_y}
            loss, shape_batch, _ = sess.run(model.list_run, feed_dict=feed_dict)

            if global_step % 10 == 0:
                num_tokens = np.sum(len_x)
                ppl = np.exp(loss/num_tokens)
                summary.summary_scalar('loss', loss, global_step)
                summary.summary_scalar('ppl', ppl, global_step)
                summary.summary_scalar('lr', lr, global_step)

                num_processed += shape_batch[0]
                used_time = time()-batch_time
                batch_time = time()
                progress = num_processed/args.dataset_train.size_dataset
                logging.info('ppl: {:.3f}\tshape_batch: {} lr:{:.6f} time:{:.2f}s {:.3f}% step: {}'.format(
                              ppl, shape_batch, lr, used_time, progress*100.0, global_step))

            if global_step % args.save_step == args.save_step - 1:
                saver.save(get_session(sess), str(args.dir_checkpoint/'model'), global_step=global_step, write_meta_graph=False)

            if global_step % args.dev_step == args.dev_step - 1:
                ppl_dev = dev(args.dataset_dev, model_infer, sess)
                summary.summary_scalar('ppl_dev', ppl_dev, global_step)
                # accuracy = dev_external('/mnt/lustre/xushuang/easton/projects/asr-tf/exp/aishell/lm_acc.txt', model_infer, input_pl, len_pl, score_T, distribution_T, sess)
                # summary.summary_scalar('accuracy', accuracy, global_step)

    logging.info('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def dev(dataset, model, sess):
    dataloader = LMDataLoader(dataset, 1, args)
    # dataloader = PTBDataLoader(dataset, 1, args)
    num_processed = 0
    progress = 0
    loss_sum = 0
    num_tokens = 0
    for x, y, len_x, len_y in dataloader:
        feed_dict = {model.list_pl[0]: x,
                     model.list_pl[1]: y,
                     model.list_pl[2]: len_x,
                     model.list_pl[3]: len_y}
        loss, shape_batch = sess.run(model.list_run, feed_dict=feed_dict)
        num_processed += shape_batch[0]
        progress = num_processed/dataset.size_dataset
        # logging.info('{:.3f}%'.format(progress*100.0))
        num_tokens += np.sum(len_x)
        loss_sum += loss
    ppl = np.exp(loss_sum/num_tokens)
    logging.info('ppl: {:.3f}'.format(ppl))

    return ppl

def dev_external(test_file, model, input_pl, len_pl, score_T, distribution_T, sess):
    """
    test the performance of LM by exteranl metrix
    """
    num_right = 0
    batch_size = 40
    list_res = []
    list_ref = []
    with open(test_file) as f:
        for i, line in enumerate(f):
            res, ref = line.strip().split(';')
            res_text = res[5:]
            ref_text = ref[5:]
            list_res.append(res_text.strip())
            list_ref.append(ref_text.strip())
            if len(list_res) >= batch_size:
                decoder_input, len_seq = array_char2idx(list_res, args.token2idx, ' ')
                pad = np.ones([decoder_input.shape[0], 1], dtype=decoder_input.dtype)*args.sos_idx
                decoder_input_sos = np.concatenate([pad, decoder_input], -1)
                score_res, distribution = sess.run([score_T, distribution_T], {input_pl: decoder_input_sos, len_pl: len_seq})

                decoder_input, len_seq = array_char2idx(list_ref, args.token2idx, ' ')
                pad = np.ones([decoder_input.shape[0], 1], dtype=decoder_input.dtype)*args.sos_idx
                decoder_input_sos = np.concatenate([pad, decoder_input], -1)
                score_ref, distribution = sess.run([score_T, distribution_T], {input_pl: decoder_input, len_pl: len_seq})
                num_right += np.sum(score_ref >= score_res)

                list_res = []
                list_ref = []
        if list_res:
            if len(list_res) >= batch_size:
                decoder_input, len_seq = array_char2idx(list_res, args.token2idx, ' ')
                score_res = sess.run(score_T, {input_pl: decoder_input, len_pl: len_seq})
                decoder_input, len_seq = array_char2idx(list_ref, args.token2idx, ' ')
                score_ref = sess.run(score_T, {input_pl: decoder_input, len_pl: len_seq})
                num_right += np.sum(score_ref >= score_res)

    logging.info('accuracy: {:.3f}'.format(num_right/i))

    return num_right/i


def infer():
    tensor_global_step = tf.train.get_or_create_global_step()

    model_infer = args.Model(
        tensor_global_step,
        is_train=False,
        args=args)
    input_pl = tf.placeholder(tf.int32, [None, None])
    len_pl = tf.placeholder(tf.int32, [None])
    score_T, distribution_T = model_infer.score(input_pl, len_pl)
    # sampled_op, num_samples_op = model_infer.sample(max_length=50)

    size_variables()
    saver = tf.train.Saver(max_to_keep=40)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        checkpoint = tf.train.latest_checkpoint(args.dirs.checkpoint_init)
        saver.restore(sess, checkpoint)

        dev(args.dataset_test, model_infer, sess)
        # dev_external('/mnt/lustre/xushuang/easton/projects/asr-tf/exp/aishell/lm_acc.txt', model_infer, input_pl, len_pl, score_T, distribution_T, sess)
        # samples = sess.run(sampled_op, feed_dict={num_samples_op: 10})
        # samples = array_idx2char(samples, args.idx2token, seperator=' ')
        # print(samples)


def test():
    """
    containing sample test and score test
    """
    tensor_global_step = tf.train.get_or_create_global_step()

    model_infer = args.Model(
        tensor_global_step,
        is_train=False,
        args=args)
    sampled_op, num_samples_op = model_infer.sample(max_length=50)

    # pl_input = tf.placeholder([None, None])

    size_variables()

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        checkpoint = tf.train.latest_checkpoint(args.dirs.checkpoint_init)
        saver.restore(sess, checkpoint)

        # samples = sess.run(sampled_op, feed_dict={num_samples_op: 10})
        # samples = array_idx2char(samples, args.idx2token, seperator='')
        # for i in samples:
        #     print(i)
# res 0 就 想 我 的 时 候 自 然 会 需 要 面 包 -61.252304 -54.955883
# res 1 就 像 我 饿 的 时 候 自 然 会 需 要 面 包 -65.39448 -60.87168
# res 2 就 想 我 饿 的 时 候 自 然 会 需 要 面 包 -66.52325 -65.72158



        list_sents = ['郑 伟 电 视 剧 有 什 么',
                      '郑 伟 电 视 剧 有 什 么 么']

        decoder_input, len_seq = array_char2idx(list_sents, args.token2idx, ' ')
        pad = np.ones([decoder_input.shape[0], 1], dtype=decoder_input.dtype)*args.sos_idx
        decoder_input_sos = np.concatenate([pad, decoder_input], -1)
        score, distribution = model_infer.score(decoder_input_sos, len_seq)
        print(sess.run(score))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=str(args.list_gpus[0]))
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', args.gpus)

    if param.mode == 'infer':
        os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
        logging.info('enter the INFERING phrase')
        infer()
    elif param.mode == 'test':
        os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
        logging.info('enter the SAVING phrase')
        test()

    elif param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logging.info('enter the TRAINING phrase')
        train()
