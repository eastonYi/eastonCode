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
import editdistance as ed

from tfTools.tfTools import get_session
from tfModels.tools import size_variables

from tfTools.tfRecord import readTFRecord
from datasets import ASRDataLoader

from utils.summaryTools import Summary
from utils.performanceTools import dev, decode_test
from utils.textTools import array_idx2char, array2text, batch_cer


def train():
    feat, label = readTFRecord(args.dirs.dev.tfdata, args, _shuffle=False, transform=True)
    dataloader_dev = ASRDataLoader(args.dataset_dev, args, feat, label, batch_size=args.batch_size, num_loops=1)
    feat, label = readTFRecord(args.dirs.train.tfdata, args, _shuffle=False, transform=True)
    dataloader_train = ASRDataLoader(args.dataset_train, args, feat, label, batch_size=args.batch_size, num_loops=10)

    tensor_global_step = tf.train.get_or_create_global_step()

    if args.dirs.lm_checkpoint:
        """
        build LM model that will be used in Model
        """
        model_lm = args.Model_LM(
            tensor_global_step,
            is_train=False,
            args=args.args_lm)
        args.lm_obj = model_lm
        saver_lm = tf.train.Saver(model_lm.variables())

        args.top_scope = tf.get_variable_scope()   # top-level scope
        args.lm_scope = model_lm.decoder.scope

    model = args.Model(
        tensor_global_step,
        is_train=True,
        args=args)

    model_infer = args.Model(
        tensor_global_step,
        is_train=False,
        args=args)

    if args.dirs.processor_checkpoint:
        """
        restore part of the model from other checkpoint
        """
        from tfTools.checkpointTools import list_variables

        list_vars_pretrained = list_variables(args.dirs.processor_checkpoint)

        list_decoder_vars = model_infer.variables('RNA_Model/decoder2')
        list_encoder_vars = model_infer.variables('RNA_Model/encoder')
        list_fc_vars = model_infer.variables('RNA_Model/decoder')
        list_vars = list_encoder_vars+list_fc_vars+list_decoder_vars

        dict_vars = {}
        for var in list_vars:
            name = var.name.split('/', maxsplit=1)[1].split(':')[0]
            for var_pre in list_vars_pretrained:
                if '/' not in var_pre[0]:
                    continue
                elif name == var_pre[0].split('/', maxsplit=1)[1]:
                    # 'var_name_in_checkpoint': var_in_graph
                    dict_vars[var_pre[0]] = var
                    break
        saver_acoustic = tf.train.Saver(dict_vars)
        print('load: ', len(dict_vars))

    start_time = datetime.now()
    saver = tf.train.Saver(model_infer.variables())

    size_variables()

    summary = Summary(str(args.dir_log))

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        if args.dirs.checkpoint_init:
            checkpoint = tf.train.latest_checkpoint(args.dirs.checkpoint_init)
            saver.restore(sess, checkpoint)
        elif args.dirs.acoustic_checkpoint:
            acoustic_checkpoint = tf.train.latest_checkpoint(args.dirs.acoustic_checkpoint)
            saver_acoustic.restore(sess, acoustic_checkpoint)
        elif args.dirs.lm_checkpoint:
            lm_checkpoint = tf.train.latest_checkpoint(args.dirs.lm_checkpoint)
            saver_lm.restore(sess, lm_checkpoint)

        dataloader_dev.sess = sess

        batch_time = time()
        num_processed = 0
        progress = 0
        while progress < args.num_epochs:
            for batch in dataloader_train:
                global_step = sess.run([tensor_global_step])
                dict_feed = {model.list_pl[i]: batch[i] for i in range(len(batch))}
                loss, shape_batch, _, info = sess.run(model.list_run, feed_dic=dict_feed)

                reward_ac, reward_lm = info

                num_processed += shape_batch[0]
                used_time = time()-batch_time
                batch_time = time()
                progress = num_processed/args.data.train.size_dataset

                if global_step % 10 == 0:
                    logging.info('loss: {:.2f}\tbatch: {} reward_ac: {:.2f}, reward_lm: {:.2f}, time:{:.2f}s {:.3f}% step: {}'.format(
                                  loss, shape_batch, used_time, progress*100.0, global_step))
                    summary.summary_scalar('loss', loss, global_step)
                    summary.summary_scalar('reward_ac', reward_ac, global_step)
                    summary.summary_scalar('reward_lm', reward_lm, global_step)

                if global_step % args.save_step == args.save_step - 1:
                    saver.save(get_session(sess), str(args.dir_checkpoint/'model'), global_step=global_step, write_meta_graph=2)

                if global_step % args.dev_step == args.dev_step - 1:
                    cer, wer = dev(
                        step=global_step,
                        dataloader=dataloader_dev,
                        model=model_infer,
                        sess=sess,
                        unit=args.data.unit,
                        idx2token=args.idx2token,
                        eos_idx=args.eos_idx,
                        min_idx=0,
                        max_idx=args.dim_output-1)
                    summary.summary_scalar('dev_cer', cer, global_step)
                    summary.summary_scalar('dev_wer', wer, global_step)

                if global_step % args.decode_step == args.decode_step - 1:
                    decode_test(
                        step=global_step,
                        sample=args.dataset_test[10],
                        model=model_infer,
                        sess=sess,
                        unit=args.data.unit,
                        idx2token=args.idx2token,
                        eos_idx=None,
                        min_idx=0,
                        max_idx=None)

                if args.num_steps and global_step > args.num_steps:
                    sys.exit()

    logging.info('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def infer():
    tensor_global_step = tf.train.get_or_create_global_step()
    dataset_dev = args.dataset_test if args.dataset_test else args.dataset_dev

    model_lm = args.Model_LM(
        tensor_global_step,
        is_train=False,
        args=args.args_lm)

    args.lm_obj = model_lm
    saver_lm = tf.train.Saver(model_lm.variables())

    args.top_scope = tf.get_variable_scope()   # top-level scope
    args.lm_scope = model_lm.decoder.scope

    model_infer = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        is_train=False,
        args=args)

    saver = tf.train.Saver(model_infer.variables())

    size_variables()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        checkpoint = tf.train.latest_checkpoint(args.dirs.checkpoint_init)
        checkpoint_lm = tf.train.latest_checkpoint(args.dirs.lm_checkpoint)
        saver.restore(sess, checkpoint)
        saver_lm.restore(sess, checkpoint_lm)

        total_cer_dist = 0
        total_cer_len = 0
        total_wer_dist = 0
        total_wer_len = 0
        with open(args.dir_model.name+'_decode.txt', 'w') as fw:
        # with open('/mnt/lustre/xushuang/easton/projects/asr-tf/exp/aishell/lm_acc.txt', 'w') as fw:
            for sample in tqdm(dataset_dev):
                if not sample:
                    continue
                dict_feed = {model_infer.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                             model_infer.list_pl[1]: np.array([len(sample['feature'])])}
                sample_id, shape_batch, beam_decoded = sess.run(model_infer.list_run, feed_dict=dict_feed)
                # decoded, sample_id, decoded_sparse = sess.run(model_infer.list_run, feed_dict=dict_feed)
                res_txt = array2text(sample_id[0], args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output-1)
                ref_txt = array2text(sample['label'], args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output-1)

                list_res_char = list(res_txt)
                list_ref_char = list(ref_txt)
                list_res_word = res_txt.split()
                list_ref_word = ref_txt.split()
                cer_dist = ed.eval(list_res_char, list_ref_char)
                cer_len = len(list_ref_char)
                wer_dist = ed.eval(list_res_word, list_ref_word)
                wer_len = len(list_ref_word)
                total_cer_dist += cer_dist
                total_cer_len += cer_len
                total_wer_dist += wer_dist
                total_wer_len += wer_len
                if cer_len == 0:
                    cer_len = 1000
                    wer_len = 1000
                if wer_dist/wer_len > 0:
                    print('ref  ' , ref_txt)
                    for i, decoded, score, rerank_score in zip(range(10), beam_decoded[0][0], beam_decoded[1][0], beam_decoded[2][0]):
                        candidate = array2text(decoded, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output-1)
                        print('res' ,i , candidate, score, rerank_score)
                        fw.write('res: {}; ref: {}\n'.format(candidate, ref_txt))
                    fw.write('id:\t{} \nres:\t{}\nref:\t{}\n\n'.format(sample['id'], res_txt, ref_txt))
                logging.info('current cer: {:.3f}, wer: {:.3f};\tall cer {:.3f}, wer: {:.3f}'.format(
                    cer_dist/cer_len, wer_dist/wer_len, total_cer_dist/total_cer_len, total_wer_dist/total_wer_len))
        logging.info('dev CER {:.3f}:  WER: {:.3f}'.format(total_cer_dist/total_cer_len, total_wer_dist/total_wer_len))


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', args.gpus)

    if param.mode == 'infer':
        os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
        logging.info('enter the INFERING phrase')
        infer()

    elif param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logging.info('enter the TRAINING phrase')
        train()

        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
