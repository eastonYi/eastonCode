#pylint: disable=W0401
import sys
from configs.arguments import args, logging, Path, AttrDict

args.dirs.train.list_files = args.dirs.train.data.split(',')
args.dirs.dev.list_files = args.dirs.dev.data.split(',')
args.dirs.test.list_files = args.dirs.test.data.split(',')

if args.dirs.test:
    args.dirs.test.list_files = args.dirs.test.data.split(',')

# dataset
from dataProcessing.dataHelper import LMDataSet
dataset_train = LMDataSet(
    list_files=args.dirs.train.list_files,
    args=args,
    _shuffle=True)
dataset_dev = LMDataSet(
    list_files=args.dirs.dev.list_files,
    args=args,
    _shuffle=False)
dataset_test = LMDataSet(
    list_files=args.dirs.test.list_files,
    args=args,
    _shuffle=False)

args.dataset_train = dataset_train
args.dataset_dev = dataset_dev
args.dataset_test = dataset_test

# model
## encoder
if args.model.encoder.type == 'Listener':
    from tfSeq2SeqModels.encoders.listener import Listener as encoder
elif args.model.encoder.type == 'cnn_listener':
    from tfSeq2SeqModels.encoders.cnn_listener import CNN_Listener as encoder
elif args.model.encoder.type == 'conv_lstm_lh':
    from tfSeq2SeqModels.encoders.conv_lstm_lh import CONV_LSTM as encoder
elif args.model.encoder.type == 'conv_lstm':
    from tfSeq2SeqModels.encoders.conv_lstm import CONV_LSTM as encoder
elif args.model.encoder.type == 'BLSTM':
    from tfSeq2SeqModels.encoders.blstm import BLSTM as encoder
elif args.model.encoder.type == 'CNN':
    from tfSeq2SeqModels.encoders.cnn import CNN as encoder
else:
    encoder = None

args.model.encoder.type = encoder

## model
if args.model.structure == 'Seq2SeqModel':
    from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel as Model
elif args.model.structure == 'rnaModel':
    from tfSeq2SeqModels.rnaModel import RNAModel as Model
elif args.model.structure == 'ctcModel':
    from tfModels.ctcModel import CTCModel as Model
elif args.model.structure == 'ctcPolicyModel':
    from tfModels.ctcModelPolicy import CTC_PolicyModel as Model
elif args.model.structure == 'seq2seqPolicyModel':
    from tfSeq2SeqModels.seq2seqPolicy2 import Seq2SeqPolicyModel as Model
elif args.model.structure == 'languageModel':
    from tfSeq2SeqModels.languageModel import LanguageModel as Model
else:
    raise NotImplementedError('not found Model type!')
args.Model = Model

# vocab
logging.info('using vocab: {}'.format(args.dirs.vocab))
