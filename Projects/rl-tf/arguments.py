#pylint: disable=W0401
import sys
from configs.arguments import args, logging, Path, AttrDict

args.dirs.train.list_files = args.dirs.train.data.split(',')
args.dirs.dev.list_files = args.dirs.dev.data.split(',')
args.dirs.test.list_files = args.dirs.test.data.split(',')

args.dirs.train.tfdata = Path(args.dirs.train.tfdata)
args.dirs.dev.tfdata = Path(args.dirs.dev.tfdata)
args.dirs.train.list_files = args.dirs.train.data.split(',')
args.dirs.dev.list_files = args.dirs.dev.data.split(',')

if not args.dirs.train.tfdata.is_dir():
    args.dirs.train.tfdata.mkdir()
if not args.dirs.dev.tfdata.is_dir():
    args.dirs.dev.tfdata.mkdir()

# dataset
if args.dirs.type == 'scp':
    from dataProcessing.dataHelper import ASR_scp_DataSet
    dataset_train = ASR_scp_DataSet(
        f_scp=args.dirs.train.data,
        f_trans=args.dirs.train.label,
        args=args,
        _shuffle=True,
        transform=False)
    dataset_dev = ASR_scp_DataSet(
        f_scp=args.dirs.dev.data,
        f_trans=args.dirs.dev.label,
        args=args,
        _shuffle=False,
        transform=False)
    dataset_test = ASR_scp_DataSet(
        f_scp=args.dirs.test.data,
        f_trans=args.dirs.test.label,
        args=args,
        _shuffle=False,
        transform=True)
elif args.dirs.type == 'csv':
    from dataProcessing.dataHelper import ASR_csv_DataSet
    dataset_train = ASR_csv_DataSet(
        list_files=args.dirs.train.list_files,
        args=args,
        _shuffle=True,
        transform=False)
    dataset_dev = ASR_csv_DataSet(
        list_files=args.dirs.dev.list_files,
        args=args,
        _shuffle=False,
        transform=False)
    dataset_test = ASR_csv_DataSet(
        list_files=args.dirs.test.list_files,
        args=args,
        _shuffle=False,
        transform=True)
else:
    raise NotImplementedError('not dataset type!')
args.dataset_dev = dataset_dev
args.dataset_train = dataset_train
args.dataset_test = dataset_test

# model
if args.model.processor.type == 'conv':
    from tfRLModels.processors.conv import CONV_Processor as processor
else:
    raise NotImplementedError('not found processor type: {}'.format(args.model.agent.type))
args.model.processor.type = processor

if args.model.agent.type == 'lstm':
    from tfRLModels.agents.lstm_agent import LSTM_Agent as agent
else:
    raise NotImplementedError('not found agent type: {}'.format(args.model.agent.type))
args.model.agent.type = agent

if args.model.env.type == 'LM':
    from tfRLModels.env.lm_env import LM_ENV as env
else:
    raise NotImplementedError('not found env type: {}'.format(args.model.env.type))
args.model.env.type = env


## model
if args.model.structure == 'Seq2SeqModel':
    from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel as Model
elif args.model.structure == 'rnaModel':
    from tfSeq2SeqModels.rnaModel import RNAModel as Model
elif args.model.structure == 'ctcModel':
    from tfSeq2SeqModels.ctcModel import CTCModel as Model
elif args.model.structure == 'ctc_LM_Model':
    from tfSeq2SeqModels.ctc_LM_Model import CTCLMModel as Model
elif args.model.structure == 'CTCLMMultilabelModel':
    from tfSeq2SeqModels.ctc_LM_Multilabel_Model import CTCLMMultilabelModel as Model
elif args.model.structure == 'ctcPolicyModel':
    from tfModels.ctcModelPolicy import CTC_PolicyModel as Model
elif args.model.structure == 'seq2seqPolicyModel':
    from tfSeq2SeqModels.seq2seqPolicy2 import Seq2SeqPolicyModel as Model
elif args.model.structure == 'languageModel':
    from tfSeq2SeqModels.languageModel import LanguageModel as Model
elif args.model.structure == 'AC_LM_Classifier':
    from tfSeq2SeqModels.AC_LM_Classifier import AC_LM_Classifier as Model
elif args.model.structure == 'transformer':
    from tfSeq2SeqModels.Transformer import Transformer as Model
else:
    raise NotImplementedError('not found Model type!')

args.Model = Model

# lm part
if args.dirs.lm_config:
    from utils.configReader import AttrDict
    import yaml
    args.args_lm = AttrDict(yaml.load(open(args.dirs.lm_config)))
    args.args_lm.dim_output = len(args.token2idx)
    args.args_lm.list_gpus = args.list_gpus
    from tfSeq2SeqModels.languageModel import LanguageModel
    args.Model_LM = LanguageModel


# vocab
logging.info('using vocab: {}'.format(args.dirs.vocab))

if args.dirs.vocab_pinyin:
    from utils.vocab import load_vocab
    logging.info('using pinyin vocab: {}'.format(args.dirs.vocab_pinyin))
    args.phone.token2idx, args.phone.idx2token = load_vocab(args.dirs.vocab_pinyin)
    args.phone.dim_output = len(args.phone.token2idx)
    args.phone.eos_idx = None
    args.phone.sos_idx = args.phone.token2idx['<blk>']

def read_tfdata_info(dir_tfdata):
    data_info = {}
    with open(dir_tfdata/'tfdata.info') as f:
        for line in f:
            if 'dim_feature' in line or \
                'num_tokens' in line or \
                'size_dataset' in line:
                line = line.strip().split(' ')
                data_info[line[0]] = int(line[1])

    return data_info

try:
    info_dev = read_tfdata_info(args.dirs.dev.tfdata)
    args.data.dev.dim_feature = info_dev['dim_feature']
    args.data.dev.size_dataset = info_dev['size_dataset']

    args.data.dim_feature = args.data.dev.dim_feature
    args.data.dim_input = args.data.dim_feature * \
        (args.data.num_context +1) *\
        (2 if args.data.add_delta else 1)
except:
    print("Unexpected error: ", sys.exc_info())

try:
    """
    exists tfdata and will build model and training
    """
    info_train = read_tfdata_info(args.dirs.train.tfdata)
    args.data.train.dim_feature = info_train['dim_feature']
    args.data.train.size_dataset = info_train['size_dataset']

    args.data.dim_feature = args.data.train.dim_feature
    args.data.dim_input = args.data.dim_feature * \
        (args.data.num_context +1) *\
        (3 if args.data.add_delta else 1)

    logging.info('feature dim: {}; input dim: {}; output dim: {}'.format(
        args.data.dim_feature, args.data.dim_input, args.dim_output))
except:
    """
    no found tfdata and will read data and save it into tfdata
    won't build model
    """
    print("Unexpected error:", sys.exc_info())
    print('no finding tfdata ...')
