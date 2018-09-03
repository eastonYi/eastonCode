import logging
import os
import sys
import yaml
from pathlib import Path
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

from utils.configReader import AttrDict
from utils.vocab import load_vocab

CONFIG_FILE = sys.argv[1]
args = AttrDict(yaml.load(open(CONFIG_FILE)))

args.num_gpus = len(args.gpus.split(','))
args.list_gpus = ['/gpu:{}'.format(i) for i in range(args.num_gpus)]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
print('CUDA_VISIBLE_DEVICES: ', args.gpus)

# dirs
args.dir_model = Path.cwd() / args.dirs.models / CONFIG_FILE.split('/')[-1]
args.dir_log = args.dir_model / args.dirs.log
args.dir_checkpoint = args.dir_model / args.dirs.checkpoint
args.path_vocab = Path.cwd() / args.dirs.vocab

if not args.dir_model.is_dir():
    args.dir_model.mkdir()
if not args.dir_log.is_dir():
    args.dir_log.mkdir()
if not args.dir_checkpoint.is_dir():
    args.dir_checkpoint.mkdir()
if not args.dir_checkpoint.is_dir():
    args.dir_checkpoint.mkdir()

# bucket
if args.bucket_boundaries:
    args.list_bucket_boundaries = [int(int(i))
        for i in args.bucket_boundaries.split(',')]
else:
    args.list_bucket_boundaries = [i
        for i in range(args.size_bucket_start,
                       args.size_bucket_end,
                       args.size_bucket_gap)]

args.list_batch_size = ([int(args.num_batch_token / boundary) * args.num_gpus
        for boundary in (args.list_bucket_boundaries)] + [args.num_gpus])
logging.info('\nbucket_boundaries: {} \nbatch_size: {}'.format(
    args.list_bucket_boundaries, args.list_batch_size))

# vocab
args.token2idx, args.idx2token = load_vocab(args.dirs.vocab)
args.dim_output = len(args.token2idx)

# learning rate
# Linear Scaling Rule
args.peak *= args.num_gpus
