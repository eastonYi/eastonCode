'''
==================================================================
yaml配置文件
数据类型是定义在配置文件中的, 并且数据类型可以直接更具atom的颜色进行区分
另外还可以任意添加新的属性
程序调用demo:

from utils.tools import AttrDict
import yaml


CONFIG_FILE = sys.argv[1]
args = AttrDict(yaml.load(open(CONFIG_FILE)))

args.train.data_path
args.vocab
args.num_gpu = len(args.debug.gpus)
'''

class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    demo:
    a ={'length': 10, 'shape': (2,3)}
    config = AttrDict(a)
    config.length #10

    here we can recurrently use attribute to access confis
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        try:
            if type(self[item]) is dict:
                self[item] = AttrDict(self[item])
            res = self[item]
        except:
            print('not found {}'.format(item))
            res = None
        return res


if __name__ == '__main__':
    from argparse import ArgumentParser
    '''
    parser 一般放到运行的源码中
    调用demo:
    '''
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    parser.add_argument("--num_units", type=int, dest='num_units', default=32, help="Network size.")
    parser.add_argument("--encoder_type", type=str, default="uni", dest='encoder_type', help="""\
          uni | bi | gnmt. For bi, we build num_layers/2 bi-directional layers.For
          gnmt, we build 1 bi-directional layer, and (num_layers - 1) uni-
          directional layers.\
          """)
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    args.config
