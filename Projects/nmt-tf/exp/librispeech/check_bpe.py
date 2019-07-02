import os
import sys
import random
from argparse import ArgumentParser
from utils.vocab import load_vocab


def main(input_file, vocab):
    token2idx, idx2token = load_vocab(vocab)
    with open(input_file) as f:
        for line in f:
            line = line.strip().split()
            for token in line:
                if token not in token2idx.keys():
                    print(token)
                    sys.exit()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, help="the rate of context will be allocated to dev")
    parser.add_argument("--vocab", type=str)
    args = parser.parse_args()

    main(args.input, args.vocab)
