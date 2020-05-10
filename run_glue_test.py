import os
import argparse
import logging

import torch
from torch.optim import SGD

import initialization
from classifier import simple_classifier
from runners import Runner, RunnerParameters


def print_args(args):
    for k, v in vars(args).items():
        print("  {}: {}".format(k, v))


def get_args(*in_args):
    parser = argparse.ArgumentParser(description='simple-classification-after-bert-embeddings')

    # === Required Parameters ===
    parser.add_argument("--data_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="training dataset directory")
    parser.add_argument("--mnli",
                        default=False,
                        help='hack for mnli task')
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="Output directory")

    # === Optional Parameters ===

    # training args for classifier
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")

    # model
    parser.add_argument("--dropout", type=float, default=0.1, help="classifier dropout probability")
    parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
    parser.add_argument("--fc_dim", type=int, default=768, help="hidden size of classifier, size of input features")

    # gpu
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=-1, help="seed")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    # others
    parser.add_argument("--verbose", action="store_true", help='showing information.')

    args = parser.parse_args(*in_args)
    return args


def main():
    pass


if __name__ == '__main__':
    main()
