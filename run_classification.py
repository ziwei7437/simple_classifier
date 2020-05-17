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
    parser.add_argument("--task_name",
                        default='snli',
                        type=str,
                        help="Task Name. Optional. For Infersent usages")

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
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = get_args()
    print_args(args)

    device, n_gpu = initialization.init_cuda_from_args(args, logger=logger)
    initialization.init_seed(args, n_gpu=n_gpu, logger=logger)

    initialization.init_output_dir(args)
    initialization.save_args(args)

    classifier = simple_classifier(n_classes=args.n_classes, n_hidden=args.fc_dim)
    classifier = classifier.to(device)

    optimizer = SGD(classifier.parameters(), lr=0.001, momentum=0.9)
    runner = Runner(classifier=classifier,
                    optimizer=optimizer,
                    device=device,
                    rparams=RunnerParameters(
                        num_train_epochs=args.num_train_epochs,
                        train_batch_size=args.train_batch_size,
                        eval_batch_size=args.eval_batch_size,
                    ))

    # dataset
    train_dataset = torch.load(os.path.join(args.data_dir, "train.dataset"))
    eval_dataset = torch.load(os.path.join(args.data_dir, "dev.dataset"))
    if args.mnli:
        mm_eval_dataset = torch.load(os.path.join(args.data_dir, "mm_dev.dataset"))
    else:
        mm_eval_dataset = None

    # run training and validation
    to_save = runner.run_train_val(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        mm_eval_set=mm_eval_dataset,
    )

    # save training state to output dir.
    torch.save(to_save, os.path.join(args.output_dir, "training.info"))


if __name__ == '__main__':
    main()
