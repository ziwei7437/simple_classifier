import os
import argparse
import logging
import pandas as pd

import torch
from torch.optim import SGD

import initialization
from classifier import simple_classifier
from runners import Runner, RunnerParameters
from run_classification import print_args, get_args


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

    classifier = simple_classifier(n_classes=args.n_classes, n_hidden=768)
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
    test_dataset = torch.load(os.path.join(args.data_dir, "test.dataset"))

    # run train and validation with state dicts returned
    eval_info, state_dicts = runner.run_train_val_with_state_dict_returned(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    torch.save(eval_info, os.path.join(args.output_dir, "training.info"))

    # find highest validation results, load model state dict, and then run prediction @ test set.
    val_acc = []
    for item in eval_info:
        val_acc.append(item['accuracy'])
    idx = val_acc.index(max(val_acc))
    print("highest accuracy on validation is: {}, index = {}. "
          "Load state dicts and run testing...".format(val_acc[idx], idx))

    torch.save(state_dicts[idx], os.path.join(args.output_dir, "state.p"))

    runner.classifier.load_state_dict(state_dicts[idx])
    logits = runner.run_test(test_dataset)
    df = pd.DataFrame(logits)
    df.to_csv(os.path.join(args.output_dir, "test_preds.csv"), header=False, index=False)
    # HACK for MNLI-mismatched
    if args.mnli:
        mm_test_dataset = torch.load(os.path.join(args.data_dir, "mm_test.dataset"))
        logits = runner.run_test(mm_test_dataset)
        df = pd.DataFrame(logits)
        df.to_csv(os.path.join(args.output_dir, "mm_test_preds.csv"), header=False, index=False)


if __name__ == '__main__':
    main()
