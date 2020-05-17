import logging
import os
from turtle import pd

import torch
from torch.optim import SGD

import initialization
from classifier import simple_classifier
from run_classification import get_args, print_args
from runners import InfersentRunner, RunnerParameters


TRAIN_SET_NUM_MAP = {
    'snli': 5
}

EVAL_SET_NUM_MAP = {
    'snli': 1
}

TEST_SET_NUM_MAP = {
    'snli': 1
}


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
    runner = InfersentRunner(classifier=classifier,
                             optimizer=optimizer,
                             device=device,
                             rparams=RunnerParameters(
                                 num_train_epochs=args.num_train_epochs,
                                 train_batch_size=args.train_batch_size,
                                 eval_batch_size=args.eval_batch_size,
                             ))

    # dataset
    train_datasets = []
    for i in range(TRAIN_SET_NUM_MAP[args.task_name]):
        train_dataset = torch.load(os.path.join(args.data_dir, "train-{}.dataset".format(i)))
        train_datasets.append(train_dataset)

    eval_dataset = torch.load(os.path.join(args.data_dir, "dev-0.dataset"))

    if args.mnli:
        mm_eval_dataset = torch.load(os.path.join(args.data_dir, "mm_dev-0.dataset"))
    else:
        mm_eval_dataset = None

    # run training and validation
    eval_info, state_dicts = runner.run_train_val_with_state_dict_returned(
        train_dataset=train_datasets,
        eval_dataset=eval_dataset,
        mm_eval_set=mm_eval_dataset,
    )

    # save training state to output dir.
    torch.save(eval_info, os.path.join(args.output_dir, "training.info"))

    # find highest validation results, load model state dict, and then run prediction @ test set.
    val_acc = []
    for item in eval_info:
        val_acc.append(item['accuracy'])
    idx = val_acc.index(max(val_acc))
    print("highest accuracy on validation is: {}, index = {}. "
          "Load state dicts and run testing...".format(val_acc[idx], idx))

    torch.save(state_dicts[idx], os.path.join(args.output_dir, "state.p"))

    test_datasets = []
    for i in range(TEST_SET_NUM_MAP[args.task_name]):
        test_dataset = torch.load(os.path.join(args.data_dir, "test-{}.dataset".format(i)))
        test_datasets.append(test_dataset)

    runner.classifier.load_state_dict(state_dicts[idx])
    logits = runner.run_test(test_datasets)

    df = pd.DataFrame(logits)
    df.to_csv(os.path.join(args.output_dir, "test_preds.csv"), header=False, index=False)
    # HACK for MNLI-mismatched
    if args.mnli:
        mm_test_dataset = torch.load(os.path.join(args.data_dir, "mm_test-0.dataset"))
        logits = runner.run_test([mm_test_dataset])
        df = pd.DataFrame(logits)
        df.to_csv(os.path.join(args.output_dir, "mm_test_preds.csv"), header=False, index=False)


if __name__ == '__main__':
    main()
