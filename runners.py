import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import numpy as np


def get_dataloader(dataset, batch_size, train=True):
    sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def compute_simple_accuracy(logits, labels):
    pred_arr = np.argmax(logits, axis=1)
    assert len(pred_arr) == len(labels)
    return (pred_arr == labels).mean()


class TrainEpochState:
    def __init__(self):
        self.tr_loss = 0
        self.global_step = 0
        self.nb_tr_examples = 0
        self.nb_tr_steps = 0


class TrainingState:
    def __init__(self):
        self.tr_loss = list()
        self.val_history = list()
        self.epoch_loss = list()


class RunnerParameters:
    def __init__(self, num_train_epochs, train_batch_size, eval_batch_size):
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size


class Runner:
    def __init__(self, classifier, optimizer, device, rparams:RunnerParameters):
        self.classifier = classifier
        self.optimizer = optimizer
        self.device = device
        self.training_state = TrainingState()
        self.eval_info = []
        self.state_dicts = []
        self.rparams = rparams

    def run_train_val(self, train_dataset, eval_dataset, mm_eval_set=None, verbose=True):
        if verbose:
            print("**** run train ****")
            print("using {}".format(self.device))
        train_dataloader = get_dataloader(train_dataset, batch_size=self.rparams.train_batch_size)
        eval_dataloader = get_dataloader(eval_dataset, batch_size=self.rparams.eval_batch_size, train=False)
        if mm_eval_set is not None:
            mm_eval_dataloader = get_dataloader(mm_eval_set, batch_size=self.rparams.eval_batch_size, train=False)

        # run train
        self.classifier.train()
        for _ in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            self.run_train_epoch(train_dataloader)
            result = self.run_val(eval_dataloader)
            if mm_eval_set is not None:
                mm_result = self.run_val(mm_eval_dataloader)
                combined_result = (result, mm_result)
                self.eval_info.append(combined_result)
            else:
                self.eval_info.append(result)
        return self.eval_info

    def run_train_val_with_state_dict_returned(self, train_dataset, eval_dataset, mm_eval_set=None, verbose=True):
        # clear previous results if there were.
        self.eval_info = []
        self.state_dicts = []

        # ignore mismatched cases here.
        if verbose:
            print("**** run train ****")
            print("using {}".format(self.device))
        train_dataloader = get_dataloader(train_dataset, batch_size=self.rparams.train_batch_size)
        eval_dataloader = get_dataloader(eval_dataset, batch_size=self.rparams.eval_batch_size, train=False)

        # Run train
        self.classifier.train()
        for _ in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            self.run_train_epoch(train_dataloader)
            result = self.run_val(eval_dataloader)
            self.eval_info.append(result)
            self.state_dicts.append(self.classifier.state_dict())
        return self.eval_info, self.state_dicts

    def run_train_epoch(self, dataloader):
        train_epoch_state = TrainEpochState()
        for step, batch in enumerate(tqdm(dataloader)):
            u = batch[0].to(self.device)
            v = batch[1].to(self.device)
            label = batch[2].to(self.device)
            loss = self.classifier(u, v, label)
            loss.backward()

            self.training_state.tr_loss.append(loss.item())
            # print("Mini-batch Loss: {:.4f}".format(self.training_state.tr_loss[-1]))

            train_epoch_state.tr_loss += loss.item()
            train_epoch_state.nb_tr_examples += batch[0].size(0)
            train_epoch_state.nb_tr_steps += 1

            self.optimizer.step()
            self.optimizer.zero_grad()
            train_epoch_state.global_step += 1
        self.training_state.epoch_loss.append(train_epoch_state.tr_loss)

    def run_val(self, dataloader):
        self.classifier.eval()
        all_logits, all_labels = [], []
        total_eval_loss = 0
        nb_eval_steps = 0
        for step, batch in enumerate(tqdm(dataloader)):
            u = batch[0].to(self.device)
            v = batch[1].to(self.device)
            label = batch[2].to(self.device)
            with torch.no_grad():
                tmp_eval_loss = self.classifier(u, v, label)
                logits = self.classifier(u, v)
            total_eval_loss += tmp_eval_loss.mean().item()
            label = label.cpu().numpy()
            logits = logits.detach().cpu().numpy()

            all_logits.append(logits)
            all_labels.append(label)
            nb_eval_steps += 1
        eval_loss = total_eval_loss / nb_eval_steps
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        accuracy = compute_simple_accuracy(all_logits, all_labels)
        return {
            "loss": eval_loss,
            "accuracy": accuracy
        }

    def run_test(self, dataloader):
        self.classifier.eval()
        all_logits = []
        for step, batch in enumerate(tqdm(dataloader, desc="Predictions (Test)")):
            u = batch[0].to(self.device)
            v = batch[1].to(self.device)
            with torch.no_grad():
                logits = self.classifier(u, v)
            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)
        all_logits = np.concatenate(all_logits, axis=0)
        return all_logits


if __name__ == '__main__':
    from torch.optim import SGD
    from classifier import simple_classifier

    path = "../bert_embeddings/qqp_qqp/"
    data = torch.load(path+"train.dataset")
    val_data = torch.load(path+"dev.dataset")
    loader = get_dataloader(data, batch_size=64)

    model = simple_classifier(2, 768)
    model.to("cuda")
    opt = SGD(model.parameters(), lr=0.001, momentum=0.9)
    runner = Runner(classifier=model, optimizer=opt, device="cuda", rparams=RunnerParameters(5, 64, 64))

    to_return = runner.run_train_val(data, val_data)
    print(to_return)
