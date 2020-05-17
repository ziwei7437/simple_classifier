import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss


class simple_classifier(nn.Module):
    def __init__(self, n_classes, n_hidden, drop=0.1):
        super(simple_classifier, self).__init__()
        self.input_dim = 4 * n_hidden
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.n_classes = n_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, n_hidden),
            nn.Linear(n_hidden, n_classes)
        )

    def forward(self, u, v, labels=None):
        u = self.drop1(u)
        v = self.drop2(v)
        features = torch.cat((u, v, torch.abs(u-v), u*v), -1)
        logits = self.classifier(features)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_classes), labels.view(-1))
            return loss
        else:
            return logits
