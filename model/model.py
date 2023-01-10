import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchmetrics

#dataset

#dataloader

#model
class SAEncoder(nn.Module):
    """
    Model encoder.
    """

    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b4()
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class SAClassifier(nn.Module):
    """
    Model classifier.
    """

    def __init__(self, output):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(1792, output),
            nn.Softmax())

    def forward(self, x):
        x = self.head(x)
        return x


class SAModel(pl.LightningModule):
    """
    Full SVHN lightning module.
    """

    def __init__(self, lr=1e-3):
        super().__init__()
        self.encoder = SAEncoder()
        self.classifier = SAClassifier(128)
        self.lr = lr
        #self.loss = nn.CrossEntropyLoss()
        #self.acc = Accuracy()

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)


#train

#test