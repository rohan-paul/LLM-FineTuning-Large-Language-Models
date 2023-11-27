import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, load_metric
import numpy as np

from typing import List, Dict

import dataloading

class Model(pl.LightningModule):
    """
    A PyTorch Lightning module that encapsulates a multilayer perceptron model.

    Args:
        hidden_dims (List[int], optional): A list of integers representing the number of hidden units
            in each layer. Defaults to [768, 128].
        dropout_prob (float, optional): The dropout probability for regularization. Defaults to 0.5.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
    """
    def __init__(self,
                 hidden_dims: List[int] = [768, 128],
                 dropout_prob: float = 0.5,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.train_acc = load_metric("accuracy")
        self.val_acc = load_metric("accuracy")
        self.test_acc = load_metric("accuracy")
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate

        self.embedding_dim = 512

        layers = []
        prev_dim = self.embedding_dim

        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            prev_dim = h
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
        # output layer
        layers.append(nn.Linear(prev_dim, 2))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): A batch of input embeddings.

        Returns:
            torch.Tensor: Output logits.
        """
        # x will be a batch of USEm vectors
        logits = self.layers(x)
        return logits

    def configure_optimizers(self):
        """
        Configure the Adam optimizer for the model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # type: ignore
        return optimizer

    def __compute_loss(self, batch):
        """
        Compute the cross-entropy loss and predictions for a batch.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing a batch of data.
                Expected keys are "embedding" and "label".

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The computed loss,
                predictions and true labels.
        """
        x, y = batch["embedding"], batch["label"]
        logits = self(x)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        loss = F.cross_entropy(logits, y)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step, calculating the loss and other metrics.

        Args:
            batch (Dict[str, torch.Tensor]): The input data for this batch, which includes
            the sentences and their labels.
            batch_idx (int): The index of this batch.

        Returns:
            torch.Tensor: The loss for this batch.
        """
        loss, preds, y = self.__compute_loss(batch)
        self.train_acc.add_batch(predictions=preds, references=y)
        acc = self.train_acc.compute()["accuracy"] # type: ignore
        values = {"train_loss": loss, "train_accuracy": acc}
        self.log_dict(values, on_step=True, on_epoch=True,
                      prog_bar=True, logger=True)
        return loss
    ''' on_epoch argument in log_dict automatically accumulates and logs at the end of the epoch.  '''

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step, calculating the loss and other metrics.

        Args:
            batch (Dict[str, torch.Tensor]): The input data for this batch, which includes
            the sentences and their labels.
            batch_idx (int): The index of this batch.

        Returns:
            torch.Tensor: The loss for this batch.
        """
        loss, preds, y = self.__compute_loss(batch)
        self.val_acc.add_batch(predictions=preds, references=y)
        acc = self.val_acc.compute()["accuracy"]    # type: ignore
        values = {"val_loss": loss, "val_accuracy": acc}
        self.log_dict(values, on_step=True, on_epoch=True,
                      prog_bar=True, logger=True)
        return loss


    def test_step(self, batch, batch_idx):
        """
        Performs a single test step, calculating the loss and other metrics.

        Args:
            batch (Dict[str, torch.Tensor]): The input data for this batch, which includes
            the sentences and their labels.
            batch_idx (int): The index of this batch.

        Returns:
            torch.Tensor: The loss for this batch.
        """
        loss, preds, y = self.__compute_loss(batch)
        self.test_acc.add_batch(predictions=preds, references=y)
        acc = self.test_acc.compute()["accuracy"]   # type: ignore
        values = {"test_loss": loss, "test_accuracy": acc}
        self.log_dict(values, on_step=False, on_epoch=True,
                      prog_bar=True, logger=True)
        return loss
