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

model_URL = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'

encoder = hub.load(model_URL)


def embed_text(text: List[str]) -> List[np.ndarray]:
    """
    Used to generate vector representations of text using an encoder
    Embeds a list of text strings using a pre-defined 'encoder' and converts them into numpy arrays.

    Args:
        text (List[str]): List of text strings to be embedded.

    Returns:
        List[np.ndarray]: List of numpy arrays representing the embedded texts.
    """
    vectors = encoder(text)
    return [vector.numpy() for vector in vectors]
''' Want the vectors to be numpy arrays, not Tensorflow tensors, b/c they'll be used in PyTorch. '''


def encoder_factory(label2int: Dict[str, int]):
    """
    creates a custom function to process a batch of data. The returned function, encode, transforms a batch by embedding its text and encoding its labels according to the provided mapping label2int.

    Args:
        label2int (Dict[str, int]): A dictionary that maps class labels to integers.

    Returns:
        Function: A function that takes a batch of data and returns the batch with
                  text data embedded and labels encoded to integers.
    """
    def encode(batch):
        """
        Encode a batch of data.

        Args:
            batch (Dict[str, Union[List[str], List[int]]]): A dictionary containing a batch of data.
                Expected keys are "text" and "label".

        Returns:
            Dict[str, Union[List[np.ndarray], List[int]]]: The input batch with text data embedded and labels encoded.
        """
        batch["embedding"] = embed_text(batch["text"])
        batch["label"] = [label2int[str(x)] for x in batch["label"]]
        return batch

    return encode

class YelpDataLoader(pl.LightningDataModule):
    """
    YelpDataLoader is a PyTorch Lightning DataModule for the Yelp Polarity Dataset.
    It loads the dataset, applies preprocessing steps, and creates DataLoaders for training, validation, and test sets.
    """
    def __init__(self,
                 batch_size: int = 32,
                 num_workers: int = 2):
        """
        Args:
            batch_size (int, optional): Size of the mini-batches. Defaults to 32.
            num_workers (int, optional): How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Defaults to 2.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available()  # type: ignore

    def prepare_data(self):
        ''' This method loads a subset of the train and test sets.
        It uses the first 2% of the train and test sets to
        train and test, respectively.
        It uses the last 1% of the training set for validation.
        '''
        self.test_ds = load_dataset('yelp_polarity', split="test[:2%]")
        self.train_ds = load_dataset('yelp_polarity', split="train[:2%]")
        self.val_ds = load_dataset('yelp_polarity', split="train[99%:]")

        self.label_names = self.train_ds.unique("label")
        label2int = {str(label): n for n, label in enumerate(self.label_names)}
        self.encoder = encoder_factory(label2int)

    def setup(self):
        """
        This method is called on every GPU when using DDP (Distributed Data Parallelism) and it's the right place to define
        transformations and dataset splits.
        Here, it applies the encoder to the train, validation and test datasets and sets the dataset format to PyTorch.
        """
        # Compute embeddings in batches, so that they fit in the GPU's RAM.
        self.train = self.train_ds.map(self.encoder, batched=True, batch_size=self.batch_size)
        self.train.set_format(type="torch", columns=["embedding", "label"],                     # type: ignore
                              output_all_columns=True)

        self.val = self.val_ds.map(self.encoder, batched=True, batch_size=self.batch_size)
        self.val.set_format(type="torch", columns=["embedding", "label"],                       # type: ignore
                            output_all_columns=True)

        self.test = self.test_ds.map(self.encoder, batched=True, batch_size=self.batch_size)
        self.test.set_format(type="torch", columns=["embedding", "label"],                      # type: ignore
                             output_all_columns=True)

    def train_dataloader(self):
        """
        DataLoader for the training set.
        Returns:
            DataLoader: DataLoader for the training set.
        """
        return DataLoader(self.train,                                                           # type: ignore
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=True)

    def val_dataloader(self):
        """
        DataLoader for the validation set.
        Returns:
            DataLoader: DataLoader for the validation set.
        """
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


