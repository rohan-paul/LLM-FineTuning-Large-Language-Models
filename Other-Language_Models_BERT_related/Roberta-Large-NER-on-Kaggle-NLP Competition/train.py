import os
import gc
import ast
import time
import wandb
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

from const import *

import warnings
warnings.filterwarnings("ignore")

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import wandb

""" # Training function

The PyTorch training uses a masked loss which avoids computing loss when target is `-100` (that's the reason of those `-100` around).

More explanations on using the Mask on `-100`

In NLP tasks, inputs are usually sequences of tokens (words or subwords). To train a model on batches of sequences, these sequences need to be of the same length. Padding is used to achieve this by adding special tokens (usually called pad tokens) to the shorter sequences until they are all of equal length.

However, during training, you don't want the model to consider these padding tokens when computing the loss, as they don't carry any meaningful information. To avoid this, a mask is applied to the loss computation, effectively ignoring the contribution of padding tokens to the overall loss.

In this specific case, the target value of -100 is used as a special identifier for padding tokens.

Take a look at get_labels_for_word_ids() method

```py
for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:
            label_ids.append(LABELS_TO_IDS[word_labels[word_idx]])
```

When computing the loss, any position in the target sequence with a value of -100 is masked out, and the loss for that position is not calculated or included in the final loss value. This technique ensures that the model is only optimized based on the relevant input tokens, which helps improve its performance on the actual NLP task. """


def train(model, optimizer, dataloader_train, epoch):

    time_start = time.time()

    # The optimizer has multiple parameter groups, each with its own learning rate and other optimization-related settings. The loop iterates over these parameter groups.
    for g in optimizer.param_groups:
        g['lr'] = config['learning_rates'][epoch]
        # The learning rate of the current parameter group (g) is updated with the value from the config['learning_rates'] list at the index corresponding to the current epoch. This allows you to set a specific learning rate for each epoch.
    lr = optimizer.param_groups[0]['lr']
    # After updating the learning rates for all parameter groups, the learning rate of the first parameter group is stored in the variable lr. This is mainly done for logging purposes, as the updated learning rate is printed in the next line of code (print(f"{epoch_prefix} Starting epoch {epoch+1:2d} with LR = {lr}")).


    epoch_prefix = f"[Epoch {epoch+1:2d} / {config['epochs']:2d}]"
    print(f"{epoch_prefix} Starting epoch {epoch+1:2d} with LR = {lr}")

    # Put model in training mode
    model.train()

    # Counter variables which will be updated during training
    tr_loss, tr_accuracy = 0, 0
    num_of_train_examples_counter, num_of_train_steps_counter = 0, 0

    # start a loop that iterates through all the batches produced by the DataLoader. In each iteration, it retrieves the index (idx) and the current batch (batch) of data. The index is useful for logging progress, performing operations at specific intervals, or debugging. The batch contains the input data and labels for the current set of examples.
    for idx, batch in enumerate(dataloader_train):

        ids = batch['input_ids'].to(config['device'], dtype = torch.long)
        # extracts the input_ids tensor from the current batch, which represents the tokenized input sequences.
        mask = batch['attention_mask'].to(config['device'], dtype = torch.long)
        # extracts the attention_mask tensor from the current batch. The attention mask is used to indicate which elements in the input sequences should be attended to and which should be ignored (e.g., padding tokens).
        labels = batch['labels'].to(config['device'], dtype = torch.long)
        # extracts the labels tensor from the current batch, which represents the true label annotations for each token in the input sequences

        loss, raw_target_output_logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                               return_dict=False)
        tr_loss += loss.item()

        num_of_train_steps_counter += 1
        num_of_train_examples_counter += labels.size(0)
        loss_step = tr_loss/num_of_train_steps_counter

        if idx % 200 == 0:
            print(f"{epoch_prefix}     Steps: {idx:4d} --> Loss: {loss_step:.4f}")


        # compute training accuracy
        flattened_labels = labels.view(-1) # shape (batch_size * seq_len,)
        # Above line reshapes the labels tensor from the shape (batch_size, seq_len) to a flattened shape (batch_size * seq_len,). This is done to simplify the computation of accuracy by treating all tokens in the batch as independent examples, irrespective of their original sequence. The -1 in the view() function is a placeholder that instructs PyTorch to automatically compute the correct size for that dimension based on the original tensor's shape and size.

        active_logits = raw_target_output_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        # This line reshapes the raw_target_output_logits tensor from the shape (batch_size, seq_len, num_labels) to a flattened shape (batch_size * seq_len, num_labels). The logits represent the model's raw output for each token and label class. By reshaping the logits in this way, we can efficiently compute the predicted labels for all tokens in the batch using the next line of code.

        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        # This line computes the predicted labels for each token in the batch by selecting the label class with the highest logit value (i.e., the most likely label according to the model). The torch.argmax() function is applied along axis=1, which corresponds to the num_labels dimension in the active_logits tensor. The output, flattened_predictions, will have the same flattened shape (batch_size * seq_len,) as flattened_labels.

        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        # The view(-1) function is used to reshape the labels tensor into a 1D (flattened) tensor. The -1 in the view() function is a placeholder that instructs PyTorch to automatically compute the correct size for that dimension based on the original tensor's shape and size.
        # the full line creates a boolean tensor by performing an element-wise comparison between the flattened labels tensor and the value -100.
        # Resulting tensor has the same shape as the flattened labels tensor (i.e., (batch_size * seq_len,))

        labels = torch.masked_select(flattened_labels, active_accuracy)
        #  returns a 1D tensor containing only the elements of flattened_labels where the corresponding elements of active_accuracy are True.
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        #  It uses the same active_accuracy mask to filter out the predicted labels corresponding to the padded or irrelevant tokens. The result is a 1D tensor containing only the predicted labels for the active tokens (i.e., non-padding and relevant tokens).

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # wandb.log({'Train Loss (Step)': loss_step, 'Train Accuracy (Step)' : tr_accuracy / num_of_train_steps_counter})

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=config['max_grad_norm']
        )
        # The clip_grad_norm_ method applies gradient clipping to the gradients of the model's parameters, and not to any other tensors or variables, because it is designed specifically for the purpose of preventing the gradients from becoming too large during training.
        # model.parameters() method returns an iterator over all the trainable parameters in the model. This includes the weights and biases of all the layers in the model.
        # max_norm=config['max_grad_norm']: This is the maximum allowed norm for the gradients. If the norm of the gradients exceeds this value, the gradients will be rescaled so that their norm is equal to max_norm, grabbed from the config dictionary.

        # The norm of the gradients refers to the magnitude or length of the gradient vector, which is a vector that contains the partial derivatives of the loss function with respect to each of the model's trainable parameters. The norm of the gradient is calculated using some norm function, such as the L2 norm, which is defined as the square root of the sum of the squared elements of the gradient vector.

        # backward pass
        optimizer.zero_grad()
        #  set the gradients of all the learnable parameters in the model to zero. This is typically done at the start of each iteration during the training process,
        # The reason for setting the gradients to zero before computing them is that PyTorch accumulates gradients by default (i.e., the gradients are not overwritten, but added to the existing gradients). Therefore, if we don't set the gradients to zero before each iteration, the gradients will accumulate, resulting in incorrect updates to the parameters.
        loss.backward()
        # loss.backward() compute the gradients of the loss function (using automatic differentiation) with respect to all the learnable parameters in the model.
        optimizer.step()
        # step() is used to update the model's parameters (i.e., weights and biases) based on the computed gradients. It applies the optimization algorithm defined by the optimizer (e.g., Adam, SGD) to adjust the model's parameters in a way that minimizes the loss.


    epoch_loss = tr_loss / num_of_train_steps_counter
    tr_accuracy = tr_accuracy / num_of_train_steps_counter

    torch.save(model.state_dict(), f'pytorch_model_e{epoch}.bin')
    torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - time_start

    print(epoch_prefix)
    print(f"{epoch_prefix} Training loss    : {epoch_loss:.4f}")
    print(f"{epoch_prefix} Training accuracy: {tr_accuracy:.4f}")
    print(f"{epoch_prefix} Model saved to pytorch_model_e{epoch}.bin  [{elapsed/60:.2f} mins]")
    # wandb.log({'Train Loss (Epoch)': epoch_loss, 'Train Accuracy (Epoch)' : tr_accuracy})
    print(epoch_prefix)

    """ ------------------------------------

### In above why do I need `num_of_train_examples_counter, num_of_train_steps_counter = 0, 0`

`num_of_train_examples_counter:` This variable keeps track of the total number of training examples processed so far in the current epoch. Each time the loop iterates over a new batch, the size of that batch (i.e., the number of examples in the batch) will be added to num_of_train_examples_counter. This variable helps to compute the average loss or accuracy over the entire epoch.

`num_of_train_steps_counter`: This variable keeps track of the total number of training steps (or iterations) completed so far in the current epoch. A training step is defined as one update to the model's parameters using a single batch of data.

In each iteration of the loop, `num_of_train_steps_counter` is incremented by 1. This variable is used to compute the average loss or accuracy over the entire epoch and can also be used for logging purposes or to trigger specific actions at certain intervals (e.g., learning rate adjustments, checkpoint saving, etc.).

As the loop iterates over the batches of data in the DataLoader, these two variables are updated accordingly:

`num_of_train_examples_counter` is updated with the number of examples in the current batch using the line

`num_of_train_examples_counter += labels.size(0)`

`num_of_train_steps_counter` is incremented by 1 in each iteration using the line

`num_of_train_steps_counter += 1`

By keeping track of these two variables, you can monitor the progress of the training process, calculate averages for evaluation metrics, and control other aspects of the training loop.

----------------------------------------

### In above why the the tensors needed to be flattened

Simplify computation: By flattening the tensors, you simplify the process of comparing the true labels and the predicted labels. Flattening the tensors essentially treats all tokens in the batch as independent examples, regardless of their original sequence. This allows you to compute accuracy in a straightforward manner using functions like accuracy_score() from scikit-learn, which expects 1D arrays or lists as inputs. Flattening the tensors makes it easier to perform element-wise comparisons and calculations without having to deal with nested loops or more complex tensor operations.

Ignore padded tokens: In many NLP tasks, input sequences are padded with special tokens (e.g., [PAD]) to ensure that all sequences in a batch have the same length. These padding tokens do not contribute to the accuracy calculation, and their presence may distort the overall accuracy metric. By flattening the tensors and using a mask to select only the "active" tokens (i.e., non-padding tokens), you can efficiently compute the accuracy while ignoring padded tokens. In the provided train() function, this is done using the active_accuracy mask and the torch.masked_select() function:

```py
active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
labels = torch.masked_select(flattened_targets, active_accuracy)
predictions = torch.masked_select(flattened_predictions, active_accuracy)

```

-------------------------------

## Here's how torch.masked_select() works:

* The input tensor is flattened into a 1D tensor with shape (n,), where n is the total number of elements in the tensor.
* The mask tensor is broadcasted to the same shape as the flattened input tensor. The broadcasted tensor has shape (n,) and each element of the tensor corresponds to the corresponding element of the flattened input tensor.
* The function selects the elements from the flattened input tensor where the corresponding element in the mask tensor is True.
* The selected elements are then returned as a 1D tensor.

example to illustrate the usage of `torch.masked_select()`:

```py
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask = torch.tensor([[True, False, True], [False, True, False], [True, False, True]])

result = torch.masked_select(x, mask)
print(result)

```

First, the input tensor x is flattened into a 1D tensor of shape (9,). The mask tensor mask is also flattened into a 1D tensor of the same shape.

Next, the mask tensor is broadcasted to the same shape as the flattened input tensor, which is (9,).

Then, the function selects the elements from the flattened input tensor x where the corresponding element in the broadcasted mask tensor is True. In this example, the selected elements are [1, 3, 5, 7, 9].

Finally, the selected elements are returned as a 1D tensor. """