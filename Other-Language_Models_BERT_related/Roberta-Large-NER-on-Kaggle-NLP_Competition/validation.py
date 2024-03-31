from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import torch

from const import *
from train import *
from utils import *

import warnings
warnings.filterwarnings("ignore")

def inference(dl, model):
    """ Performs inference on a given dataset using the provided Named Entity Recognition (NER) model.

    This function processes the input text in batches and outputs the predicted named entity types for each token in the original text. The function ensures that each word is assigned a predicted label only once, even when it appears in multiple overlapping chunks.

    Args:
    dl (DataLoader): A DataLoader object that provides an iterable over the given dataset.
    model (nn.Module): A pre-trained NER model for inference.

    Returns:
    final_predictions (List[List[str]]): A list of lists containing the predicted named entity types for each token in the original text. """

    predictions = defaultdict(list)
    seen_words_idx = defaultdict(list)

    for batch in dl:
        ids = batch["input_ids"].to(config['device'])
        mask = batch["attention_mask"].to(config['device'])
        outputs = model(ids, attention_mask=mask, return_dict=False)
        # print('batch ', batch) # see output format below in comment

        del ids, mask

        batch_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy()
        # outputs[0] tensor contains the logits (raw output values) for each token, representing the probability distribution over all possible label classes.
        # torch.argmax() function is used to find the index of the highest value (the most probable class) along the last dimension (axis=-1) of the logits tensor. this index corresponds to the predicted label for each token.
        # It is needed because the model's output is in the form of logits, which are not directly interpretable. By using torch.argmax(), we can obtain the predicted label indices that are easily understandable and can be mapped back to their corresponding string labels for evaluation or further processing.

        # Go over each prediction, getting the text_id reference
        # batch_preds: contains the predicted label indices for each token in the batch.
        # batch['overflow_to_sample_mapping'].tolist(): This list maps the predicted labels back to their original text_ids. It is necessary because the input text may have been split into multiple chunks (due to token limits or other reasons), and we need to keep track of which part of the original text each prediction belongs to.
        # zip(): This function combines batch_preds and batch['overflow_to_sample_mapping'].tolist() element-wise, creating pairs of (chunk_preds, text_id).
        for k, (chunk_preds, text_id) in enumerate(zip(batch_preds, batch['overflow_to_sample_mapping'].tolist())):
            # print('chunk_preds ', chunk_preds)
            # its just a list like [2 2 2 2 2 2... 2 2]
            # print('batch.keys() ', batch.keys())
            # => batch.keys()  dict_keys(['input_ids', 'attention_mask', 'overflow_to_sample_mapping', 'labels', 'wids'])

            # The word_ids are absolute references in the original text
            word_ids = batch['wids'][k].numpy()

            # Map from ids to labels
            chunk_preds = [IDS_TO_LABELS[i] for i in chunk_preds]
            # This line is needed because the model's output is in the form of label indices, which are not directly interpretable. By mapping these indices to their corresponding string labels, we can easily understand the predicted named entity types for each token and evaluate the model's performance or use the predictions for further processing.

            for idx, word_idx in enumerate(word_ids):
                #  If the word index is -1, the loop should do nothing and continues to the next word index. This is because -1 represents a padding token or a special token that doesn't have a corresponding word in the original text.
                if word_idx == -1:
                    pass
                # ensure, that each word in the original text is assigned a predicted label only once, even when it appears in multiple overlapping chunks.
                elif word_idx not in seen_words_idx[text_id]:
                    #  This checks if the current word index has not been processed for the given text_id. This is done to avoid processing the same word multiple times when it appears in overlapping chunks.
                    # Add predictions if the word doesn't have a prediction from a previous chunk
                    predictions[text_id].append(chunk_preds[idx])
                    # Also add the current word index to the seen_words_idx dictionary,
                    seen_words_idx[text_id].append(word_idx)

    # print('predictions ', predictions)
    # predictions  defaultdict(<class 'list'>, {0: ['I-Concluding Statement', 'I-Concluding Statement', 'I-Concluding Statement', 'I-Concluding Statement', 'I-Concluding Statement', 'I-Concluding Statement', 'I-Concluding Statement'.... ]})

    #  list comprehension that iterates through each sorted key and retrieves the corresponding value (i.e., the predicted labels) from the predictions dictionary.  sorts the keys in ascending order, ensuring that the final predictions are in the same order as the original texts.
    final_predictions = [predictions[k] for k in sorted(predictions.keys())]
    return final_predictions

""" ### Output of `print('batch ', batch)` in `inference()` method

batch  {'input_ids': tensor([[    0,  4129,  6909,     8, 19181,  6893,   198,    47,   328,   404,
             ...]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1...]]), 'overflow_to_sample_mapping': tensor([0]), 'labels': tensor([[-100,    0,    0,    0,    0,    1,    2,  ... -100]]), 'wids': tensor([[ -1,   0,   0,   1,   2,   3,   4,   5,   5,   6,   7,   8,   9,  10,
          11,  12,  13,  14 ...  -1]])}

"""


def get_predictions(df, dl, model):
    """
    This function generates entity predictions for a given dataframe using a specified model and dataloader.

    Args:
        df (pd.DataFrame): A DataFrame containing input data for which predictions will be generated.
        dl (torch.utils.data.DataLoader): A DataLoader object to batch and process the input data.
        model (torch.nn.Module): A PyTorch model for Named Entity Recognition (NER) to generate predictions.

    Returns:
        df_pred (pd.DataFrame): A DataFrame containing entity predictions for each input in the original DataFrame.
                                The columns in the output DataFrame are:
                                - 'id': The input ID from the original DataFrame.
                                - 'class': The predicted entity class.
                                - 'predictionstring': A string of space-separated indices corresponding to the
                                                      predicted entity in the input.
    """
    predicted_labels = inference(dl, model)
    # initializes an empty list called final_preds that will store the final entity predictions for each input in the DataFrame.
    final_preds = []

    for i in range(len(df)):
        #extract the ID of the current input from the DataFrame.
        idx = df.id.values[i]
        #get the predicted label corresponding to the current input
        pred = predicted_labels[i]
        j = 0

        while j < len(pred):
            #assign the label at index j from the predicted labels pred to the variable cls.
            cls = pred[j]
            #check if the current label cls is an 'O' (representing no entity). If it is, the code will skip
            if cls == 'O': pass
            # The approach I am following here is that, during inference our model will make predictions for each subword token. Some single words consist of multiple subword tokens. In the code below, we use a word's first subword token prediction as the label for the entire word. Other alternatives that could have been tried are like averaging all subword predictions or taking `B` labels before `I` labels etc.

            #If the current label is not 'O', this line replaces 'B' with 'I' in the label. This is done to consider 'B' and 'I' tags as the same, simplifying the task of extracting the entity.
            #The reason for replacing 'B' with 'I' in this line is to treat both 'B' and 'I' tags as the same for the purpose of extracting entities. By doing this, we can easily identify and extract the entire entity by looking for a continuous sequence of the same 'I' tag. This simplification makes it easier to find the start and end indices of the entities in the input text.
            else: cls = cls.replace('B','I')
            # the purpose of below block of code is to find the end index of a continuous entity in the predicted labels.
            end = j + 1
            #above line is done because we want to start looking for the end of the entity from the next position in the predicted labels.

            #start a while-loop that iterates through the predicted labels, starting from index end.
            # The loop continues as long as the following conditions are met:
            # end is less than the length of pred (ensuring we don't exceed the boundaries of the list)
            #The label at index end (pred[end]) is equal to the current label cls (indicating that the current entity is still continuing)
            #end += 1 - This line increments the value of end by 1 in each iteration of the while-loop. This ensures that the loop progresses through the predicted labels until it finds the end of the continuous entity or reaches the end of the list.
            #The significance of this block is to identify the continuous sequence of the same entity label in the predicted labels, which represents a single entity in the input text. By finding the start and end indices of this continuous sequence, we can effectively extract the complete entity from the input text.
            while end < len(pred) and pred[end] == cls:
                end += 1
            # j > 0 for only testing the whole nb with just few samples of the entire .txt data
            # Else I will get axis mismatch problem from pandas
            # but for full training data its j > 7
            if cls != 'O' and cls != '' and end - j > 0:
                final_preds.append((idx, cls.replace('I-',''),
                                    ' '.join(map(str, list(range(j, end))))))
            j = end # at last increment or update the j by pushing it forward to the 'end' position
            """
            If the current label cls is not an 'O' (representing no entity) + not an empty string + The length of the continuous entity sequence is greater than 7 => Then

            If the above conditions are met, then append a tuple to the final_preds list with the following elements:

            idx: The input ID from the original DataFrame.

            cls.replace('I-', ''): The predicted entity class, with the 'I-' prefix removed. (Note, earlier we have already replaced all 'B' with 'I', so here I will be left with 'I-' only)

            ' '.join(map(str, list(range(j, end)))): A string of space-separated indices corresponding to the predicted entity in the input, created by converting the list of indices from j to end (exclusive) to a string representation.

            end - j > 7 This condition is a filter to only include entities that span more than 7 tokens in the input text. Depending on the problem at hand, this threshold may be adjusted or removed to include entities of any length.

            The significance of this block is to extract the entity information from the predicted labels, filter entities based on the length threshold, and store the final entity predictions in a structured format.
            """

    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['id','class','predictionstring']
    df_pred.head(2)
    return df_pred


"""
Explanations of the below line

' '.join(map(str, list(range(j, end))))


range(j, end): This part of the code creates a range of integers starting from j (inclusive) to end (exclusive). j is the starting index of the current entity, and end is the index immediately after the last token of the current entity.


list(range(j, end)): This part converts the range object into a list of integers. The list contains all the indices that correspond to the tokens of the predicted entity in the input text.


map(str, list(range(j, end))): This part applies the str function to each element of the list using the map() function. The purpose of this step is to convert the list of integers into a list of strings. This is necessary because we want to join these indices into a single string in the next step.

' '.join(map(str, list(range(j, end)))): Finally, this part uses the join() method of the string ' ' (a single space) to concatenate all the string elements in the list, separated by a space. This results in a single string with space-separated indices corresponding to the predicted entity in the input text.


The reason for using this line in the get_predictions() method is to create a compact and human-readable representation of the token indices for the predicted entity in the input text. This string representation is then added to the final_preds list as part of the tuple storing the entity prediction information,


"""

def validate(model, df_train_org, df_val, dataloader_val, epoch, unique_valid_id_list ):
    """ Validates the performance of a given NER model on a validation dataset.

    This function computes the F1-score for each class and the overall F1-score for the model on the validation dataset. It prints the F1-score per class and the overall F1-score.

    Args:
    model (nn.Module): A pre-trained NER model for validation.
    df_train_org (pd.DataFrame): The original training DataFrame, containing both training and validation samples.
    df_val (pd.DataFrame): The validation DataFrame, containing a subset of the columns of df_train_org.
    dataloader_val (DataLoader): A DataLoader object that provides an iterable over the validation dataset.
    epoch (int): The current epoch number in the training process.
    unique_valid_id_list (List[int]): A list of unique validation sample IDs. """

    time_start = time.time()

    # Put model in eval model
    model.eval()

    df_valid = df_train_org.loc[df_train_org['id'].isin(unique_valid_id_list)]

    out_of_fold = get_predictions(df_val, dataloader_val, model)

    f1s = []
    classes = out_of_fold['class'].unique()

    epoch_prefix = f"[Epoch {epoch+1:2d} / {config['epochs']:2d}]"
    print(f"{epoch_prefix} Validation F1 scores")

    f1s_log = {}
    for c in classes:
        # creates a new DataFrame pred_df by filtering the out_of_fold DataFrame for rows where the 'class' column matches the current class c. The .copy() method is used to create a copy of the filtered data, ensuring that any changes made to pred_df won't affect the original out_of_fold DataFrame. out_of_fold contains the model's predictions for the validation dataset. By creating pred_df, the function can focus on the performance of the model for the specific class c when calculating the F1-score.
        pred_df = out_of_fold.loc[out_of_fold['class']==c].copy()
        gt_df = df_valid.loc[df_valid['discourse_type']==c].copy()
        f1 = compute_macro_f1_score(pred_df, gt_df)
        print(f"{epoch_prefix}   * {c:<10}: {f1:4f}")
        f1s.append(f1)
        f1s_log[f'F1 {c}'] = f1

    elapsed = time.time() - time_start
    print(epoch_prefix)
    print(f'{epoch_prefix} Overall Validation F1: {np.mean(f1s):.4f} [{elapsed:.2f} secs]')
    print(epoch_prefix)
    f1s_log['Overall F1'] = np.mean(f1s)
    wandb.log(f1s_log)