import os
import wandb
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

from train import *
from const import *

import warnings
warnings.filterwarnings("ignore")

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def initialize_wandb(project: str, run_name: str, config: dict, entity: str = None) -> None:
    """
    Initializes the Weights & Biases (WandB) API key and logs relevant information for the current run.

    Args:
    project (str): The name of the project being tracked in WandB.
    run_name (str): The name of the current run.
    config (dict): A dictionary containing the hyperparameters and other configuration settings being used for the current run.
    entity (str, optional): The name of the organization or team running the project.

    Returns:
    None
    """
    wandb.login()
    wandb.init(project=project, entity=entity, name=run_name, config=config)

""" In above, the entity parameter is now optional, meaning that it has a default value of None. If an entity value is provided when the function is called, it will be used in the WandB initialization; otherwise, the WandB initialization will not include an entity. """

###########################################################################
###### From text create a new column with NER labels attached #############
###########################################################################

def attach_ner_to_text(input_dir: str, output_file: str, df_train_org: pd.DataFrame) -> pd.DataFrame:
    """
    Convert text words into NER labels and save the results in a CSV file.

    This function reads text files from the specified input directory, processes the texts, and creates
    a dataframe with NER labels for each word in the texts. The resulting dataframe is then saved as a
    CSV file.

    :param input_dir: Path to the directory containing the input text files.
    :param output_file: Path to the output CSV file.
    :param df_train_org: DataFrame containing id, discourse_type, and predictionstring columns.
    :return: DataFrame containing the NER labels for each word in the texts.
    """
    train_names, train_texts = [], []
    for f in tqdm(list(os.listdir(input_dir))):
        # Remove the ‘.txt’ extension from the current file name f and append the result to the train_names list.
        train_names.append(f.replace('.txt', ''))
        # Open the current file, read its content, and append the text to the train_texts list.
        train_texts.append(open(input_dir + f, 'r').read())

    df_ner_texts = pd.DataFrame({'id': train_names, 'text': train_texts})

    # Add a new column ‘text_split’ to the df_ner_texts DataFrame, which contains the list of words in the ‘text’ column.
    df_ner_texts['text_split'] = df_ner_texts.text.str.split()

    # Initialize an empty list all_entities, which will store the NER labels for each word in the texts.
    all_entities = []
    # Iterate over the rows of the df_ner_texts DataFrame using the iterrows() method
    # df.iterrows() is an iterator that allows you to loop through the rows of a DataFrame. It returns an iterator that yields pairs of index and rows as pandas Series objects. This is useful when you want to perform operations or access data for each row in the DataFrame. Very useful where you need to process each row individually.
    # However, df.iterrows() can be slow, especially for large DataFrames, because it returns a Series object for each row, which can have significant overhead. If you need to perform operations on the entire DataFrame or on specific columns, it's usually better to use vectorized operations or the apply method.
    for _, row in tqdm(df_ner_texts.iterrows(), total=len(df_ner_texts)):
        # Get the number of words in the current row’s ‘text_split’ column and store it in the variable total.
        total = len(row['text_split'])
        # Create a list entities with the same length as total, initializing all elements to the label “O” (which stands for “Outside” in NER, meaning no entity).
        entities = ["O"] * total

        # Iterate over the rows in df_train_org that has the same ‘id’ as the current row in df_ner_texts
        for _, row2 in df_train_org[df_train_org['id'] == row['id']].iterrows():
            # Get the ‘discourse_type’ value from the current row in df_train_org and store it in the variable discourse.
            discourse = row2['discourse_type']
            # Convert the ‘predictionstring’ value in the current row of df_train_org into a list of integers and store it in the variable list_ix.
            list_ix = [int(x) for x in row2['predictionstring'].split(' ')]
            # Assign the label “B-{discourse}” (e.g., “B-LEAD”) to the first element of list_ix in the entities list.
            entities[list_ix[0]] = f"B-{discourse}"
            # Now, assign an intermediate (I) label to all the words in the entity (except the first word, which gets a beginning (B) label) to indicate that they are part of the same entity.
            #  This is a loop that iterates over the elements of list_ix, starting from the second element (index 1) until the end of the list.
            # list_ix contains the indices of the words that are part of the entity (discourse) in the text. The first element (index 0) in list_ix is assigned a "beginning" (B) label, so we start iterating from the second element (index 1) to assign "intermediate" (I) labels to the remaining words.
            for k in list_ix[1:]:
                #  assign an intermediate (I) label to the word at the index k in the entities list.
                entities[k] = f"I-{discourse}"
        all_entities.append(entities)

    df_ner_texts['entities'] = all_entities
    df_ner_texts.to_csv(output_file, index=False)
    return df_ner_texts



###########################################################################
##################  download Huggingface model  ###########################
###########################################################################

def download_hf_model(model_path, model_name ):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # os.mkdir(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    # save the tokenizer in disk at path `MODEL_PATH`.
    # To use the save_pretrained method, you need to first create a model or tokenizer object, which can be either a pre-trained model provided by Hugging Face or a custom model/tokenizer that you have trained yourself. Once you have the model/tokenizer object, you can call the save_pretrained method on it and provide a path where you want to save the serialized model/tokenizer.
    tokenizer.save_pretrained(model_path)

    # initialize a configuration object for a pre-trained model. The "AutoConfig.from_pretrained()" method is used to automatically retrieve the configuration associated with the specified pre-trained model from the Hugging Face Model Hub.
    # The configuration object contains various settings and hyperparameters that define the architecture and behavior of the pre-trained model. These can include the number of layers, hidden layer sizes, attention mechanisms, activation functions, dropout rates, and many other options that can affect the model's performance and computational efficiency.
    config_model = AutoConfig.from_pretrained(model_name)
    config_model.num_labels = 15
    config_model.save_pretrained(model_path)

    backbone = AutoModelForTokenClassification.from_pretrained(model_name,
                                                               config=config_model)
    backbone.save_pretrained(model_path)
    print(f"Model downloaded to {model_path}/")

""" Why do I need `tokenizer.save_pretrained(model_path)` at all
In most cases, you would need to save the tokenizer in addition to the model because the tokenizer is an essential component of the model. The tokenizer is used to convert raw input text into the format expected by the model, and it is also used to convert the model's output back into text.

When you train a model or fine-tune a pre-trained model, you typically also train or fine-tune a tokenizer to handle the specific input data and output format of your task. If you do not save the tokenizer along with the model, you would not be able to use the model for inference or further training without the corresponding tokenizer.

Additionally, saving the tokenizer along with the model ensures that the same vocabulary and encoding scheme used during training are used during inference. If you were to use a different tokenizer during inference, you could encounter encoding mismatches or other issues that could negatively impact the model's performance.

In summary, you need to save the tokenizer along with the model to ensure that you can use the model for inference or further training and to ensure consistency between the encoding scheme used during training and during inference. """


#################################################

# This function is for creating a mapping between text_split and entities
# See above: (df_ner_texts['text_split'].str.len() == df_ner_texts['entities'].str.len()).all() == True
def get_labels_for_word_ids(word_ids, word_labels, LABELS_TO_IDS):
    # print("word_ids ", word_ids)
    # [None, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46 .... ]
    # The word_ids is the output of word_ids(i) method of the Encoding object which returns a list of indices, where each index corresponds to the original word in the input text for each token. It is useful when you want to map the tokenized text back to the original words.
    label_ids = []
    #  For each word_idx in word_ids, it checks if word_idx is None. If it is None, this indicates a padding token, so the function appends -100 to the label_ids list. If word_idx is not None, it looks up the corresponding label ID in LABELS_TO_IDS using word_labels[word_idx] and appends this value to label_ids.
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:
            label_ids.append(LABELS_TO_IDS[word_labels[word_idx]])
    # print('label_ids ', label_ids)
    # It will be a list like [-100, 2,2 ... 12]
    return label_ids

""" In above case why use the specific number -100

The primary reason for choosing this specific number is that it is outside the typical range of label IDs used in NER and other NLP tasks.

By using a value like -100, which is unlikely to collide with any valid label ID, it allows the loss function to easily identify and ignore these padding tokens. This ensures that the model does not learn from these padding tokens and focuses only on the actual input tokens and their corresponding labels.

It's worth noting that you could use a different value to represent padding tokens as long as it doesn't conflict with any valid label IDs, and you would need to modify the masking mechanism in the loss function to accommodate the new value. However, the choice of -100 has become a convention in the PyTorch community, so using it helps maintain consistency and compatibility with existing code and pre-trained models. """

# Tokenize texts, possibly generating more than one tokenized sample for each text
def tokenize(df, tokenizer, DOC_STRIDE, MAX_LEN, LABELS_TO_IDS, to_tensor=True, GET_LABELS=True):
    """ This function basically returns the encoded/tokenized object but after adding some extra information like encoded['labels'] and encoded['wids']
    For the encoded['wids']  - This information can be useful later to map back to the original words in the input text, especially when dealing with chunked text. By replacing None values with -1, it ensures that the list only contains integer values, making it easier to work with downstream.
    """

    encoded = tokenizer(df['text_split'].tolist(),
                        is_split_into_words=True,
                        return_overflowing_tokens=True,
                        stride=DOC_STRIDE,
                        max_length=MAX_LEN,
                        padding="max_length",
                        truncation=True)

    # print('encoded.keys() ', encoded.keys())
    #  dict_keys(['input_ids', 'attention_mask', 'overflow_to_sample_mapping'])
    print("encoded.word_ids ", encoded.word_ids )

    if GET_LABELS:
        encoded['labels'] = []

    encoded['wids'] = []
    n = len(encoded['overflow_to_sample_mapping'])
    print("encoded['overflow_to_sample_mapping'] ", encoded['overflow_to_sample_mapping'])
    # [0, 0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 10, 11, 12, 12, 12, 13]
    for i in range(n):

        # Map back to original row
        text_idx = encoded['overflow_to_sample_mapping'][i]
        # print('text_idx ', text_idx)
        # will print 1, 2, 3 etc

        # Get word indexes (this is a global index that takes into consideration the chunking :D )
        word_ids = encoded.word_ids(i)
        # retrieves the word indices corresponding to the original words in the input text for the ith tokenized sample (or chunk) in the encoded object. The encoded object is an instance of the Encoding class, which is created when tokenizing text using the Hugging Face's tokenizer.
        # encoded.word_ids(i) is a method of the Encoding object that takes an integer i as an argument, representing the index of the tokenized sample. It returns a list of indices, where each index corresponds to the original word in the input text for each token.
        print('word_ids ', word_ids)
        # [None, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46 .... ]


        if GET_LABELS:
            # Get word labels of the full un-chunked text
            word_labels = df['entities'].iloc[text_idx]
            # print('word_labels ', word_labels)
            #  ['O', 'O', 'O', 'B-Lead', 'I-Lead', 'I-Lead ...... ]

            # Get the labels associated with the word indexes
            label_ids = get_labels_for_word_ids(word_ids, word_labels, LABELS_TO_IDS)
            encoded['labels'].append(label_ids)
        encoded['wids'].append([w if w is not None else -1 for w in word_ids])

    if to_tensor:
        encoded = {key: torch.as_tensor(val) for key, val in encoded.items()}
    return encoded


""" # Tokenization and chunking

The `tokenizer` in above is called with the following parameters:

1. **is_split_into_words (bool, optional, defaults to False)** — Whether or not the input is already pre-tokenized (e.g., split into words). If set to True, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize.

`is_split_into_words` DOES NOT MEAN that the text was already pre-tokenized. This is not the case, it means that the string was split into words (not tokens), i.e., split on spaces.

2. `return_overflowing_tokens=True`, which activates the "chunking" mechanism (aka: will generate more than one tokenized sample for texts with more than 512 tokens).

3. `stride`: This is a parameter that determines the number of tokens by which to slide the window or move the starting point of the next chunk while breaking the document into smaller pieces. For example, if you have a `STRIDE` of 50 tokens and a maximum sequence length of 200 tokens, the first chunk will consist of tokens 1-200, the second chunk of tokens 51-250, the third chunk of tokens 101-300, and so on.

4. `return_overflowing_tokens=True`, besides creating the extra samples for long texts, sets the key `overflow_to_sample_mapping` in the resulting dictionary, which has the index of the original text that generated each of the samples.

5. The `word_ids(idx)` - When you tokenize a piece of text, you will receive an Encoding object. This object contains information about the tokenized text, such as token IDs, attention masks, and more.

The word_ids method of the Encoding object returns a list of indices, where each index corresponds to the original word in the input text for each token. It is useful when you want to map the tokenized text back to the original words.

So this method returns a back-reference to the word index in the original text, indexed correctly no matter the chunk. This is, for each token in the tokenized output, it says which word of the original text generated that token. """


######################################
######## Validation Metric  ##########
######################################

# https://www.kaggle.com/robikscube/student-writing-competition-twitch
def overlap_percentage(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    In both the overlap, the numerator is the size of the intersection. Both calculations have to be at least 0.5 for the prediction to be considered a hit.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # The set() function is used to convert the strings in each column
    # into sets of individual words. Then, the len() function is used
    # to calculate the length of each set.
    len_gt = len(set_gt)
    len_pred = len(set_pred)

    # Next, the intersection() method is used to find the common elements
    # between the two sets, i.e., the overlap between
    # predictionstring_pred and predictionstring_gt.
    inter = len(set_gt.intersection(set_pred))

    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred

    # returns a list of two values: the first value represents
    # the overlap of the ground truth with respect to the prediction, and
    # the second value represents the overlap of the prediction with respect to the ground truth.
    return [overlap_1, overlap_2]


def compute_macro_f1_score(pred_df, gt_df):
    """
    Just follow Evaluation Instructions of the Kaggle Competition:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    As per this - f1 scores here are averaged across classes making it macro_f1 and not micro_f1
    """
    # First extract the relevant columns from the predicted and ground truth dataframes,
    # and creates copies of these dataframes to avoid modifying the original data.
    gt_df = gt_df[['id', 'discourse_type', 'predictionstring']
                  ].reset_index(drop=True).copy()
    pred_df = pred_df[['id', 'class', 'predictionstring']
                      ].reset_index(drop=True).copy()
    """ The `reset_index(drop=True)` call is used to reset the index of the DataFrame after filtering out some columns. By default, when columns are filtered from a DataFrame, the index remains the same, and the original index values are preserved. However, in some cases, such as when joining or merging DataFrames, it can be helpful to have a simple sequential index with no missing values.

    By setting `drop=True`, the old index is dropped and a new sequential index is created. The `copy()` method is used to create a copy of the DataFrame with the new index, instead of modifying the original DataFrame in place. """

    # next two lines add a unique identifier for each prediction and ground truth,
    # by adding a new column called pred_id and gt_id respectively.
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index

    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id', 'class'],
                           right_on=['id', 'discourse_type'],
                           how='outer',
                           suffixes=('_pred', '_gt')
                           )
    """ Above does an outer join to ensure all predicted and ground truth rows are included in the resulting dataframe.
    An outer join returns all the rows from both data frames and combines them based on the specified columns and any missing values in the columns used for merging will be filled with NaNs.
    The "left_on" and "right_on" parameters are used to specify the columns from the left and right data frames on which to merge i.e. which columns should be used as left join keys or Right join keys, respectively. The "suffixes" parameter is used to add suffixes to the column names to differentiate between the columns in the merged data frame.

    In the pred_df dataframe, id and class columns represent the identifier and class of the predicted values, respectively. In the gt_df dataframe, id and discourse_type columns represent the identifier and class of the ground truth values, respectively. By using these columns to join the dataframes, we can compare the predicted and ground truth values that have the same identifier and class.
    """

    # replacing any missing predictionstring values with an empty string.
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(
        ' ')

    # The overlap_percentage function is applied to each row of the joined dataframe
    joined['overlaps'] = joined.apply(overlap_percentage, axis=1)

    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])
    # applies the eval() function (to evaluate it to get the actual tuple), to each element
    # e.g. (0.4, 0.6), will evaluate to the list [0.4, 0.6] with eval()
    # in the 'overlaps' column to retrieve the list of two values,
    # which were previously returned by the overlap_percentage() function.

    # So, after these two lines, the joined dataframe will have two new columns overlap1 and overlap2
    # that respectively contain the overlap percentage between the ground truth and prediction, and
    # the overlap percentage between the prediction and ground truth.

    # As per Kaggle - If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (
        joined['overlap2'] >= 0.5)

    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis=1)
    # By setting axis=1, we are telling max() to look across each row instead of looking at the entire column.
    # This means that the max() method will return the maximum overlap between prediction and ground truth pairs for each row in the dataframe.
    # i.e. max out of  'overlap1' and 'overlap2'

    tp_pred_ids = joined.query('potential_TP') \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id', 'predictionstring_gt']).first()['pred_id'].values
    # `joined.query` filters the rows in joined where potential_TP is True.
    # .sort_values('max_overlap', ascending=False) sorts the resulting dataframe by max_overlap in descending order. This is because we want to prioritize the predicted IDs with the highest overlap.
    # .groupby(['id', 'predictionstring_gt']).first() groups the resulting dataframe by id and predictionstring_gt, and selects the first row in each group. This is because we want to keep only the ID with the highest overlap
    # ['pred_id'].values extracts the pred_id values from the resulting dataframe and converts them into a NumPy array.
    # Thus, tp_pred_ids contains an array of predicted IDs that are potentially true positives based on their overlap with the ground truth.

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique()
                   if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    # joined.query('potential_TP') filters the joined DataFrame to include only those rows where the potential_TP column is True, i.e., where there is a potential true positive match between prediction and ground truth.
    # ['gt_id'].unique() selects the gt_id column from the filtered DataFrame and returns an array of unique ground truth ids

    unmatched_gt_ids = [c for c in joined['gt_id'].unique()
                        if c not in matched_gt_ids]

    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # F1 Score = 2*(Recall * Precision) / (Recall + Precision)
    # Recall = TP/(TP+FN) and Precision = TP/(TP+FP)
    # by replacing the expressions for precision and recall in F1 scores the expression becomes
    f1_score = TP / (TP + 0.5 * (FP + FN))
    return f1_score