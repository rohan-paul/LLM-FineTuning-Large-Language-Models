from spacy import displacy
import spacy
import os
import warnings

import numpy as np
import tensorflow as tf

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def multi_gpu_use():
    if os.environ["CUDA_VISIBLE_DEVICES"].count(',') == 0:
        strategy = tf.distribute.get_strategy()
        print('single strategy')
    else:
        strategy = tf.distribute.MirroredStrategy()
        print('multiple strategy')
    return strategy


def prepare_data(train,
                 unique_doc_ids,
                 train_lens,
                 tokenizer,
                 train_tokens,
                 train_attention,
                 ROOT_DIR,
                 MAX_LEN,
                 target_id_map,
                 targets_b,
                 targets_i,
                 ):
    """
    It is used to convert raw text data into a format suitable for feeding into an NER model. The primary purpose of this function is to tokenize the input text, create attention masks, and generate target labels for the beginning and intermediate tokens of each named entity. And to achieve that this method updates 2 arrays `targets_b` and `targets_i`

    `targets_b` and `targets_i` arrays are used to store the ground truth labels for the beginning and intermediate tokens of each named entity class in the tokenized input. Each class has its own pair of arrays (`B_*` and `I_*`), with the same shape as `train_tokens` and `train_attention` arrays.

    Align with tokenized input and attention masks: By having the same shape as `train_tokens` and `train_attention` arrays, the `targets_b` and `targets_i` arrays ensure that the ground truth labels are aligned with the tokenized input and attention masks. This alignment is essential for training the model, as it helps the model learn to map the tokenized input to the correct named entity labels.

    And basically this method builds the 2 tensor targets_b and targets_i

    Args:
    train (pandas DataFrame): the training data with columns "id", "discourse_start", "discourse_end", and "discourse_type".
    unique_doc_ids (list): list of unique document IDs.
    train_lens (list): an empty list to hold the lengths of each training text.
    tokenizer (transformers Tokenizer): Tokenizer object from the transformers library.
    train_tokens (numpy array): an empty numpy array to hold the encoded training tokens.
    train_attention (numpy array): an empty numpy array to hold the training attention masks.
    ROOT_DIR (str): the path to the root directory containing the training data.
    MAX_LEN (int): the maximum length of the input sequence.
    target_id_map (dict): dictionary mapping each discourse type to a target ID.
    targets_b (numpy array): an empty numpy array to hold the beginning target labels.
    targets_i (numpy array): an empty numpy array to hold the intermediate target labels.

    Returns:
    None
    """
    assert np.sum(train.groupby("id")["discourse_start"].diff() <= 0) == 0

    """
    **discourse_start** - character position where discourse element begins in the essay response
    checks whether the difference between consecutive "discourse_start" values for each "id" group in the "train" dataframe is strictly greater than zero, i.e., the "discourse_start" values are in ascending order for each "id" group.

    ["discourse_start"].diff() calculates the difference between consecutive "discourse_start" values for each group. This returns a new series with the same length as the original series, where the first value is NaN (since there is no previous value to calculate the difference with).
    <= 0 checks whether each value in the series is less than or equal to zero, which SHOULD BE FALSE, because I am assuming the 'discourse_start" values are in ascending order and hence (nextvalue-previousvalue) should be a positive number.

    np.sum() sums the boolean values in the series (False = 0, True = 1).
    The assertion statement assert ... == 0 raises an error if the sum is not equal to zero, i.e., if there is at least one group where the "discourse_start" values are not in ascending order.
    The reason for this check is that the function processes each document sequentially and assumes that the input data is sorted. If some "discourse_start" values are not sorted in ascending order, it would mean that the training data is not properly ordered by the start position of the discourse in the text. This could potentially cause issues when iterating through the offsets of the tokenized text and aligning them with the target discourse spans. Specifically, if the discourse spans in the training data are not in ascending order, it may be possible to miss or incorrectly map some of the target spans to the tokenized text.
    """

    # start looping through each train text
    for id_num in range(len(unique_doc_ids)):
        if id_num % 100 == 0:
            print(id_num, ", ", end="")
        """ opens the text file, reads the contents, tokenizes the text using the tokenizer, and saves the tokenized input and attention masks in the train_tokens and train_attention arrays, respectively. It also saves the length of the original text in the train_lens array. """
        current_doc_id_num = unique_doc_ids[id_num]
        name = ROOT_DIR + f"train/{current_doc_id_num}.txt"
        txt = open(name, "r").read()
        train_lens.append(len(txt.split()))
        # train_lens is a list that holds the lengths (number of words) of each original text document in the training dataset. The line train_lens.append(len(txt.split())) appends the length of the current document to the train_lens list.
        # At the end of the execution of this entire prepare_data(), the train_lens list will contain the lengths of all original text documents in the training dataset. This is useful for understanding the distribution of document lengths in the dataset, and we will indeed build a histogram based on this variable
        tokens = tokenizer.encode_plus(
            txt,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        train_tokens[id_num, ] = tokens["input_ids"]
        train_attention[id_num, ] = tokens["attention_mask"]
        """ find the offsets of each token in the tokenized input using the return_offsets_mapping option of the tokenizer.encode_plus() method. This is important for mapping the targets in the original text to the tokenized input. """
        # find targets in text and save in target arrays
        offsets = tokens["offset_mapping"]
        # the above "offsets" will be a list of tuples, and the list has same length as the number of tokens, where each tuple contains the start and end index of the corresponding token in the original input text.
        # like [(0, 5), (6, 9), (10, 13), (14, 17), (18, 23)]
        offset_index = 0
        #`offset_index`: The index of the token in the tokenized input for which the label is being assigned.
        df = train.loc[train.id == current_doc_id_num]
        for _, row in df.iterrows():
            disc_start = row.discourse_start
            disc_end = row.discourse_end
            # check whether the current offset index is within the bounds of the offset array.
            # If the index is out of bounds, we will break out of the loop.
            if offset_index > len(offsets) - 1:
                break
            # "token_start" and "token_end" are the start and end indices of the current token in the tokenized text
            # token_start = offsets[offset_index][0]
            # token_end = offsets[offset_index][1]

            token_start, token_end = offsets[offset_index]
            # note offsets is a list of tuple like  [(0, 0),    (0, 2),    (3, 5),    (6, 8),  (9, 10)]

            # set a boolean variable to track whether the current target is the beginning of a span or not.
            beginning = True
            while disc_end > token_start:
                # This entire while loop continues as long as the discourse end position disc_end is greater than the token start offset token_start.
                # For example, consider a target span in the input text with a start position of 4 and an end position of 7. If the offset_mapping of the tokenized input text is:
                # [(0, 0),    (0, 2),    (3, 5),    (6, 8),  (9, 10)]
                # In this case, the while loop would iterate through the tokens with offset positions (3, 5) and (6, 8), since they fall within the boundaries of the target span. The final token (9, 10) would be skipped since it falls outside the target span.
                if (token_start >= disc_start) & (disc_end >= token_end):
                    # This conditional statement is used to determine if a token should be assigned a label (in `targets_b` or `targets_i` arrays) as part of a named entity.
                    # See Further explanations below
                    # Next map the discourse type to an integer using the target_id_map dictionary.
                    k = target_id_map[row.discourse_type]
                    # The beginning variable is used to keep track of whether we are currently processing the first token of the target span or not. If it is the first token (beginning = True), we update the corresponding position in the targets_b array with a value of 1. Otherwise, we update the corresponding position in the targets_i array with a value of 1.
                    if beginning:
                        targets_b[k][id_num][offset_index] = 1
                        # `targets_b`: A list of numpy arrays, where each array corresponds to the beginning target labels of a specific named entity class.
                        # `k`: The index of the named entity class in the `targets_b` list.
                        # `id_num`: The index of the current document in the `unique_doc_ids` list. This helps to locate the correct document in the `targets_b` array.
                        # `offset_index`: The index of the token in the tokenized input for which the label is being assigned.

                        # By setting targets_b[k][id_num][offset_index] = 1, you mark the token at the given offset_index as the beginning of a named entity of the class k. This assignment is crucial for training the NER model, as it helps the model learn to recognize the start of named entities in the text. All other tokens in the array that are not part of the named entity or not the beginning of the named entity will have a default value of 0, indicating they should not be considered as the start of a named entity of the class k.
                        # set the beginning variable to False, as the first token of the discourse span has already been processed.
                        beginning = False
                    else:
                        targets_i[k][id_num][offset_index] = 1
                        # If the token is not the first token in the discourse span, this line updates the corresponding position in the targets_i array to 1, indicating that the token is inside the discourse span.
                offset_index += 1
                if offset_index > len(offsets) - 1:
                    break
                # After updating the target arrays, we increment the offset_index variable to move to the next token in the input text, and update the values of token_start and token_end accordingly. This process continues until we have processed all the tokens in the input text or have reached the end position disc_end of the current target span.
                token_start = offsets[offset_index][0]
                token_end = offsets[offset_index][1]


""" Explanation 1. In `tokenizer.encode_plus` what exactly the param `return_offsets_mapping=True,` do

When return_offsets_mapping = True, the offset mappings for each token in the input text are returned in the output dictionary.

Offset mapping is a mapping between the original text and the tokenized text, where each element in the mapping is a tuple (start_offset, end_offset) that corresponds to a token in the tokenized text. start_offset is the starting position of the original text corresponding to the token and end_offset is the ending position of the original text corresponding to the token.

By default, return_offsets_mapping is set to False which means that the offset mapping is not returned.

Let's say we have the following sentence:

sentence = "The quick brown fox jumps over the lazy dog."

When we tokenize this sentence using a tokenizer, such as the Tokenizer class from the transformers library, we get the following tokenized output:

tokens = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']

Each token corresponds to one or more characters in the original sentence. The return_offsets_mapping=True parameter in the encode_plus function tells the tokenizer to return a list of offset mappings, where each mapping indicates the character positions in the original sentence that correspond to a particular token.

For example, the offset mapping for the token "quick" might be (4, 9), indicating that the characters at positions 4, 5, 6, 7, and 8 in the original sentence correspond to the "quick" token.

Here's an example of how the offset mappings might look for the entire sentence:

offsets = [(0, 3), (4, 9), (10, 15), (16, 19), (20, 25), (26, 30), (31, 34), (35, 39), (40, 43), (43, 44)]
This information can be useful in various natural language processing tasks, such as named entity recognition or part-of-speech tagging.


Explanation 2. if (token_start >= disc_start) & (disc_end >= token_end):

This line is a conditional statement that checks whether a token lies within the span of a named entity (discourse element) in the text. It's used to determine if the token should be assigned a label as part of the named entity.

*   `token_start`: The start position of the current token in the text.
*   `token_end`: The end position of the current token in the text.
*   `disc_start`: The start position of the discourse element (named entity) in the text.
*   `disc_end`: The end position of the discourse element (named entity) in the text.

The condition `(token_start >= disc_start) & (disc_end >= token_end)` checks if the current token lies within the boundaries of the discourse element.

Let's go through an example:

Consider the following text with a named entity (discourse element) "Named Entity" enclosed in square brackets:

`This is an example text with a [Named Entity] in it.`

Here, `disc_start` is the position of the opening square bracket `[`, and `disc_end` is the position of the closing square bracket `]`. Now, let's consider two tokens: "Named" and "text."

For the token "Named":

*   `token_start`: Position of the letter "N" in "Named"
*   `token_end`: Position of the letter "d" in "Named"

Since `token_start` is within the named entity and `disc_end` is greater than or equal to `token_end`, the condition `(token_start >= disc_start) & (disc_end >= token_end)` is satisfied. Therefore, the token "Named" is part of the named entity.

For the token "text":

*   `token_start`: Position of the letter "t" in the first "text"
*   `token_end`: Position of the letter "t" in the first "text"

In this case, `token_start` is less than `disc_start`, so the condition `(token_start >= disc_start) & (disc_end >= token_end)` is not satisfied. Therefore, the token "text" is not part of the named entity.

This conditional statement is used to determine if a token should be assigned a label (in `targets_b` or `targets_i` arrays) as part of a named entity.


"""

######################################
# Validation Metric
######################################


""" Below function is for evaluating the performance of an NLP model that predicts text spans, such as entity recognition or question answering. The overlap percentages is used to determine whether the predicted text span matches the ground truth text span with sufficient overlap.

The function takes a single argument, row, which is expected to be a pandas DataFrame row containing two columns, predictionstring_pred and predictionstring_gt.

"""


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
    Code referred from - https://www.kaggle.com/robikscube/student-writing-competition-twitch
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


######################################
########### displacy render  #########
######################################
colors = {
    'Lead': '#8000ff',
            'Position': '#2b7ff6',
            'Evidence': '#2adddd',
            'Claim': '#80ffb4',
            'Concluding Statement': 'd4dd80',
            'Counterclaim': '#ff8042',
            'Rebuttal': '#ff0000'
}


def displacy_discourse_type_visualize(example, train, path):
    ents = []
    # iterate over each row in the `train` dataframe where the `id` column
    # matches the given `example`.",
    # For each row, extracts the start and end indices of a discourse segment, and
    # its type, and appends a dictionary containing this information to the `ents` list.
    for i, row in train[train['id'] == example].iterrows():
        ents.append({
            'start': int(row['discourse_start']),
            'end': int(row['discourse_end']),
            'label': row['discourse_type']
        })

    # opens a text file whose name is the given `example`,
    # reads the contents into a variable called `data`.
    with open(path / f'{example}.txt', 'r') as file:
        data = file.read()

    doc2 = {
        "text": data,
        "ents": ents,
        "title": example
    }

    # create a dictionary called `options` with two keys:
    # "ents", which is a list of all the unique discourse types in the `train` dataframe, and
    # "colors", which is a dictionary mapping each discourse type to a color.
    options = {"ents": train.discourse_type.unique().tolist(),
               "colors": colors}

    # Finally, render a visualization of the text with the extracted discourse segments
    # highlighted according to their type and color.
    displacy.render(doc2, style="ent", options=options,
                    manual=True, jupyter=True)

    """ ## `displacy.render` function arguments

- `doc` (required): The Spacy `Doc` object to be rendered. This contains the text and any annotations that will be visualized.
- `style` (optional): The name of the visualization style to use. Spacy provides several built-in styles, such as "ent" for named entities or "dep" for dependency parse trees. Custom styles can also be defined.
- `options` (optional): A dictionary of visualization options. The available options depend on the chosen visualization style. For example, for the "ent" style, options can include  list of entity labels to show, colors to use for each label, and whether or not to show the entity's text label. Options can also include settings for the arrows, fonts, and backgrounds of the visualization.
- `manual` (optional): In manual mode, the user can edit or add annotations to the visualization.
- `jupyter` (optional): If `True`, the visualization will be rendered inline in the notebook.

There are also additional arguments for more advanced usage, such as providing a custom component to modify the document before rendering, or specifying an output file to save the visualization to. """
