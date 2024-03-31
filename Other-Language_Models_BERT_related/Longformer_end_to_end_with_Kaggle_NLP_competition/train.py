import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, TFAutoModel
from transformers import *
from spacy import displacy
import spacy
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.ticker import FuncFormatter
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_model(MODEL_NAME, MAX_LEN):
    """
    Builds a model for NLP tasks using a pre-trained language model.

    Args:
        MODEL_NAME (str): Name of the pre-trained language model.
        MAX_LEN (int): Maximum length of input sequences.

    Returns:
        tf.keras.Model: Built model for NLP tasks.

    """

    tokens = tf.keras.layers.Input(
        shape=(MAX_LEN,), name='tokens', dtype=tf.int32)
    attention = tf.keras.layers.Input(
        shape=(MAX_LEN,), name='attention', dtype=tf.int32)

    # load the configuration file for the pre-trained language model
    config = AutoConfig.from_pretrained(MODEL_NAME)
    base_model_instance = TFAutoModel.from_pretrained(
        MODEL_NAME, config=config)

    x = base_model_instance(tokens, attention_mask=attention)

    # output x[0] is the final hidden activations, also called  "last_hidden_state" and has shape of
    # (batch_size, token sequence width, number of longformer features). and
    # x[1], is the pooler outputs. it has shape (batch_size, number of longformer features).
    # Since we just ignore the pooler output, we do not apply loss to the pooler output layers and they don't have gradients.
    # And reason for not using the pooler output, is because we need features for each token. The pooler output will only give us features for the entire sentence. (However someone may be able to find a innovative way to use the pooler output to see if that helps overall result).
    # Now add a dense layer with 256 units and a ReLU activation function to the model
    x = tf.keras.layers.Dense(256, activation='relu')(x[0])
    # Now add a final dense layer with 15 units and a softmax activation function, which outputs the probabilities of each of the 15 entity types for each token in the input sequence.
    x = tf.keras.layers.Dense(15, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=[tokens, attention], outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                  loss=[tf.keras.losses.CategoricalCrossentropy()],
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model

# Function to get the final inference_df for the final Kaggle submission
def calculate_preds(tokenizer, target_id_map_updated, MAX_LEN, ROOT_DIR, text_ids, preds, dataset='train', verbose=True):
    """
    Convert token-level predictions into word-level predictions.

    Before making predictions on the validation texts note this - Our model makes label predictions for each token, we need to convert this into a list of word indices for each label.

    Args:
        tokenizer: A tokenizer object used to tokenize the input text.
        target_id_map_updated: A dictionary mapping numerical target labels to their corresponding string labels.
        MAX_LEN: The maximum length of a tokenized input sequence.
        ROOT_DIR: The root directory where the input text files are stored.
        text_ids: A list of IDs corresponding to the input text files.
        preds: A numpy array of predicted labels for each token in the input text files.
        dataset: A string indicating whether the input text files are from the training or test set. Default is 'train'.
        verbose: A boolean indicating whether to print progress updates during function execution. Default is True.

    Returns:
        A pandas DataFrame containing word-level predictions for each input text file. The DataFrame has three columns:
        'id' (the ID of the input text file), 'class' (the predicted label for the word), and 'predictionstring' (a
        space-separated list of the token positions that make up the word).
    """
    all_predictions = []  # to store the word-level predictions.

    # start a loop to iterate through each prediction
    for id_num in range(len(preds)):

        if (id_num % 100 == 0) & (verbose):
            print(id_num, ', ', end='')
        # retrieve the ID of the current input text file based on its index in the text_ids list.
        # Note text_ids = unique_doc_ids[valid_idx]
        file_id_num = text_ids[id_num]

        # Tokenize the input text file and retrieve
        # the positions of the tokens in the original text
        name = ROOT_DIR + f'{dataset}/{file_id_num}.txt'
        txt = open(name, 'r').read()
        tokens = tokenizer.encode_plus(txt, max_length=MAX_LEN, padding='max_length',
                                       truncation=True, return_offsets_mapping=True)
        # See detail noes below for explanation
        offset = tokens['offset_mapping']

        # retrieve the positions of the words in the original text.
        # by iterating through each character in the text and
        # keeping track of spaces and other delimiters
        w = []
        # The variable blank is initialized as True to keep track of whether the current character is a space or not.
        blank = True
        for i in range(len(txt)):  # then loop through each character in the input text file txt
            # see note below for these unicode chars
            if (txt[i] != ' ') & (txt[i] != '\file_id_num') & (txt[i] != '\xa0') & (txt[i] != '\x85') & (blank == True):
                w.append(i)
                blank = False
            elif (txt[i] == ' ') | (txt[i] == '\file_id_num') | (txt[i] == '\xa0') | (txt[i] == '\x85'):
                #  When replicating Python's split() we need to split on two unicode characters '\xa0'
                blank = True
            """ The "If" condition first chexk if the current character is not a space or any of the other delimiters specified (i.e., '\file_id_num', '\xa0', or '\x85') and blank is True. If this condition is met, the code appends the index of the character i to the list w and sets blank to False to indicate that the current character is not a space.

            If the current character is a space or any of the other delimiters specified, the code sets blank to True. This is done to ensure that each word in the text is only recorded once, even if there are multiple spaces between the words. When blank is True, the code does not append the index of the current character to w.
            """

        w.append(1e6)
        # appends a very large number (1e6) to the end of the w list to ensure that any tokens beyond the end of the text are mapped to the last word in the text.
        """ w array stores the starting positions of each word in the text.
        Thus, w_i (declared below) represents the index of the word in w that the current token belongs to.
        For an example text; "The quick brown fox jumps over the lazy dog"
        The w array would store the following starting positions:
        w = [0, 4, 10, 16, 20, 26, 30, 35] """

        # map the positions of the tokens to the positions of the words.
        # `word_map` is a list of indices, where each index represents the position of a word in the original text.
        # the multiplication of -1 will initialize all elements of the word_map array to -1.
        word_map = -1 * np.ones(MAX_LEN, dtype='int32')
        # w_i keeps track of the current index in the w array, which is updated whenever we encounter a new word boundary.
        w_i = 0
        for i in range(len(offset)):
            if offset[i][1] == 0:
                continue
            while offset[i][0] >= w[w_i + 1]:
                w_i += 1
            word_map[i] = int(w_i)

        # See detail notes below for the explanations on above block:
        """ example of what actual form the variable "word_map" and "w_i" may take
        If we hve following inputs:

        MAX_LEN = 20
        offset = [(0, 5), (5, 6), (6, 10), (10, 12), (12, 20)]
        w = [0, 6, 10, 12, 20]

        Then, Here's how the word_map and w_i variables would look like after running the code:

        word_map = [-1, 0, 1, 2, 2, 3, -1, -1, -1, -1, -1, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        w_i = 0

        """

        # convert the token-level predictions into word-level predictions
        # by grouping together adjacent tokens with the same predicted label.

        # 0: B_Lead, 1: I_Lead
        # 2: B_Position, 3: I_Position
        # 4: B_Evidence, 5: Evidence_I
        # 6: CLAIM_B, 7: CLAIM_I
        # 8: CONCLUSION_B, 9: CONCLUSION_I
        # 10: COUNTERCLAIM_B, 11: COUNTERCLAIM_I
        # 12: REBUTTAL_B, 13: REBUTTAL_I
        # 14: NOTHING i.e. O

        pred = preds[id_num, ] / 2.0
        """ This notation is used to access all the elements in the row with index `id_num` of a 2-dimensional NumPy array preds.

        The comma in preds[id_num, ] is used to separate the row index id_num from the column index, which is left blank, indicating that we want to select all the columns in the specified row.

        The resulting 1D NumPy array pred contains the model's prediction probabilities for all the output classes for the sample represented by id_num.

        Dividing the prediction probabilities by 2.0 can be useful in some cases, such as when the model's predictions are too confident and need to be scaled down, or when the prediction probabilities need to be adjusted to balance the model's accuracy and recall.
        """

        # while loop starting with i = 0 is used to iterate through all the token predictions.
        i = 0
        while i < MAX_LEN:
            prediction = []
            # iterates through each token prediction in the pred array
            # where each element in pred corresponds to the predicted
            # class probabilities for a single token in the input text.
            start = pred[i]
            # "start" is the starting index of a predicted word
            # In other words, "start" is the predicted label for the current character, which belongs to one of the eight possible labels: 0, 1, 2, 3, 4, 5, 6, or 7.
            # The if condition is checking if the predicted starting index "start"
            # is one of the eight values in the list [0, 1, 2, 3, 4, 5, 6, 7] coming from target_id_map_updated
            # target_id_map_updated = {0: 'Lead', 1: 'Position', 2: 'Evidence', 3: 'Claim', 4: 'Concluding Statement', 5: 'Counterclaim', 6: 'Rebuttal', 7: 'blank'}
            # These values correspond to the eight possible labels for the starting index of a word in the given dataset.
            # So, if the starting index start has a value in this list, it means that the model
            # has predicted that the current token is the starting token of a word.
            if start in [0, 1, 2, 3, 4, 5, 6, 7]:
                prediction.append(word_map[i])
                i += 1
                if i >= MAX_LEN:
                    break  # check if the index i has exceeded the maximum length of the input text, MAX_LEN. If exceeded, it means that we have processed all the characters in the input text, and we need to exit the loop to avoid any index out of bounds errors.
                while pred[i] == start + 0.5: # See bottom of file for explanation
                    if not word_map[i] in prediction:
                        prediction.append(word_map[i])
                    i += 1
                    if i >= MAX_LEN:
                        break
            # If the starting index start is NOT in this list [0, 1, 2, 3, 4, 5, 6, 7], it means that the model has predicted that the current token is either a continuation of the previous word or is not a part of any word.
            # In this case, the code simply skips to the next token without appending any predictions.
            else:
                i += 1
            #  create a new list, which only contains elements from the ‘prediction’ list that are not equal to -1. And -1 represents that a word/token is not part of any named entity. So, By filtering out these -1 labels, the new ‘prediction’ list will only contain the labels or indices corresponding to actual named entities in the text.
            prediction = [x for x in prediction if x != -1]
            # store the word-level predictions to all_predictions
            # The prediction variable contains the word-level predictions for a given token in the input text. If the length of prediction is less than 5, it means that the token does not have enough context to make a confident prediction about its entity label.
            if len(prediction) > 4:
                all_predictions.append((file_id_num, target_id_map_updated[int(start)],
                                        ' '.join([str(x) for x in prediction])))
            # Recollect that "start" is the starting index of a predicted word
            # In other words, "start" is the predicted label for the current character, which belongs to one of the eight possible labels: 0, 1, 2, 3, 4, 5, 6, or 7.
            # See detail explanations below

    print("all_predictions ", all_predictions)
    df_inference = pd.DataFrame(all_predictions)
    df_inference.columns = ['id', 'class', 'predictionstring']

    # this is the final df that will be submitted to the Kaggle Competition
    return df_inference


"""
1. Explanation of (txt[i] != '\xa0') & (txt[i] != '\x85') -
The Unicode character '\xa0' represents a non-breaking space in the Unicode character set. A non-breaking space prevents the line break at its position. This means that if a line of text wraps to the next line, a non-breaking space will ensure that the text before and after the space will stay together on the same line.

In some programming languages or systems, the '\xa0' character may be represented as a question mark or other symbol, depending on the character encoding used or how it is displayed.

And \x85' represents the "Next Line" control character (NEL) in the Unicode character set. the NEL character can be used to indicate a line break or new paragraph, similar to the newline character '\n', but with different behavior depending on the context in which it is used.

========================================

2. Explanation on `offset = tokens['offset_mapping']`

In NLP, text is tokenized, meaning that it is split into individual words or subwords. The `tokenizer.encode_plus()` function is used to tokenize the input text and obtain a dictionary of various properties, including the `offset_mapping`.

`offset_mapping` is a list of tuples, where each tuple represents the start and end character positions of the corresponding token in the original text. For example, if the token "cat" appears in the text "The cat sat on the mat", the `offset_mapping` for this token would be `(4, 7)`.

In the `calculate_preds()` function, the `offset_mapping` is used to map the positions of the tokens to the positions of the words in the original text. This is done by iterating through the `offset_mapping` and keeping track of the positions of the spaces and other delimiters in the original text.

The resulting `word_map` is a list of indices, where each index represents the position of a word in the original text.

The `word_map` is then used to convert the token-level predictions into word-level predictions. This is done by grouping together adjacent tokens with the same predicted label and mapping them to the corresponding words in the original text using the `word_map`.

========================================

3. Explanations on below:

```
word_map = -1 * np.ones(MAX_LEN, dtype='int32')
        w_i = 0
        for i in range(len(offset)):
            if offset[i][1] == 0:
                continue
            while offset[i][0] >= w[w_i + 1]:
                w_i += 1
            word_map[i] = int(w_i)
```

This block is responsible for mapping the positions of tokens to the positions of words in the original text.

The variable `word_map` is initialized to an array of size MAX_LEN, which represents the maximum number of tokens that can be processed for a given text. Each element in the word_map array corresponds to a token in the text, and its value will be set to the index of the word that the token belongs to.

The variable w_i is initialized to 0, and it keeps track of the current index of the w array, which contains the positions of the words in the original text. The loop iterates over each token in the text, represented by the offset array, which contains the starting and ending positions of each token in the original text. If the ending position of a token is 0, it means that the token is a padding token and should be skipped.

The while loop in the body of the for loop ( while offset[i][0] >= w[w_i + 1]: ) updates the index w_i until the starting position of the current token is less than the ending position of the next word in the w array. This ensures that the current token belongs to the word represented by w_i.

Once the correct index in w is found, the value of w_i is used to set the corresponding element in word_map to the index of the word that the current token belongs to.

========================================

4. Explanation for this block

i = 0
        while i < MAX_LEN:
            prediction = []
            start = pred[i]
            if start in [0, 1, 2, 3, 4, 5, 6, 7]:
                prediction.append(word_map[i])
                i += 1
                if i >= MAX_LEN:
                    break
                while pred[i] == start + 0.5:
                    if not word_map[i] in prediction:
                        prediction.append(word_map[i])
                    i += 1
                    if i >= MAX_LEN:
                        break
            else:
                i += 1
            prediction = [x for x in prediction if x != -1]
            # store the word-level predictions to all_predictions
            if len(prediction) > 4:
                all_predictions.append((file_id_num, target_id_map_updated[int(start)],
                                        ' '.join([str(x) for x in prediction])))

This section of the code implements the logic for extracting the predicted words for each target. The code iterates through each token prediction in the pred array, where each element in pred corresponds to the predicted class probabilities for a single token in the input text.

The while loop starting with i = 0 is used to iterate through all the token predictions. Inside the loop, the code checks whether the current token corresponds to the start of a new word.

If it does, then the code extracts the corresponding word-level prediction by looking for all tokens with the same class prediction (i.e., pred[i]) and appending their corresponding word indices in word_map to a list called prediction. The loop continues until all tokens with the same class prediction have been collected.

After the loop completes, the code checks if the length of the prediction list is greater than 4. If it is, then the code assumes that the predicted words in prediction correspond to a valid answer for the current target, and stores the target ID, the predicted word, and the corresponding file_id_num in the all_predictions list.

The prediction list is also filtered to remove any -1 values, which correspond to tokens that are not associated with any word in the input text (i.e., tokens outside the boundaries of any words in the w array).

Overall, this section of the code implements a simple algorithm for extracting word-level predictions from token-level predictions, which is necessary for evaluating the model's performance on the competition's word-level evaluation metric.


#####################

5. For the line `if start in [0, 1, 2, 3, 4, 5, 6, 7]:` => These values correspond to the eight possible labels for the starting index of a word in the given dataset.

The eight possible labels correspond to the following classes:

target_id_map_updated = {0: 'Lead', 1: 'Position', 2: 'Evidence', 3: 'Claim',
                         4: 'Concluding Statement', 5: 'Counterclaim', 6: 'Rebuttal', 7: 'blank'}

========================================

6. Explantion of the line `while pred[i] == start + 0.5`

It checks if the next character in the prediction array is a continuation of the current token.

The pred array contains the model's predictions for each character in the input text.
Each prediction is a floating point value between 0 and 1 that represents the probability of the character belonging to one of the eight possible labels.

If the value of pred[i] is equal to start + 0.5, it means that the current token has not ended and the next character belongs to the same token.

"start" is the label assigned to the first character of the current token.

In other words, "start" is the predicted label for the current character, which belongs to one of the eight possible labels: 0, 1, 2, 3, 4, 5, 6, or 7.

Therefore, if pred[i] is equal to start + 0.5, it means that the model has predicted the continuation of the current token.

Inside the while loop, the code appends the word index of the current token to the prediction list if it has not already been added. Then, it increments the index i to move to the next character in the input text.

This process continues as long as the next character in the prediction array belongs to the current token.

The while loop exits when the next character in the prediction array does not belong to the current token or the index i exceeds the maximum length of the input text.

For example, let's say the model predicts the labels for the tokens/chracters in a given text as follows:

predicted_labels = [0, 0, 0.5, 1, 1.5, 1.5, 2, 3, 3, 0, 0, 4, 4.5, 4.5, 4.5, 4.5, 5, 6, 6, 7, 7]

Here, the character at position 2 has a predicted label of 0.5, indicating that it is the continuation of the first word. Similarly, the character at position 4 has a predicted label of 1.5, indicating that it is the continuation of the second word.

The while loop in the code block checks if the current character is part of a word or a phrase by looking at the predicted label for that character. If the predicted label is start + 0.5, then the loop continues, indicating that the current character is part of the same word or phrase as the previous character. If the predicted label is not start + 0.5, the loop breaks and the word or phrase is complete.

In the above example, the while loop will execute for the following characters:

For character at position 2: start = 0, pred[i] = 0.5, loop continues.
For characters at positions 4 and 5: start = 1, pred[i] = 1.5, loop continues.
For characters at positions 12, 13, 14, and 15: start = 4, pred[i] = 4.5, loop continues.

The while loop stops when the predicted label for the current character is not equal to start + 0.5

===================================================================

7. Explain while pred[i] == start + 0.5:

In this line, `start` represents the predicted label for the current token (scaled down by a factor of 2.0). The purpose of adding 0.5 to `start` is to check whether the next token in the sequence has a label that corresponds to the continuation of the same target class as the current token.

For example, if the original labels in the `preds` array were integers (e.g., 0, 1, 2, 3), after dividing by 2.0, you would get values like 0.0, 0.5, 1.0, 1.5. In this case, if the current token has a label of 0.0 (start), adding 0.5 to it will result in 0.5. The while loop will continue as long as the next token has a label of 0.5, indicating that the next token is a continuation of the same target class (in this example, class 0).

This approach is particularly useful when dealing with multi-token entities (e.g., named entities) in NLP tasks. By checking for `pred[i] == start + 0.5`, the code ensures that it gathers all consecutive tokens belonging to the same entity before moving on to the next entity or non-entity token.


==================================================================

8.  The reason of this line `if len(prediction) > 4:`

Its used to filter out predictions that have less than 5 word-level predictions. The prediction variable contains the word-level predictions for a given token in the input text. If the length of prediction is less than 5, it means that the token does not have enough context to make a confident prediction about its entity label.

By setting this threshold, the code ensures that only predictions with sufficient context are stored in all_predictions. This can help avoid incorrect or unreliable predictions that may be made based on incomplete information.

"""
