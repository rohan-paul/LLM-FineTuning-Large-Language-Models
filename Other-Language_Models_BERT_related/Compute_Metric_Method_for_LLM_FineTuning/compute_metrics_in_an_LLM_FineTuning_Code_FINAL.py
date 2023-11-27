from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset, DatasetDict

tokenized_dataset = DatasetDict.load_from_disk(args.dataset_dir)
# Metric
metric = evaluate.load("rouge")

pad_tok = 50256

token_id="Salesforce/instructcodet5p-16b"

tokenizer = AutoTokenizer.from_pretrained(token_id)

###############################
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    for idx in range(len(preds)):
        for idx2 in range(len(preds[idx])):
            if preds[idx][idx2]==-100:
                preds[idx][idx2] = 50256

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != pad_tok, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)


    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # The results of the Rouge metric are then multiplied by 100 and rounded to four decimal places.

    result = {k: round(v * 100, 4) for k, v in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

    result["gen_len"] = np.mean(prediction_lens)

    return result