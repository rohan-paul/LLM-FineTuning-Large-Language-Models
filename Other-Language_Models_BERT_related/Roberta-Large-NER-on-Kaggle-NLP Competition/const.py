import os
import torch

# You can run the Notebook `Pytorch-Roberta_Large.ipynb`  either Locally or in Kaggle - Just modify the 'ROOT_DIR' variable to properly refer to the dataset
# ROOT_DIR = '../input/feedback-prize-2021/' # Kaggle
ROOT_DIR = '../input/' # local

# MODEL_NAME = 'roberta-large'
MODEL_NAME = 'roberta-base'

MODEL_PATH = 'model'

RUN_NAME = f"{MODEL_NAME}"

MAX_LEN = 512

DOC_STRIDE = 128

config = {'train_batch_size': 4,
          'valid_batch_size': 1,
          'epochs': 5,
          'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
          'max_grad_norm': 10,
          'device': 'cuda' if torch.cuda.is_available() else 'cpu',
          'model_name': MODEL_NAME,
          'max_length': MAX_LEN,
          'doc_stride': DOC_STRIDE,
          }

# Note in above, I have 5 Learning rates for 5 epochs

output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim',
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

LABELS_TO_IDS = {v:k for k,v in enumerate(output_labels)}
IDS_TO_LABELS = {k:v for k,v in enumerate(output_labels)}