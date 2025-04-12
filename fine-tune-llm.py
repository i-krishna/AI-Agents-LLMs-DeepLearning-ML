# Fine-tuning Pre-Trained LLM

## Import necessary libraries

!pip install datasets evaluate transformers peft

from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np

## Dataset

# how dataset was generated
from datasets import load_dataset
# load imdb data
imdb_dataset = load_dataset("imdb")

# define subsample size
N = 1000 
# generate indexes for random subsample
rand_idx = np.random.randint(24999, size=N) 

# extract train and test data
x_train = imdb_dataset['train'][rand_idx]['text']
y_train = imdb_dataset['train'][rand_idx]['label']

x_test = imdb_dataset['test'][rand_idx]['text']
y_test = imdb_dataset['test'][rand_idx]['label']

# create new dataset
dataset = DatasetDict({'train':Dataset.from_dict({'label':y_train,'text':x_train}),
                              'validation':Dataset.from_dict({'label':y_test,'text':x_test})})
