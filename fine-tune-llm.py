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
from datasets import Dataset, load_dataset, DatasetDict

import numpy as np

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
train_dataset = Dataset.from_dict({'label': y_train, 'text': x_train})
validation_dataset = Dataset.from_dict({'label': y_test, 'text': x_test})

# create DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset
})

# load dataset
dataset = load_dataset('shawhin/imdb-truncated')
dataset

# display % of training data with label=1
np.array(dataset['train']['label']).sum()/len(dataset['train']['label'])

## Model

from transformers import AutoModelForSequenceClassification
model_checkpoint = 'distilbert-base-uncased'
# model_checkpoint = 'roberta-base' # you can alternatively use roberta-base but this model is bigger thus training will take longer

# define label maps
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative":0, "Positive":1}

# generate classification model from model_checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

# display architecture
model
