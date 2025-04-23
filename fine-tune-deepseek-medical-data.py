# Fine tune DeepSeek-R1 on Medical COT Dataset. 
# Medical reasoning and clinical case analysis using LoRA (Low-Rank Adaptation) with Unsloth.

# Ref: https://huggingface.co/develops20/DeepSeek-R1-Distill-Llama-8B-Medical-COT 

# Create API in Hugging_Face & weights_and_biases. Link them to Kaggle Notebook. 
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hugging_face_token = user_secrets.get_secret("HF") # Hugging_Face_Token
wnb_token = user_secrets.get_secret("wnl") #weights_and_biases_token

# install unsloth for efficient fine-tuning and inference for LLMs
# !pip install unsloth 
# !pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git 

# Modules for fine-tuning
from unsloth import FastLanguageModel
import torch # Import PyTorch
from trl import SFTTrainer # Trainer for supervised fine-tuning (SFT)
from unsloth import is_bfloat16_supported # Checks if the hardware supports bfloat16 precision

# Hugging Face modules
from huggingface_hub import login # Lets you login to API
from transformers import TrainingArguments # Defines training hyperparameters
from datasets import load_dataset # Lets you load fine-tuning datasets

# Import weights and biases
import wandb
# Import kaggle secrets
from kaggle_secrets import UserSecretsClient

# Initialize Hugging Face & WnB tokens
user_secrets = UserSecretsClient() # from kaggle_secrets import UserSecretsClient
hugging_face_token = user_secrets.get_secret("HF")
wnb_token = user_secrets.get_secret("wnl")

# Login to Hugging Face
login(hugging_face_token) # from huggingface_hub import login

# Login to WnB
wandb.login(key=wnb_token) # import wandb
run = wandb.init(
    project='Fine-tune-DeepSeek-R1-Distill-Llama-8B on Medical COT Dataset', 
    job_type="training", 
    anonymous="allow"
)
