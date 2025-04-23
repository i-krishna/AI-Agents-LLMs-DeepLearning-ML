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

# Loading DeepSeek R1 and the Tokenizer using FastLanguageModel.from_pretrained()

# Set key parameters
max_seq_length = 2048 # Define the maximum sequence length a model can handle (i.e. how many tokens can be processed at once)
dtype = None # Set to default 
load_in_4bit = True # Enables 4 bit quantization (Compression) — a memory saving optimization 

# Load the DeepSeek R1 model and tokenizer using unsloth — imported using: from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",  # Load the pre-trained DeepSeek R1 model (8B parameter version)
    max_seq_length=max_seq_length, # Ensure the model can process up to 2048 tokens at once
    dtype=dtype, # Use the default data type (e.g., FP16 or BF16 depending on hardware support)
    load_in_4bit=load_in_4bit, # Load the model in 4-bit quantization to save memory
    token=hugging_face_token, # Use hugging face token
)
