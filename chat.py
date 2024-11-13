#!/usr/bin/env python3

import transformers, torch, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# model
lama_size = '8B'
device_map={ '': 'cuda:7' } # i.e. put entire model on GPU 7
model_path = f'/home1/shared/Models/Llama3/Llama-3.1-{lama_size}-Instruct'

def read_json_file(settings_file='settings.json'):
  """Read generation and other parameters"""

  with open(settings_file, 'r') as file:
    data = json.load(file)
  return data

def main():
  """Ask for input and feed into llama2"""

  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type= 'nf4')

  tokenizer = AutoTokenizer.from_pretrained(model_path)

  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quant_config,
    device_map=device_map)

  generator = transformers.pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    pad_token_id=tokenizer.eos_token_id)

  gen_params = read_json_file()

  conversation = \
    [{'role': 'system', 'content': 'You are a helpful assistant.'}]

  while True:

    user_input = input('\nPrompt: ')
    conversation.append({'role': 'user', 'content': user_input})

    output = generator(
      conversation,
      do_sample=gen_params['do_sample'],
      temperature=gen_params['temperature'],
      top_p=gen_params['top_p'],
      max_new_tokens=gen_params['max_new_tokens'])

    conversation = output[0]['generated_text']
    print('conversation:', conversation)

if __name__ == "__main__":

  main()
