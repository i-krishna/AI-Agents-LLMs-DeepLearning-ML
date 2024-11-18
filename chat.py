#!/usr/bin/env python3

import transformers, torch, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# model
device_map={'': 'cuda:0' } # i.e. put entire model on GPU 1
settings_file='settings.json'

def read_json_file(settings_json_file):
  """Read generation and other parameters"""

  with open(settings_json_file, 'r') as file:
    data = json.load(file)
  return data

def main():
  """Ask for input and feed into llama2"""

  settings = read_json_file(settings_file)

  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type= 'nf4')

  tokenizer = AutoTokenizer.from_pretrained(settings['model_path'])

  model = AutoModelForCausalLM.from_pretrained(
    settings['model_path'],
    quantization_config=quant_config,
    device_map=device_map)

  generator = transformers.pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    pad_token_id=tokenizer.eos_token_id)

  conversation = [{'role': 'system', 'content': settings['sys_prompt']}]

  while True:

    user_input = input('\n>>> ')
    conversation.append({'role': 'user', 'content': user_input})

    output = generator(
      conversation,
      do_sample=settings['do_sample'],
      temperature=settings['temperature'],
      top_p=settings['top_p'],
      max_new_tokens=settings['max_new_tokens'])

    conversation = output[0]['generated_text']
    print('\n' + conversation[-1]['content'])

if __name__ == "__main__":

  main()
