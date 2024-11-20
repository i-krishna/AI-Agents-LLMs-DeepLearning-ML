#!/usr/bin/env python3

import transformers, torch, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# model
device_map1 = {'': 'cuda:0'}
device_map2 = {'': 'cuda:1'}
settings_file = 'settings.json'

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

  tokenizer1 = AutoTokenizer.from_pretrained(settings['model_path'])
  tokenizer2 = AutoTokenizer.from_pretrained(settings['model_path'])

  model1 = AutoModelForCausalLM.from_pretrained(
    settings['model_path'],
    quantization_config=quant_config,
    device_map=device_map1)
  generator1 = transformers.pipeline(
    task='text-generation',
    model=model1,
    tokenizer=tokenizer1,
    torch_dtype=torch.bfloat16,
    device_map=device_map1,
    pad_token_id=tokenizer1.eos_token_id)

  model2 = AutoModelForCausalLM.from_pretrained(
    settings['model_path'],
    quantization_config=quant_config,
    device_map=device_map2)
  generator2 = transformers.pipeline(
    task='text-generation',
    model=model2,
    tokenizer=tokenizer2,
    torch_dtype=torch.bfloat16,
    device_map=device_map1,
    pad_token_id=tokenizer2.eos_token_id)

  conversation1 = [{'role': 'system', 'content': 'You read what is said to you and comment on it. Your comments are pretty short.'}]
  conversation2 = [{'role': 'system', 'content': 'You read what is said to you and comment on it. Your comments are pretty short.'}]

  init_prompt = 'What is the meaning of life?'
  conversation1.append({'role': 'user', 'content': init_prompt})
  conversation2.append({'role': 'user', 'content': init_prompt})

  while True:

    output1 = generator1(
      conversation1,
      do_sample=settings['do_sample'],
      temperature=settings['temperature'],
      top_p=settings['top_p'],
      max_new_tokens=settings['max_new_tokens'])

    conversation1 = output1[0]['generated_text']
    print('\nA:' + conversation1[-1]['content'])

    response = conversation1[-1]['content']
    conversation2.append({'role': 'user', 'content': response})

    output2 = generator1(
      conversation2,
      do_sample=settings['do_sample'],
      temperature=settings['temperature'],
      top_p=settings['top_p'],
      max_new_tokens=settings['max_new_tokens'])

    conversation2 = output2[0]['generated_text']
    print('\nB:' + conversation2[-1]['content'])


if __name__ == "__main__":

  main()
