#!/usr/bin/env python3

import transformers, torch, os
from sympy.physics.units import temperature
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# model
lama_size = '8B'
device_map={ '': 'cuda:7' } # i.e. put entire model on GPU 7
model_path = f'/home1/shared/Models/Llama3/Llama-3.1-{lama_size}-Instruct'

# configuration
do_sample = False
t = 1.0
top_p = 1
max_new_tokens = 100

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
    device_map='auto')

  while True:

    user_input = input('\nPrompt: ')
    prompt = [{'role': 'system', 'content': 'You are a helpful assistant.'},
              {'role': 'user', 'content': user_input}]

    generated_outputs = generator(
      prompt,
      do_sample=do_sample,
      temperature=t,
      top_p=top_p,
      max_new_tokens=max_new_tokens)

    print(generated_outputs[0]['generated_text'][2]['content'])

if __name__ == "__main__":

  main()
