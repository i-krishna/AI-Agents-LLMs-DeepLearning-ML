#!/usr/bin/env python3

import transformers, torch, os, numpy, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time

sys.path.append('../Lib/')

lama_size = '7b'
drbench_dev_path = 'DrBench/Csv/summ_0821_dev.csv'
model_path = f'/home1/shared/Models/Llama2/Llama-2-{lama_size}-chat-hf'

print(model_path)

# if '7b' in model_path:
#   os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# elif '13b' in model_path:
#   os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# elif '70b' in model_path:
#   os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

def main():
  """Ask for input and feed into llama2"""

  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    load_in_8bit=True)
  pipeline = transformers.pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map='auto')

  prompt_text = "do something: "

  generated_outputs = pipeline(
    prompt_text,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    temperature=0.001,
    max_length=512)

  print('************************************************\n')
  print(generated_outputs[0]['generated_text'])

if __name__ == "__main__":

  base_path = os.environ['DATA_ROOT']
  main()
