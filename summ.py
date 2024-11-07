#!/usr/bin/env python3

import transformers, torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM

lama_size = '8B'
model_path = f'/home1/shared/Models/Llama3/Llama-3.1-{lama_size}-Instruct'

# if '8B' in model_path:
#   os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# elif '13b' in model_path:
#   os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# elif '70b' in model_path:
#   os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

def main():
  """Ask for input and feed into llama2"""

  prompt = [{"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
            {"role": "user", "content": "What's Deep Learning?"}]

  tokenizer = AutoTokenizer.from_pretrained(model_path)

  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto')

  generator = transformers.pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map='auto')

  generated_outputs = generator(
    prompt,
    do_sample=False,
    temperature=1.0,
    top_p=1,
    max_new_tokens=50)

  print(generated_outputs[0]['generated_text'])

if __name__ == "__main__":

  main()
