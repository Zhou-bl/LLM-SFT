import huggingface_hub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, AdamW, get_scheduler
from datasets import load_dataset
import json
from torch.utils.data import DataLoader
import argparse
import tqdm
import deepspeed
import matplotlib.pyplot as plt
import re

def get_new_output(text, tokenizer):
    tokenized = tokenizer(text, return_tensors="pt")
    # replace the pad_token_id with whitespace to make the output more readable:

def clean_prompt(prompt):
    # 去除非ASCII字符
    prompt = re.sub(r'[^\x00-\x7F]+', ' ', prompt)

    # 去除特殊字符，只保留字母、数字、空格、标点符号
    prompt = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', prompt)

    # 去除多余的空格
    prompt = ' '.join(prompt.split())

    return prompt

def main():
    model_path = "/cpfs01/shared/LLMAlignment/huggingface/models/llama-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 去掉输出中的 '<PAD>'：
    tokenizer.add_special_tokens({'pad_token': "<pad>"})
    json_path = "/cpfs01/user/juzhaoxun/DL/evaluation/result/llama/epoch1_model.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    # print(data[0])
    new_json_list = []
    cnt =0
    
    
    with open("/cpfs01/user/juzhaoxun/DL/evaluation/test.json", "w", encoding='utf-8') as f:
        json.dump(json_file, f, ensure_ascii=True, indent=4)

if __name__ == "__main__":
    main()
        