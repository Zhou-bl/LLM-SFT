import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, AdamW, get_scheduler
import argparse
import torch
import tqdm
import json

def get_output(model, tokenizer, input):
    # input is a list of strings
    input_ids = tokenizer(input, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    output_ids = model.generate(input_ids, max_length=1024, do_sample=True)
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return output

def load_model(model_path):
    if args.model_name == "llama":
        tokenizer = LlamaTokenizer.from_pretrained("/cpfs01/shared/LLMAlignment/huggingface/models/llama-7b-hf", trust_remote_code=True)
        #tokenizer = LLaMATokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        if tokenizer.pad_token is None:
            print("Add pad token to tokenizer")
            tokenizer.add_special_tokens({'pad_token': "<pad>"})
            model.resize_token_embeddings(len(tokenizer))
    elif args.model_name == "baichuan":
        token_path = "/cpfs01/user/juzhaoxun/baichuan"
        #model_path = "baichuan-inc/Baichuan-7B"
        tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        if tokenizer.pad_token is None:
            print("Add pad token to tokenizer")
            tokenizer.add_special_tokens({'pad_token': "<pad>"})
            model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def batch_data(data, batch_size):
    batched_data = []
    for i in range(0, len(data), batch_size):
        batched_data.append(data[i:i+batch_size])
    return batched_data

def main():
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    #print("raw_eval_set:", raw_eval_set)
    
    json_list = []
    # 截取dataset的前15个
    # eval_set = batch_data(eval_set, args.batch_size)
    # print("batch_set", eval_set[0])
    # print("batch_set", eval_set[1])
    # print("batch_set", eval_set[2])
    model, tokenizer = load_model(args.model_path)
    for example in tqdm.tqdm(eval_set):
        #print("example:", example)
        #generate here is a placeholder for your models generations
        instruction = example["instruction"]
        try:
            output = get_output(model, tokenizer, instruction)
        except:
            print("[Warning] get_output failed, instruction:", instruction)
            output = ""
        json_list.append({"instruction": instruction, "output": output})
    
    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(json_list, ensure_ascii=False, indent=4))
        
argparser = argparse.ArgumentParser()
argparser.add_argument("--model_path", type=str, required=True)
argparser.add_argument("--output_path", type=str, required=True)
argparser.add_argument("--model_name", type=str, required=True)
args = argparser.parse_args()

if __name__ == "__main__":
    main()