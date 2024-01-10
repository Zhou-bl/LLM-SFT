import huggingface_hub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, AdamW, get_scheduler
from datasets import load_dataset
import json
import dataprocess.dataformatter as dataformatter
from torch.utils.data import DataLoader
import argparse
import tqdm
import deepspeed
import matplotlib.pyplot as plt

def main():

    # Load Llama2-hf model from the local cache:
    # path = /cpfs01/shared/LLMAlignment/huggingface/models/llama-7b-hf

    if args.model_name == "llama":
        model_path = "/cpfs01/shared/LLMAlignment/huggingface/models/llama-7b-hf"
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
        #tokenizer = LLaMATokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_path2 = "/cpfs01/user/juzhaoxun/DL/save/epoch_0"
        model = LlamaForCausalLM.from_pretrained(model_path2, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        if tokenizer.pad_token is None:
            print("Add pad token to tokenizer")
            tokenizer.add_special_tokens({'pad_token': "<pad>"})
            model.resize_token_embeddings(len(tokenizer))
    elif args.model_name == "baichuan":
        model_path = "/cpfs01/user/juzhaoxun/baichuan"
        #model_path = "baichuan-inc/Baichuan-7B"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        if tokenizer.pad_token is None:
            print("Add pad token to tokenizer")
            tokenizer.add_special_tokens({'pad_token': "<pad>"})
            model.resize_token_embeddings(len(tokenizer))
        

    #dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
    # input_text = "How many people live in New York City?"
    # input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
    # output_ids = model.generate(input_ids, max_length=50, do_sample=True)
    # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # print(output_text)
    data_path = "/cpfs01/user/juzhaoxun/DL/data/splited_data.json"
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]
    raw_data = raw_data
    formatted_data = dataformatter.format_all(raw_data, tokenizer, args.max_length)
    train_data = dataformatter.batch_data(tokenizer, formatted_data, args.batch_size)
    # get 1/3 data for training
    train_data = train_data[:len(train_data) // 3]
    # print(train_data[0])
    # print(train_data[0]["input_ids"].shape)

    # train the model:

    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_training_steps = args.epoch * len(train_data)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print("Training steps:", num_training_steps)
    model.train()
    loss_step = []
    loss_epoch = []
    for epoch in range(args.epoch):
        epoch_loss = 0
        print("Begin epoch {}".format(epoch))
        for batch in tqdm.tqdm(train_data):
            #print(batch["input_ids"].shape)
            batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            epoch_loss += loss.item()
            print("[lr]:", optimizer.param_groups[0]["lr"])
            print("[loss]:", loss.item())
            loss_step.append(loss.item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        loss_epoch.append(epoch_loss / len(train_data))
        print("Epoch {} finished".format(epoch + 1))
        # save model:
        model.save_pretrained("save/"+args.model_name+"/{}".format(epoch))
    plt.figure("epoch loss")
    plt.plot(loss_epoch)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("epoch loss")
    plt.savefig("save/epoch_loss.png")

    plt.figure("step loss")
    plt.plot(loss_step)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("step loss")
    plt.savefig("save/step_loss.png")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--local_rank", type=int, default=0)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

if __name__ == "__main__":
    main()