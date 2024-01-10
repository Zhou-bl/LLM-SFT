import transformers
from transformers import AutoTokenizer, AutoModel, LlamaTokenizer, LlamaForCausalLM
import json
import tqdm
import torch

def get_special_token(tokenizer, flag):
    # flag=0: start token id
    # flag=1: end token id
    if flag == 0:
        return "\n"
    if flag == 1:
        return tokenizer.eos_token

IGNORE_INDEX = -100
def format_example(data, tokenizer, max_length):
    """
    attention: data is a list of sentences.
    The user and model content emerage alternatively.
    The first sentence is user's sentence.
    """
    cnt=0
    end_token = get_special_token(tokenizer, 1)
    start_token = get_special_token(tokenizer, 0)
    bos_token = tokenizer.bos_token
    tags = []
    labels = []
    tokenized_ids = []
    for i in range(len(data)):
        if i % 2 == 0:
            tags.append("User")
        else:
            tags.append("Assistant")
    for index, sentence in enumerate(data):
        new_sentence = tags[index] + ": " + sentence + end_token
        if index % 2 == 0:
            # User:
            if index == 0:
                new_sentence = bos_token + new_sentence
            else:
                new_sentence = start_token + new_sentence
            tokenized_sentence = tokenizer(new_sentence, add_special_tokens=False)
            tokenized_ids += tokenized_sentence["input_ids"]
            labels += ([IGNORE_INDEX] * len(tokenized_sentence["input_ids"]))
        else:
            # Model:
            # First is the prompt:
            input_sentence = start_token + tags[index] + ": "
            tokenized_input_sentence = tokenizer(input_sentence, add_special_tokens=False)
            tokenized_ids += (tokenized_input_sentence["input_ids"])
            labels += ([IGNORE_INDEX] * len(tokenized_input_sentence["input_ids"]))
            # Then the response:
            generate_sentence = sentence + end_token
            tokenized_generate_sentence = tokenizer(generate_sentence, add_special_tokens=False)
            tokenized_ids += (tokenized_generate_sentence["input_ids"])
            labels += (tokenized_generate_sentence["input_ids"])
    assert len(tokenized_ids) == len(labels)
    # truncate the tokenized_ids and labels
    if len(tokenized_ids) > max_length:
        tokenized_ids = tokenized_ids[:max_length]
        labels = labels[:max_length]
    return {"input_ids": torch.tensor(tokenized_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}

def format_all(raw_data, tokenizer, max_length):
    """
    Format the data into the format that can train the model
    """
    reformat_data = []
    """
    every item in reformat_data list is a dict:
    {"input_ids": torch.tensor, "labels": torch.tensor}
    attention that: both tensor are long tensor
    """
    #cnt = 0
    print("hhello:", raw_data[0])
    for item in tqdm.tqdm(raw_data):
        ids = item["id"]
        sample_data = item["data"]
        reformat_data.append(format_example(sample_data, tokenizer, max_length))
        # if cnt==0:
        #     print("ids:", ids)
        #     print("data:", sample_data)
        #     print(reformat_data[-1])
        # cnt += 1

    return reformat_data

def batch_data(tokenizer, data, batch_size):
    # data is a list of dict
    # every dict has two keys: input_ids and labels
    # batch_data is a list of dict
    # every dict has three keys: input_ids, labels and attention_mask
    res = []
    cur_index = 0
    while cur_index < len(data):
        input_ids, labels = [], []
        l = cur_index
        r = min(cur_index + batch_size, len(data))
        for i in range(l, r):
            input_ids.append(data[i]["input_ids"])
            labels.append(data[i]["labels"])
        cur_index = r
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        assert input_ids.shape == labels.shape
        assert input_ids.shape == attention_mask.shape
        res.append({"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask})
    return res

def main():
    model_path = "/cpfs01/shared/LLMAlignment/huggingface/models/llama-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("hello hello")
    print(tokenizer.pad_token)
    data_path = "/cpfs01/user/juzhaoxun/DL/data/splited_data.json"
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]
    print(raw_data[0])
    print(raw_data[1])
    print(raw_data[2])
    formatted_data = format_all(raw_data, tokenizer)
    print(formatted_data[0])

if __name__ == "__main__":
    main()