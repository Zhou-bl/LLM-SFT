import json
from typing import Dict, Sequence, Optional
from transformers import AutoTokenizer, AutoModel, LlamaTokenizer, LlamaForCausalLM
import transformers
from tqdm import tqdm

# load data
data_path = "../data/splited_data.json"
# read line by line
with open(data_path, "r") as f:
    data = [json.loads(line) for line in f]

print(len(data))
length_list = []
cnt=0
for i in range(len(data)):
    length_list.append(len(data[i]["data"]))
    if length_list[-1]==2:
        cnt+=1
print(cnt)
print(length_list[:100])