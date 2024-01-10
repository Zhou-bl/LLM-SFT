from openai import OpenAI
import tqdm
import os
import numpy as np
import argparse
import time
import json
from datasets import load_dataset

client = OpenAI(
    api_key="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",
)

def get_en_to_ch(prompt, model="gpt-3.5-turbo-instruct"):
    prompt = "You are a English-Chinese translator, and now you need to translate the following English text into Chinese:\n" + prompt + "\nChinese translation:"
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
      	model=model,
      	messages=messages,
      	temperature=0.7,
    )
    return response.choices[0].message.content

def get_ch_to_en(prompt, model="gpt-3.5-turbo"):
    prompt = "You are a Chinese-English translator, and now you need to translate the following Chinese text into English:\n" + prompt + "\nEnglish translation:"
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
      	model=model,
      	messages=messages,
      	temperature=0.7,
    )
    return response.choices[0].message.content

def translate_wrapper(task, text):
    #if task == "en-ch":
    flag = 0
    max_retry_cnt = 0
    break_flag = 0
    base_sleep_time = 0.1
    while flag == 0:
        if break_flag: break
        try:
            if task == "en-ch":
                response = get_en_to_ch(text)
            elif task == "ch-en":
                response = get_ch_to_en(text)
            else:
                raise ValueError("The task must be \'en-ch\' or \'ch-en\'!")
            flag = 1
        except Exception as e:
            if str(e) == "The task must be \'en-ch\' or \'ch-en\'!":
                raise ValueError("The task must be \'en-ch\' or \'ch-en\'!")
            print(e)
            #time.sleep(base_sleep_time)
            max_retry_cnt += 1
            if max_retry_cnt == 500:
                print("The api key may be invalid.")
                print("Please check the api key.")
                return ""
    return response

def main():
    print(get_ch_to_en("你好, 我是来自上海交通大学的一名学生。"))
if __name__ == "__main__":
    main()