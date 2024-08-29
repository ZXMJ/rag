import requests
import json
import pandas as pd
import argparse
import collections
import json
import numpy as np
import re
import string
import sys
import math
import os
import time
from textwrap import dedent
from PIL import Image
import json
import re
import pandas as pd
from nltk.translate import meteor_score as ms
from rouge_score import rouge_scorer
from bs4 import BeautifulSoup
import requests
import nltk
import gc
import torch
from nltk.translate import bleu_score
import numpy as np
from simhash import Simhash
from bleurt import score
import string
import collections
import matplotlib
import difflib
from datasets import load_dataset


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer1 = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
# def get_prompt(context, question):
#     text = f"""    
#     Answer the following question using only information from the article,give the answer to the question, don't just give True or False. If there is no good answer in the article, say \"I don'\''t know\".\n\n ```: Article: {context} 
#     \n\nQuestion: {question}\nAnswer:```
#     """
#     return text

# import torch
# import re


# def process_squad(context, question):
#     # 确定设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 将模型移到设备
#     model.to(device)

#     # 创建输入提示
#     prompt = get_prompt(context.replace('`', ''), question)

#     # 使用 tokenizer 对输入进行编码并将其移到设备
#     inputs = tokenizer1(prompt, return_tensors='pt').to(device)

#     # 使用模型进行推理
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs['input_ids'],
#             max_new_tokens=50,  # 最大生成词数
#             min_new_tokens=1,  # 最小生成词数
#             do_sample=False  # 贪心解码
#         )

#     # 解码生成的文本
#     generated_text = tokenizer1.decode(outputs[0], skip_special_tokens=True)
#     print(generated_text)

#     # 去除多余空格和换行符
#     result = re.sub(r"\s+", " ", generated_text).strip()

#     # 返回处理后的结果
#     return result

import torch
import re

def get_prompt(context, question):
    text = f"""
    Answer the following question using only information from the article.
    Give a detailed answer to the question; do not just say True or False.
    If there is no good answer in the article, say "I don't know."

    Article: {context}

    Question: {question}

    Answer:
    """
    return text

def process_squad(context, question):
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model_name = "t5-base"

    
    # Move model to device
    model.to(device)

    # Create input prompt
    prompt = get_prompt(context.replace('`', ''), question)

    # Encode inputs and move to device
    inputs = tokenizer1(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

    # Generate output using the model
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=50,  # Maximum number of tokens to generate
            min_new_tokens=1,  # Minimum number of tokens to generate
            do_sample=True,  # Enable sampling
            top_k=50,  # Consider the top 50 tokens by probability
            top_p=0.95,  # Use nucleus sampling
            num_beams=5  # Use beam search with 5 beams
        )

    # Decode generated text
    generated_text = tokenizer1.decode(outputs[0], skip_special_tokens=True)
    print("Generated Text:", generated_text)

    # Remove extra spaces and line breaks
    result = re.sub(r"\s+", " ", generated_text).strip()

    # Return processed result
    return result




context = """
乔治·华盛顿是美国的第一任总统，他于1789年担任总统。他被认为是美国的“国父”之一。
"""
question = "乔治·华盛顿是谁，他有什么成就？"
result = process_squad(context, question)
print(result)
context = """
The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.
"""
question = "In what country is Normandy located?"
result = process_squad(context, question)
print(result)
