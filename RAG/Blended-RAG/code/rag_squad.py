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
import warnings

warnings.filterwarnings('ignore')
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.autograph.set_verbosity(3)

import warnings

warnings.filterwarnings('ignore')

# import nltk

# # 下载 punkt 数据包
# nltk.download('punkt')

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError

es_client = Elasticsearch(
    ["http://10.224.160.239:9200"]
)
# 示例操作
try:
    # 尝试执行一个简单的请求
    response = es_client.info()
    print(response)
except RequestError as e:
    print(f"RequestError: {e.info}")
except Exception as e:
    print(f"An error occurred: {e}")


## Downloading methods

from sentence_transformers import SentenceTransformer, util

model1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model2 = BertModel.from_pretrained("bert-base-uncased")

from transformers import AutoTokenizer, AutoModel

# 加载模型和分词器
# 你的访问令牌

# access_token = "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"

# 加载分词器和模型
# Load model directly
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer1 = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
# tokenizer1 = AutoTokenizer.from_pretrained(
#     "meta-llama/Meta-Llama-3.1-8B-Instruct",
#     trust_remote_code=True,
#     use_auth_token=access_token  # 使用访问令牌
# )

# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Meta-Llama-3.1-8B-Instruct",
#     trust_remote_code=True,
#     use_auth_token=access_token
# ).half().cuda()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def bert_score(reference, candidate, return_similarity_matrix=False):
    # Load the BERT tokenizer and model
    # Tokenize the input text
    ref_tokens = tokenizer(reference, return_tensors="pt", add_special_tokens=False)
    can_tokens = tokenizer(candidate, return_tensors="pt", add_special_tokens=False)
    # Get the BERT embeddings
    model2.eval()
    with torch.no_grad():
        ref_outputs = model2(**ref_tokens)
        ref_embeddings = ref_outputs.last_hidden_state[0]
        can_outputs = model2(**can_tokens)
        can_embeddings = can_outputs.last_hidden_state[0]
    # Compute cosine similarities
    cosine_similarities = np.zeros((can_embeddings.shape[0], ref_embeddings.shape[0]))
    for i, c in enumerate(can_embeddings):
        for j, r in enumerate(ref_embeddings):
            cosine_similarities[i, j] = cosine_similarity(c, r)
    # Align cosine similarities
    max_similarities = cosine_similarities.max(axis=1)
    # Average similarity scores
    bertscore = max_similarities.mean()
    if return_similarity_matrix:
        return bertscore, cosine_similarities
    else:
        return bertscore


def sentence_similarity(ideal_answer, generated_answer):
    embedding_1 = model1.encode(ideal_answer, convert_to_tensor=True)
    embedding_2 = model1.encode(generated_answer, convert_to_tensor=True)
    sim_score = util.pytorch_cos_sim(embedding_1, embedding_2)
    sim_score = sim_score.cpu().numpy()[0][0]  # Move tensor to CPU and then convert to NumPy array
    return sim_score


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def Sim_hash(ideal_answer, generated_answer):
    return Simhash(generated_answer).distance(Simhash(ideal_answer))


def calculate_perplexity(ideal_answer, answer):
    answer_tokens = answer.strip().split()
    ideal_tokens = ideal_answer.strip().split()

    # Build a frequency distribution of ideal tokens
    token_frequency = {}
    total_tokens = 0
    for token in ideal_tokens:
        token_frequency[token] = token_frequency.get(token, 0) + 1
        total_tokens += 1

    # Calculate perplexity
    log_sum = 0
    for token in answer_tokens:
        frequency = token_frequency.get(token, 0)
        if frequency == 0:
            # Set a small probability for unseen tokens
            probability = 1 / (total_tokens + 1)
        else:
            probability = frequency / total_tokens
        log_sum += math.log2(probability)
    if len(answer_tokens) > 0:
        perplexity = 2 ** (-log_sum / len(answer_tokens))
    else:
        perplexity = 0
    return perplexity


def bleurt_score(ideal_answer, generated_answer):
    checkpoint = "/mnt/workspace/Blended-RAG/code/checkpoints/bleurt-large-128/bleurt-large-128"
    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=[generated_answer], candidates=[ideal_answer])
    assert isinstance(scores, list) and len(scores) == 1
    return scores[0]


def blue(answer, ideal_answer):
    generated_tokens = nltk.word_tokenize(answer)
    reference_token_lists = [nltk.word_tokenize(answer) for answer in [ideal_answer]]
    bleu_score = nltk.translate.bleu_score.sentence_bleu(reference_token_lists, generated_tokens)
    return bleu_score


def meteor(answer, ideal_answer):
    generated_tokens = nltk.word_tokenize(answer)
    reference_token_lists = [nltk.word_tokenize(answer) for answer in [ideal_answer]]
    # Calculate the METEOR score
    meteor_score = ms.meteor_score(reference_token_lists, generated_tokens)
    # Instantiate a ROUGE scorer
    return meteor_score


def rouge(answer, ideal_answer):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # Calculate the ROUGE score
    score = scorer.score(answer, ideal_answer)
    # Extract the F1 score for ROUGE-1
    rouge_score = score['rouge1'].fmeasure
    return rouge_score


def compute_exact_match_ratio(output_text, gen_query):
    matcher = difflib.SequenceMatcher(None, output_text, gen_query)
    return matcher.ratio()


def get_prompt(context, question):
    text = f"""    
    Answer the following question based on the information given in the article. If there is no good answer, respond with "I don't know."\n\nArticle: {context} \n\nQuestion: {question}\nAnswer:"""
    # text = f"""    
    # Answer the following question using only information from the article. If there is no good answer in the article, say \"I don'\''t know\".\n\n ```: Article: {context} 
    # \n\nQuestion: {question}\nAnswer:```
    # """
    return text



# def process_squad(context, question):
    
#     # 创建输入提示
#     prompt = get_prompt(context.replace('`', ''), question)

#     # 使用 tokenizer 对输入进行编码
#     inputs = tokenizer1(prompt, return_tensors='pt').to('cuda')

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

#     # 去除多余空格和换行符
#     result = re.sub(r"\s+", " ", generated_text).strip()

#     # 返回处理后的结果
#     return result
import torch
import re

def process_squad(context, question):
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将模型移到设备
    model.to(device)

    # 创建输入提示
    prompt = get_prompt(context.replace('`', ''), question)

    # 使用 tokenizer 对输入进行编码并将其移到设备
    inputs = tokenizer1(prompt, return_tensors='pt').to(device)

    # 使用模型进行推理
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=50,  # 最大生成词数
            min_new_tokens=1,   # 最小生成词数
            do_sample=False,    # 贪心解码
            temperature=0.8,    # 控制生成的多样性
            top_k=50,           # 仅选择概率最高的50个词
            top_p=0.9           # nucleus sampling (可选)
        )

    # 解码生成的文本
    generated_text = tokenizer1.decode(outputs[0], skip_special_tokens=True)

    # 去除多余空格和换行符
    result = re.sub(r"\s+", " ", generated_text).strip()

    # 返回处理后的结果
    return result

# def process_squad(context, question):
#     headers = {
#         'Content-Type': 'application/json',
#         'Authorization': "Replace Token Here XXXXXX",
#     }

#     json_data = {
#         'model_id': 'google/flan-t5-xxl',
#         "inputs": [get_prompt(context.replace('`', ''), question)],
#         "parameters": {
#             "decoding_method": "greedy",
#             "min_new_tokens": 1,
#             "max_new_tokens": 50,
#             "moderations": {
#                 "hap": {
#                     "input": True,
#                     "threshold": 0.95,
#                     "output": True
#                 }
#             }
#         }
#     }


#     response = requests.post('https://bam-api.res.ibm.com/v1/generate', headers=headers, json=json_data)
#     json_response = json.loads(response.content.decode("utf-8"))
#     print("API Response:", json_response)
#     result = json_response['results'][0]['generated_text'].replace("\n", " ").replace("\s+", "");
#     return result


# def processESIndex_RAG(df_squad, index, search_query2):
#     count = 0
#     squad_count = 0
#     question_count = 0
#     count_value = 0
#     count_value_f1 = 0
#     blue_score_count = 0
#     meteor_score_count = 0
#     rouge_score_count = 0
#     sentence_similarity_score_count = 0
#     Sim_hash_score_count = 0
#     bleurt_score1_count = 0
#     bert_score1_count = 0
#     perplexity_score_count = 0
#     for ind in df_squad.index:
#         data = df_squad['data'][ind]
#         title = data['title']
#         content = data['paragraphs']
#         squad_count = squad_count + len(content)
#         for i in range(len(content)):
#             story = content[i]['context']
#             qas = content[i]['qas']
#             question_count = question_count + len(qas)
#             print("Questions----", question_count)
#             for j in range(len(qas)):
#                 gold_answer = qas[j]['answers'][0]['text']
#                 question = qas[j]['question']
#                 response = es_client.search(
#                     index=index,
#                     body=search_query2,
#                     size=1  # Set the number of documents to retrieve per scroll
#                 )
#                 all_hits = response['hits']['hits']
import ast
def processESIndex_RAG(df_squad, index):
    # 初始化各种计数器和指标
    count = 0
    squad_count = 0
    question_count = 0
    count_value = 0
    count_value_f1 = 0
    blue_score_count = 0
    meteor_score_count = 0
    rouge_score_count = 0
    sentence_similarity_score_count = 0
    Sim_hash_score_count = 0
    bleurt_score1_count = 0
    bert_score1_count = 0
    perplexity_score_count = 0
    # 遍历 DataFrame 的每一行
    for ind in df_squad.index:
        # 提取当前行的数据
        data = df_squad.loc[ind]
        # 使用 ast.literal_eval 将字符串转换为字典
        dict_obj = ast.literal_eval(data['answers'])
        # 提取 'answers', 'passages', 'question' 等字段
        gold_answer = dict_obj['text'][0]
        question = data['question']
        # 统计每个数据项中的段落和问题数量
        question_count += 1
        print("Questions----", question_count)
        response = es_client.search(
            index=index,
            body={"query" : {"multi_match": {
                                     "query": question, 
                                     "type": "best_fields",
                                     "fields": ["story", "title"],
                                     "tie_breaker":0.3
                                     }}},
            size=1  # 每次检索一个文档
        )
        # 提取搜索结果
        all_hits = response['hits']['hits']
        for num, doc in enumerate(all_hits):
            # print ("DOC ID:", doc["_id"], "--->", doc, type(doc), "\n")
            print("question:",question)
            print("gold_answer:",gold_answer)
            print("doc['_source']['story']:",doc['_source']['story'])

            if gold_answer in doc['_source']['story']:
                context = doc['_source']['story']
                ans = process_squad(context, question)
                # print("gold_answer:",gold_answer)
                print("ans:",ans)
                value = compute_exact_match_ratio(gold_answer, ans)
                count_value = count_value + value

                value_f1 = compute_f1(gold_answer, ans)
                count_value_f1 = count_value_f1 + value_f1
                blue_score = blue(gold_answer, ans)
                blue_score_count = blue_score + blue_score_count

                meteor_score = meteor(gold_answer, ans)
                meteor_score_count = meteor_score + meteor_score_count

                rouge_score = rouge(gold_answer, ans)
                rouge_score_count = rouge_score + rouge_score_count

                sentence_similarity_score = sentence_similarity(gold_answer, ans)
                sentence_similarity_score_count = sentence_similarity_score + sentence_similarity_score_count

                Sim_hash_score = Sim_hash(gold_answer, ans)
                Sim_hash_score_count = Sim_hash_score + Sim_hash_score_count

                perplexity_score = calculate_perplexity(gold_answer, ans)
                perplexity_score_count = perplexity_score + perplexity_score_count

                bleurt_score1 = bleurt_score(gold_answer, ans)
                bleurt_score1_count = bleurt_score1 + bleurt_score1_count
                try:
                    bert_score1 = bert_score(gold_answer, ans)
                    bert_score1_count = bert_score1 + bert_score1_count
                except Exception as e:
                    print(f"Error calculating BERT score: {e}")
                    continue

                print("Value -----", value, "f1----", value_f1)
        if question_count == 5:
            break


    print("Count value ----", count_value, "F1----", count_value_f1)
    print("Avg EM Accuracy", count_value / question_count)
    print("Avg f1 Accuracy", count_value_f1 / question_count)
    print("Avg blue_score Accuracy", blue_score_count / question_count)
    print("Avg meteor_score Accuracy", meteor_score_count / question_count)
    print("Avg rouge_score Accuracy", rouge_score_count / question_count)
    print("Avg sentence_similarity_score Accuracy", sentence_similarity_score_count / question_count)
    print("Avg Sim_hash_score_count Accuracy", Sim_hash_score_count / question_count)
    print("Avg perplexity_score_count Accuracy", perplexity_score_count / question_count)
    print("Avg bleurt_score1_count Accuracy", bleurt_score1_count / question_count)
    print("Avg bert_score1_count Accuracy", bert_score1_count / question_count)


# ## Read the file
# search_query = ""
# with open('./input/search_query/BM25/bm25_best.txt', 'r') as file:
#     search_query = file.read().rstrip()

# ## If you need to replace any file or  other query basis of Your index you can use the respective folders
# index_name = "research_index_bm25"
# df_questions = pd.read_csv("input question file")

# processESIndex_RAG(df_questions, index_name, search_query)
index_name ="research_index_bm25"
# df_questions =pd.read_json("dev-v2.0.json")
# validation_dataset = load_dataset('rajpurkar/squad_v2', split='validation')
# # 将 Hugging Face Dataset 转换为 Pandas DataFrame
# df_questions = pd.DataFrame(validation_dataset)
validation_dataset =pd.read_csv("/mnt/workspace/Blended-RAG/code/filtered_data.csv")
df_questions = pd.DataFrame(validation_dataset)

processESIndex_RAG(df_questions, index_name)
