import os
import re
from datetime import date
import pandas as pd
import json
from datetime import datetime
import requests
from datasets import load_dataset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError



## Replace elastic instance here
es_client = Elasticsearch(
    ["http://10.224.160.85:9200"]
)
es_client.info()

## Download model for KNN
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

## get the files from specific folder
def get_all_files(folder_name):
    # Change the directory
    os.chdir(folder_name)
    # iterate through all file
    file_path_list =[]
    for file in os.listdir():
        print(file)
        file_path = f"{folder_name}/{file}"
        file_path_list.append(file_path)
    return file_path_list


## create the index
def create_index(index_name,mapping):
    try:
        es_client.indices.create(index=index_name,body = mapping)
        print(f"Index '{index_name}' created successfully.")
    except RequestError as e:
        if e.error == 'resource_already_exists_exception':
            print(f"Index '{index_name}' already exists.")
        else:
            print(f"An error occurred while creating index '{index_name}': {e}")



# def index_data(df_docs,source,index_name,index_name_knn):
#     i=0
#     for index, row in df_docs.iterrows():
#         i=i+1
#         print("Processing i",i)
#         id_ = row['id']
#         text = row['text']
#         title = row['title']
#         source = source
#         text_embedding = model.encode(text)
#         doc ={
#                         "id": ""+title+"",
#                         "source": ""+source+"",
#                         "text_field": ""+text+"",
#                         "title": ""+title+"",
#                         "metadata": ""
#             }
#         doc_knn = {
#                         "id": ""+title+"",
#                         "source": ""+source+"",
#                         "text": ""+text+"",
#                         "title": ""+title+"",
#                         "metadata": "",
#                         "text_embedding": text_embedding
#                     }
#         response = es_client.index(index=index_name, body=doc)
#         print(response)
#         response = es_client.index(index=index_name_knn, body=doc_knn)
#         print(response)

def index_data(df_docs,source,index_name,index_name_knn):
    i=0
    for index, row in df_docs.iterrows():
        i=i+1
        print("Processing i",i)
        id = row['id']
        text = row['context']
        title = row['title']
        source = source
        text_embedding = model.encode(text)
        doc ={
                        "id": ""+id+"",
                        "source": ""+source+"",
                        "story": ""+text+"",
                        "title": ""+title+""
            }
        doc_knn = {
                        "id": ""+id+"",
                        "source": ""+source+"",
                        "story": ""+text+"",
                        "title": ""+title+"",
                        "text_embedding": text_embedding
                    }
        response = es_client.index(index=index_name, body=doc)
        print(response)
        response = es_client.index(index=index_name_knn, body=doc_knn)
        print(response)

def delete_index(index_name):
    # 1. 删除现有索引
    try:
        if es_client.indices.exists(index=index_name):
            es_client.indices.delete(index=index_name)
            print(f"Index '{index_name}' has been deleted.")
        else:
            print(f"Index '{index_name}' does not exist.")
    except Exception as e:
        print(f"Error deleting index: {e}")




## Example Index name
index_name_knn = 'research_index_knn'
index_name = "research_index_bm25"
index_name_elser = 'research_index_elser'

delete_index(index_name)
delete_index(index_name_knn)

## Create Index BM25
with open('../input/mapping/bm25.txt', 'r') as file:
    mapping_str = file.read().rstrip()
        # 将 JSON 字符串解析为 Python 字典
# print("Mapping string:", mapping_str)

mapping = json.loads(mapping_str)
create_index(index_name,mapping)

## Create Index Knn
with open('../input/mapping/knn.txt', 'r') as file:
    mapping_str = file.read().rstrip()
mapping = json.loads(mapping_str)
create_index(index_name_knn,mapping)

## Create Index Sparse Encoder for ELSER V1 model
with open('../input/mapping/sparse_encoder.txt', 'r') as file:
    mapping_str = file.read().rstrip()
mapping = json.loads(mapping_str)
create_index(index_name_elser,mapping)


# ## Define folder name
# doc_folder_msmarco = '/Users/abhilashamangal/Documents/Semantic Search/data/msmarco/'
# files_msmarco = get_all_files(doc_folder_msmarco)
# df_corpus = pd.read_json(files_msmarco[2],lines=True)

# source ="msmarco"
# index_data(df_corpus,source,index_name,index_name_knn)

# 加载 MS MARCO 数据集的 validation 分割
validation_dataset = load_dataset('rajpurkar/squad_v2', split='validation')
# 将 Hugging Face Dataset 转换为 Pandas DataFrame
df_docs = validation_dataset.to_pandas()

source ="squad_v2"
index_data(df_docs,source,index_name,index_name_knn)