from datasets import load_dataset

# 下载 MS MARCO 的开发集
dev_queries = load_dataset('unicamp-dl/msmarco', 'v2.1', split='dev')

# 获取数据集的缓存路径
cache_path = dev_queries.cache_files
print(cache_path)
