from datasets import load_dataset

# 下载 MS MARCO 的开发集
dev_queries = load_dataset('microsoft/ms_marco', 'v2.1', split='validation')

# 获取数据集的缓存路径
cache_path = dev_queries.cache_files
print(cache_path)
print(dev_queries[0])
import pandas as pd

# 将数据集转换为 DataFrame
df = pd.DataFrame(dev_queries)

# 查看 DataFrame 的前几行
print(df.head())


