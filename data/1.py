from datasets import Dataset

# 直接加载 .arrow 文件（注意：需是完整的 dataset 文件，不是索引）
ds = Dataset.from_file("/home/user02/deepcode/deepcode/data/iaminju___paper2code/default/0.0.0/0e4a4bb780505d7497dc1201f4051c1c0121fba8/paper2code-test.arrow")
print(ds[0])  # 查看第一条
print(ds.column_names)  # 查看字段