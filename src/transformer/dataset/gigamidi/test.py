# 读取一个巨大的csv文件，并打印前5行内容
import pandas as pd
file_path = './metadata.csv'

# 读取这个csv，并把它的前10行保存为一个测试用的csv
df = pd.read_csv(file_path)
df.head(10).to_csv('./test_metadata.csv', index=False)

print("Test metadata CSV created with first 10 rows.")

# 统计有多少个metadata的包含no-drums
no_drums_count = df['file_path'].str.contains('no-drums').sum()
print(f"Number of metadata entries containing 'no-drums': {no_drums_count}")