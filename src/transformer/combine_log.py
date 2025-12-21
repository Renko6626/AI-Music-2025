import os
import pandas as pd
from tbparse import SummaryReader
import matplotlib.pyplot as plt

# 1. 指定你的日志目录（包含那 4 个文件的文件夹）
log_dir = './logs_gpt_standard/music_gpt_standard_datav3' 
#log_dir = "./logs_gpt_nano/music_gpt_nano_datav3"
log_dir = "./logs_gpt_heavy/music_gpt_heavy_datav3"
# 2. 读取数据 (tbparse 会自动扫描并合并目录下所有 tfevents 文件)
reader = SummaryReader(log_dir)
df = reader.scalars

# 3. 筛选出所有的 training loss
# 注意：在 DataFrame 里检查一下你的 tag 叫什么，可能是 'loss' 或 'train/loss'

print(df['tag'].unique())  # 打印所有可用的 tag，确认训练损失的名称
train_loss = df[df['tag'] == 'Train/Loss_step'].sort_values('step')
validation_loss = df[df['tag'] == 'Val/Loss_epoch'].sort_values('step')

# 4. 检查数据量
print(f"共读取到 {len(train_loss)} 条训练记录")

# 5. 绘图验证
"""
plt.plot(train_loss['step'], train_loss['value'])
plt.title("Full Training Loss (Merged)")
plt.show()
plt.savefig("complete_training_loss.png")
"""
# 6. 导出为json
train_loss.to_json("../draw/heavy_train.json", orient="records", lines=True)
validation_loss.to_json("../draw/heavy_validation.json", orient="records", lines=True)