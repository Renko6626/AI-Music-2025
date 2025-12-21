import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- 1. 配置信息 ---
file_configs = [
    {'name': 'Nano', 'train': 'nano_train.json', 'val': 'nano_validation.json', 'color': '#1f77b4'},
    {'name': 'Standard', 'train': 'standard_train.json', 'val': 'standard_validation.json', 'color': '#ff7f0e'},
    {'name': 'Heavy', 'train': 'heavy_train.json', 'val': 'heavy_validation.json', 'color': '#2ca02c'}
]

def load_json_log(filepath):
    """读取每行一个JSON对象的日志文件"""
    steps = []
    values = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                steps.append(data['step'])
                values.append(data['value'])
    except FileNotFoundError:
        print(f"警告: 未找到文件 {filepath}")
    return np.array(steps), np.array(values)

def smooth_curve(values, weight=0.85):
    """指数移动平均平滑"""
    last = values[0]
    smoothed = []
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# --- 2. 开始画图 ---
plt.style.use('seaborn-v0_8-paper') # 使用学术风格
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

for config in file_configs:
    # 处理 Training Data
    t_steps, t_vals = load_json_log(config['train'])
    if len(t_vals) > 0:
        # 画原始数据（浅色细线）和增强平滑线（深色粗线）
        ax1.plot(t_steps, t_vals, color=config['color'], alpha=0.2, linewidth=1)
        ax1.plot(t_steps, smooth_curve(t_vals, 0.9), color=config['color'], 
                 label=f"{config['name']}", linewidth=2)

    # 处理 Validation Data
    v_steps, v_vals = load_json_log(config['val'])
    if len(v_vals) > 0:
        # 验证集通常不需要过度平滑，直接画带点线
        ax2.plot(v_steps, v_vals, color=config['color'], marker='o', 
                 markersize=4, label=f"{config['name']}", linewidth=2)

# --- 3. 修饰图表 ---
# 左图：Training Loss
ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
ax1.set_xlabel('Steps', fontsize=12)
ax1.set_ylabel('Cross Entropy Loss', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# 右图：Validation Loss
ax2.set_title('Validation Loss (Per Epoch)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# 自动调整布局
plt.tight_layout()

# 保存为矢量图（PDF），方便放入 LaTeX 论文
plt.savefig('model_training_comparison.pdf')
plt.show()