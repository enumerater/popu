from poputransform import PopulationTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

lookback = 5


model = PopulationTransformer(lookback=lookback)
model.load_state_dict(torch.load('population_transformer.pth'))

# 读取数据
data = pd.read_csv('population.csv')
values = data[['总人口(亿)']].values.astype(np.float32)

# 手动实现Min-Max归一化（按列归一化）
def manual_minmax_scale(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    scaled = (data - min_vals) / (max_vals - min_vals + 1e-8)  # 防止除以0
    return scaled, min_vals, max_vals

# 反归一化函数
def manual_inverse_scale(scaled_data, min_vals, max_vals):
    return scaled_data * (max_vals - min_vals) + min_vals

# 归一化数据
scaled_data, data_min, data_max = manual_minmax_scale(values)
original_min = data_min  # 保存初始归一化参数
original_max = data_max

last_window = scaled_data[-lookback:]

# 预测未来五年
model.eval()
with torch.no_grad():
    last_window = scaled_data[-lookback:]  # 取最后5个已知数据点
    input_tensor = torch.FloatTensor(last_window).unsqueeze(0)  # 添加batch维度
    prediction = model(input_tensor).numpy().reshape(-1, 1)

# 反归一化得到实际人口值
predicted_population = manual_inverse_scale(prediction, data_min, data_max)

# 输出预测结果
print("\n预测结果：")
for i, value in enumerate(predicted_population.flatten(), 1):
    print(f"第{i}年预测人口：{value:.2f}亿")


# 读取数据
data = pd.read_csv('population.csv')
values = data[['总人口(亿)']].values.astype(np.float32)

# 设置中文字体以避免显示方框
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 绘制人口曲线
years = data['年份'].values
pred_years = np.arange(2025, 2025 + len(predicted_population))

plt.figure(figsize=(10, 6))
plt.plot(years, values, label='总人口（亿）', color='blue')
plt.plot(pred_years, predicted_population, label='预测人口（亿）', color='red', linestyle='--')
plt.title('人口增长曲线及预测')
plt.xlabel('年份')
plt.ylabel('总人口（亿）')
plt.xticks(ticks=np.concatenate((years, pred_years)), rotation=90, fontsize=8)  # 显示所有年份并旋转以避免重叠
plt.legend()
plt.grid(True)
plt.show()
