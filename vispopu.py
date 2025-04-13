import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math


# 读取数据
data = pd.read_csv('population.csv')
values = data[['总人口(亿)']].values.astype(np.float32)


# 设置中文字体以避免显示方框
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 绘制人口曲线
years = data['年份'].values  
plt.figure(figsize=(10, 6))
plt.plot(years, values, label='总人口（亿）', color='blue')
plt.title('人口增长曲线')
plt.xlabel('年份')
plt.ylabel('总人口（亿）')
plt.xticks(ticks=years, labels=years, rotation=90, fontsize=8)  # 显示所有年份并旋转以避免重叠，设置字号小一点
plt.legend()
plt.grid(True)
plt.show()