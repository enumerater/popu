import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from poputransform import PopulationTransformer,PositionalEncoding

# 设置随机种子以确保结果可复现
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(11111)

# 超参数设置
lookback = 5
batch_size = 5
lr = 1e-3
num_epochs = 80


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

def create_dataset(data, lookback=lookback, pred_steps=5):
    X, Y = [], []
    for i in range(len(data)-lookback-pred_steps+1):
        X.append(data[i:i+lookback])
        Y.append(data[i+lookback:i+lookback+pred_steps, 0])
    return np.array(X), np.array(Y)

X, y = create_dataset(scaled_data, lookback=lookback)

# 使用全部数据训练
X_train_tensor = torch.FloatTensor(X)
y_train_tensor = torch.FloatTensor(y)
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 初始化模型
model = PopulationTransformer(lookback=lookback)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)



# 绘制损失曲线
losses = []
# 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}')

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid()
plt.show()

# 保存模型
torch.save(model.state_dict(), './population_transformer.pth')
print("模型已保存到 ./population_transformer.pth")