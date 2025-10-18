import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data_read import get_train_data, get_test_data


class Net_work_v1(nn.Module):
    def __init__(self, input_N):
        super(Net_work_v1, self).__init__()

        self.fc1 = nn.Linear(input_N, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


# 训练实现

# 超参数
lr = 0.001
batch_size = 64
epochs = 500

# 加载数据
x_train_df, y_train_s = get_train_data()
x_test_df, test_ids = get_test_data()

# 转换数据给torch用
x_train = torch.tensor(x_train_df.astype(np.float32).values)
y_train = torch.tensor(y_train_s.astype(np.float32).values).view(-1, 1)
x_test = torch.tensor(x_test_df.astype(np.float32).values)

# 构建数据加载器
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
model = Net_work_v1(input_N=x_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


print("\n显卡嗷嗷叫环节")
start_time = time.time()

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

model.to(device)

for epoch in range(epochs):
    # 训练模式~
    model.train()

    # loss统计
    running_loss = 0.0

    for inputs, targets in train_loader:

        # 移动数据
        inputs, targets = inputs.to(device), targets.to(device)

        # 清除梯度
        optimizer.zero_grad()

        # 向前传播
        outputs = model(inputs)

        # 损失计算
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        # 累计损失
        running_loss += loss.item()

    # 统计每个epoch平均损失
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

end_time = time.time()
print(f"Finish,time: {(end_time - start_time):.2f}")


# 使用训练好的模型进行预测

print("\n生成预测文件")
# 顺序模式~
model.eval()
with torch.no_grad():
    # 转换测试数据给cuda
    x_test = x_test.to(device)
    predictions_log = model(x_test)

# 将结果移回CPU以便使用Numpy
predictions_log_cpu = predictions_log.cpu()
predictions = np.expm1(predictions_log_cpu.numpy().flatten())
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})
submission.to_csv('submission.csv', index=False)
print("提交文件 'submission.csv' 已生成！")
