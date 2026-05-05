from torch import nn, optim
import torch


class SimpleNN(nn.Module):

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=3)  # 输入层到隐藏层
        self.fc2 = nn.Linear(in_features=3, out_features=1)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 2 创建模型实例
model = SimpleNN()  # 创建网络实例

# 3 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4 假设我们有训练数据X和Y
X = torch.randn(10, 2)  # 10个样本 2个特征
Y = torch.randn(10, 1)  # 10个目标值

# 5、训练循环
for epoch in range(100): # 训练100轮
    optimizer.zero_grad() # 清空之前的梯度
    output = model(X) # 前向传播
    loss = criterion(output, Y) # 计算损失
    loss.backward() #反向传播
    optimizer.step() # 更新参数

    # 每 10 轮输出一次损失
    if (epoch + 1) % 10 == 0:
        print("==============================")
        for param in model.parameters():
            print(f"param is {param}")
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')


