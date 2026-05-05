import torch
from matplotlib import pyplot as plt
from torch import nn

"""
    线性回归模型学习
"""
# 随机种子，确保每次运行结果一致
torch.manual_seed(42)

# 生成训练数据
X = torch.randn(100, 2)  # 100个样本，每个样本2个特征
"""
    true_w是一维向量，PyTorch 会自动把一维向量当做[2,1]的大小进行矩阵乘法
"""
true_w = torch.tensor([2.0, 3.0])  # 真实权重
print(f"true w is {true_w} shape is {true_w.shape}")
true_b = 4.0
Y = X @ true_w + true_b + torch.randn(100) * 0.1

# 打印部分数据 切片操作 取前五行数据
print(X[:5])
print(Y[:5])


# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 定义线性层，输入为2个特征，输出为一个预测值
        # nn.Linear 会自动创建权重矩阵和偏置向量，不需要手动定义。
        self.linear = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.linear(x)  # 前向传播，预测返回结果


# 创建模型实例
model = LinearRegressionModel()
# 定义损失函数
criterion = nn.MSELoss()
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 开始训练
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    # 前向传播
    predictions = model(X)  # 调用对应的__call__ 方法， 最后会调用到forward方法
    #print(f"predictions shape is {predictions.shape} predictions squeeze shape is {predictions.squeeze().shape} Y shape is {Y.shape} ")
    """
      predictions的shape [100,1]
      Y的shape [100]
      predictions.squeeze() 将[100,1]变为[100]
    """
    loss = criterion(predictions.squeeze(), Y)

    # 反向传播
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新模型参数

    # 打印损失
    if (epoch + 1) % 100 == 0:
        print("==================================================")
        for param in model.parameters():
            print(f" param is {param}")
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')



# 查看训练后的权重和偏置
print(f'Predicted weight: {model.linear.weight.data.numpy()}')
print(f'Predicted bias: {model.linear.bias.data.numpy()}')

# 在新数据上做预测
with torch.no_grad():  # 评估时不需要计算梯度
    predictions = model(X)

# 可视化预测与实际值
plt.scatter(X[:, 0], Y, color='blue', label='True values')
plt.scatter(X[:, 0], predictions, color='red', label='Predictions')
plt.legend()
plt.show()