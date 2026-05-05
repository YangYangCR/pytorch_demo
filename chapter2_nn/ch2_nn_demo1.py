import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim

# 定义输入层大小、隐藏层大小、输出层大小和批量大小
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

# 创建虚拟输入数据和目标数据
x = torch.randn(batch_size, n_in)  # 随机生成输入数据
y = torch.tensor([[1.0], [0.0], [0.0],
                  [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])  # 目标输出数据

# 定义神经网络   Sequential把多个神经网络层按照顺序串起来执行
model = nn.Sequential(
    nn.Linear(n_in, n_h),  # 输入层到隐藏层的线性变换
    nn.ReLU(),  # 隐藏层的激活函数
    nn.Linear(n_h, n_out),  # 隐藏层到输出层的线性变化
    nn.Sigmoid()  # 输出层的激活函数
)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 梯度下降优化器，学习率0.01

# 用于存储每轮的损失值
losses = []

"""
参数分析
    第一层：
    5个输出神经元，每个都需要10个输入权重 所以 5 × 10 = 50 个参数
    每个输出神经元一个偏置：5 个 bias 
    所以一共  50 + 5 = 55 个参数 
    第二层：
    1个输出神经元，5维输入，1维输出，所以 5 个参数
    每个输出神经元一个偏置：1 个 bias
    所以一共  5 + 1 = 6 个参数 
    总结：
    模型一共 55 + 6 = 61 个参数
"""
# 执行梯度下降算法进行模型训练
for epoch in range(100):
    y_pred = model(x)  # 前向传播，计算预测值
    loss = criterion(y_pred, y) # 计算损失值
    losses.append(loss.item()) # 记录损失值
    print("===============================================")
    print(f'Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}')  # 打印损失值
    print(f"param is ")
    for param in model.parameters():
        print(f'\t{param}')
    optimizer.zero_grad() # 梯度清零
    loss.backward() # 反向传播
    optimizer.step() # 更新模型参数


# 可视化损失变化曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, 101), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

# 可视化预测结果与实际目标值对比
y_pred_final = model(x).detach().numpy()  # 最终预测值
y_actual = y.numpy()  # 实际值

plt.figure(figsize=(8, 5))
plt.plot(range(1, batch_size + 1), y_actual, 'o-', label='Actual', color='blue')
plt.plot(range(1, batch_size + 1), y_pred_final, 'x--', label='Predicted', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid()
plt.show()