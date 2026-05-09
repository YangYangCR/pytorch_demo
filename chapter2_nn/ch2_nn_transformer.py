import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


class TransformerModel(nn.Module):

    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        """
            生成一个 input_dim * model_dim 的矩阵，
            input_dim 代表词表中token的个数 model_dim 为token的向量长度
            embedding([2,3,6]) 代表从embedding中取出第2,3,6行组成新的矩阵
        """
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, model_dim))  # 假设序列长度最大为1000
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    # src 10 * 32 tgt 10 * 32
    def forward(self, src: Tensor, tgt: Tensor):
        # 获取第二个维度
        src_seq_length, tgt_seq_length = src.size(1), tgt.size(1)
        """
            假设 embedding [
              1,1,1
              2,2,2
              3,3,3
            ]
            src = [
              2,1
              1,3
            ]
            embedding(src) 为 [
                [
                   [2,2,2]
                   [1,1,1] 
                ],
                [
                    [1,1,1]
                    [3,3,3]
                ]
            ]
            
            positional_encoding的size为a * b * c  
            positional_encoding[:, :src_seq_length, :]  取a维度的所有数据 b的前src_seq_length维数据，c的所有维数据
        """
        src = self.embedding(src) + self.positional_encoding[:, :src_seq_length, :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt_seq_length, :]
        transformer_output = self.transformer(src, tgt)
        output = self.fc(transformer_output)
        return output


# 超参数
input_dim = 10000  # 词汇表大小
model_dim = 512  # 模型维度
num_heads = 8  # 多头注意力头数
num_layers = 6  # 编码器和解码器层数
output_dim = 10000  # 输出维度（通常与词汇表大小相同）

# 初始化模型、损失函数和优化器
model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 假设输入数据
src = torch.randint(0, input_dim, (10, 32))  # (序列长度, 批量大小) 10*32 每个元素在0~input_dim
tgt = torch.randint(0, input_dim, (20, 32))  # (序列长度, 批量大小) 10*32 每个元素在0~input_dim

# 前向传播
output = model(src, tgt)
print(f"model param is ")
for param in model.parameters():
    print(f"param shape is {param.shape}")
print(f"out put shape is {output.shape}")

# 计算损失
loss = criterion(output.view(-1, output_dim), tgt.view(-1))

# 反向传播和优化
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())

# 模型保存
torch.save(model.state_dict(), "./ch2_nn_transformer.pth")

# 加载模型
model = TransformerModel()
model.load_state_dict(torch.load("./ch2_nn_transformer.pth"))

# 切换推理模型
model.eval()

