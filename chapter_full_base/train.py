import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


class TransformerModel(nn.Module):

    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 1000, model_dim)
        )
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, src: Tensor, tgt: Tensor):
        src_seq_length = src.size(1)
        tgt_seq_length = tgt.size(1)
        src = self.embedding(src) + self.positional_encoding[:, :src_seq_length, :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt_seq_length, :]
        transformer_output = self.transformer(src, tgt)
        output = self.fc(transformer_output)
        return output


# 超参数
input_dim = 10000
model_dim = 128
num_heads = 8
num_layers = 2
output_dim = 10000

# 初始化模型
model = TransformerModel(
    input_dim,
    model_dim,
    num_heads,
    num_layers,
    output_dim
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模拟训练数据
src = torch.randint(0, input_dim, (32, 10))
tgt = torch.randint(0, input_dim, (32, 20))

# 训练
for epoch in range(5):
    output = model(src, tgt)

    loss = criterion(
        output.view(-1, output_dim),
        tgt.view(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"epoch={epoch}, loss={loss.item()}")

# 保存模型参数
torch.save(model.state_dict(), "transformer.pth")
print("模型保存成功")
