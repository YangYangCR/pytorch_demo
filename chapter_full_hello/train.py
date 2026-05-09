import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# 简单 tokenizer
# =========================

vocab = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "hello": 3,
    "world": 4
}

id2word = {v: k for k, v in vocab.items()}


def tokenize(text):
    words = text.lower().split()
    vocab_ids = [vocab.get(w, 1) for w in words]
    return vocab_ids


# =========================
# Transformer Model
# =========================

class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 100, 32)
        )
        self.transformer = nn.Transformer(
            d_model=32,
            nhead=4,
            num_encoder_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(32, 100)

    def forward(self, src, tgt):
        src_len = src.size(1)
        tgt_len = tgt.size(1)
        src = self.embedding(src) + self.positional_encoding[:, :src_len, :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt_len, :]
        out = self.transformer(src, tgt)
        out = self.fc(out)
        return out


# =========================
# 创建模型
# =========================

model = TransformerModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# 构造训练数据
# =========================

src_text = "hello world"
tgt_text = "world hello"
src_tokens = tokenize(src_text)
tgt_tokens = tokenize(tgt_text)
src = torch.tensor([src_tokens])
tgt = torch.tensor([tgt_tokens])

print(src)
print(tgt)

# =========================
# 训练
# =========================

for epoch in range(200):
    output = model(src, tgt)
    loss = criterion(
        output.view(-1, 100),
        tgt.view(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"epoch={epoch}, loss={loss.item()}")

# =========================
# 保存模型
# =========================
# torch.save(model.state_dict(), "transformer.pth")
torch.save(model.state_dict(), "transformer.safetensors")
print("模型保存成功")
