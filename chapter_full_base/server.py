import torch
import torch.nn as nn
from torch import Tensor
from fastapi import FastAPI
from pydantic import BaseModel


# =========================
# 定义模型结构
# =========================

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


# =========================
# 模型参数
# =========================

input_dim = 10000
model_dim = 128
num_heads = 8
num_layers = 2
output_dim = 10000


# =========================
# 加载模型
# =========================

model = TransformerModel(
    input_dim,
    model_dim,
    num_heads,
    num_layers,
    output_dim
)

model.load_state_dict(torch.load("transformer.pth"))
model.eval()

print("模型加载完成")


# =========================
# 创建 FastAPI
# =========================

app = FastAPI()

# =========================
# 请求结构
# =========================

class PredictRequest(BaseModel):
    src: list
    tgt: list


# =========================
# 推理接口
# =========================

@app.post("/predict")
def predict(req: PredictRequest):

    with torch.no_grad():
        src_tensor = torch.tensor(req.src)
        tgt_tensor = torch.tensor(req.tgt)
        output = model(src_tensor, tgt_tensor)
        print(f"model output: {output}")
        predicted = torch.argmax(output, dim=-1)
        return {
            "predicted_tokens": predicted.tolist()
        }