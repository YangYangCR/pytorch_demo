import torch
import torch.nn as nn

from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# tokenizer
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

    return [vocab.get(w, 1) for w in words]


def decode(ids):
    return [id2word.get(i, "<unk>") for i in ids]


# =========================
# Transformer
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
# 加载模型
# =========================

model = TransformerModel()
# model.load_state_dict(torch.load("transformer.pth"))
model.load_state_dict(torch.load("transformer.safetensors"))
model.eval()

print("模型加载成功")

# =========================
# FastAPI
# =========================

app = FastAPI()


class RequestData(BaseModel):
    text: str


@app.post("/predict")
def predict(req: RequestData):
    text = req.text
    src_tokens = tokenize(text)
    src = torch.tensor([src_tokens])
    tgt = src.clone()
    with torch.no_grad():
        output = model(src, tgt)
        print(f"output is {output}")
        predicted_ids = torch.argmax(output, dim=-1)
        print(f"predicted_ids is {predicted_ids}")
        predicted_ids = predicted_ids[0].tolist()
        result = decode(predicted_ids)

    return {
        "input": text,
        "tokens": src_tokens,
        "predicted_ids": predicted_ids,
        "result": result
    }
