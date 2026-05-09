import torch
from torch import nn

data = torch.randint(5, 10, (2, 2))
# print(data)

embedding = nn.Embedding(5, 3)
print(f"embedding is {embedding.weight}")

x = torch.tensor([1, 3, 4])

print(f"embedding x: {embedding(x)}")

print(torch.zeros(2, 4, 5))

x = torch.randn(3, 2, 4)
print(f"x size() {x.size()}")
print(f"x size(0) {x.size(0)}")
print(f"x size(1) {x.size(1)}")
print(f"x size(2) {x.size(2)}")