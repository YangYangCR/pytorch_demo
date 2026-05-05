from torch.utils.data import Dataset, DataLoader
import torch

"""
    自定义Dataset
    torch.utils.data.Dataset 是一个抽象类，允许你从自己的数据源中创建数据集。
"""


class MyDataset(Dataset):

    def __init__(self, X_data, Y_data):
        """
         初始化数据集， X_data 和 Y_data 是两个列表或数组
        :param X_data: 输入特征
        :param Y_data: 目标标签
        """
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        """ 返回数据集的大小 """
        return len(self.X_data)

    def __getitem__(self, idx):
        """ 返回指定索引的数据 """
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)
        y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        return x, y


# 示例数据
X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]
Y_data = [1, 0, 1, 0]

# 创建数据集实例
dataset = MyDataset(X_data, Y_data)

"""
   DataLoader 是 PyTorch 提供的一个重要工具，用于从 Dataset 中按批次（batch）加载数据。
"""

# batchsize 设置每次加载的样本数量 shuffle 是否加载数据时将数据打乱
dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

# 打印加载的数据
for epoch in range(1):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        print(f'Batch {batch_idx + 1}:')
        print(f'Inputs: {inputs}')
        print(f'Labels: {labels}')





