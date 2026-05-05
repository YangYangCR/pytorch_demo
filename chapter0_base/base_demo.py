import torch

# 当前安装的 PyTorch 库的版本
print(torch.__version__)
# 检查 CUDA 是否可用，即你的系统有 NVIDIA 的 GPU
print(torch.cuda.is_available())


def zhangliang():
    a = torch.zeros(2, 3)  # 创建一个 2x3 的全 0 张量
    print(a)
    b = torch.ones(2, 3)  # 创建一个 2x3 的全 1 张量
    print(b)
    c = torch.rand(2, 3)  # 创建一个 2x3 的随机数张量
    print(c)

    print("=========")
    e = torch.rand(2, 3)
    print(e)
    f = torch.rand(2, 3)
    print(f)
    print(f"e + f is {e + f}")
    print(f"e * f is {e * f}")  # 点积
    print(f"e 转置 {e.t()}")
    print(f"e shape {e.shape}")


# PyTorch 张量可以存在于不同的设备上，包括CPU和GPU，可以将张量移动到 GPU 上以加速计算
def device():
    if torch.cuda.is_available():
        pass
        # tensor_gpu = tensor_from_list.to("cuda")


# 张量求导， 对标量求导
def grad_biaoliang():
    # 创建一个需要梯度的张量
    tensor_requires_grad = torch.tensor([2.0], requires_grad=True)
    # 进行一些操作 y = x * 2
    # tensor_result = tensor_requires_grad * 2
    # 进行一些操作 y = x^2
    tensor_result = tensor_requires_grad ** 2
    # 计算梯度 backward 只能对标量求导
    tensor_result.backward()
    # 输出梯度
    print(tensor_requires_grad.grad)


# 张量求导， 对向量求导(雅可比矩阵)
def grad_xiangliang():
    # 创建一个需要梯度的张量
    tensor_requires_grad = torch.tensor([2.0, 1.0], requires_grad=True)
    # 进行一些操作 y = x * 2
    # tensor_result = tensor_requires_grad * 2
    # 进行一些操作 y = x^2
    tensor_result = tensor_requires_grad ** 2
    # 计算梯度 backward 只能对标量求导
    tensor_result.backward(gradient=torch.tensor([1.0, 2.0]))
    # 输出梯度
    print(tensor_requires_grad.grad)


def grad_biaoliang1():
    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    print("======")
    # 执行某些操作
    y = x + 2
    z = y * y * 3
    print(f"z is {z}")
    out = z.mean()  #对张量中所有元素求平均值
    print(out)


if __name__ == '__main__':
    # zhangliang()
    # grad_biaoliang()
    grad_xiangliang()
    # grad_biaoliang1()
