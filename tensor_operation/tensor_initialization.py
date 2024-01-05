# change the file to the jupyter, makes the result stored
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float, requires_grad=True,
                 device=device)
print(x)
print(x.shape)
print(x.device)
print(x.requires_grad)

y = torch.empty(size=(2, 3))    # 分配的内存空间上本来存在的值，不一定是零
zero_y = torch.zeros((2, 3))    # 创建一个全零的tensor
rand_y = torch.rand((3, 3))     # 来自[0, 1]的均匀分布
ones_y = torch.ones((2, 3))
print(ones_y)
