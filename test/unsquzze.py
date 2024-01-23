import torch

x = torch.randn(2, 3, dtype=torch.float)
print(x)
print(x.shape)
# 最后一个维度的元素切分，使最后一个维度里的每个元素成为一个新维度，直观一点就是把最后一维的所有元素用[]框起来
x = x.unsqueeze(-1)
print(x)
print(x.shape)

"""
输出结果如下：
tensor([[-1.5889,  0.8244,  0.0208],
        [-0.1917,  1.5729,  0.9128]])
torch.Size([2, 3])
tensor([[[-1.5889],
         [ 0.8244],
         [ 0.0208]],

        [[-0.1917],
         [ 1.5729],
         [ 0.9128]]])
torch.Size([2, 3, 1])
"""