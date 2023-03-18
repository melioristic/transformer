import torch 
import numpy as np

from MHA import MultiHeadAttention

x = MultiHeadAttention(15, 5)
q = torch.Tensor(np.ones((36, 100, 15)))
k = torch.Tensor(np.ones((24, 100, 15)))
v = torch.Tensor(np.ones((24, 100, 15)))

a, aw = x(q, k, v)

print(a.shape)
print(aw.shape)