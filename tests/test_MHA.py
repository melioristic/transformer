from MHA import MultiHeadAttention
import torch
import numpy as np

def test_MultiHeadAttention():
    x = MultiHeadAttention(15, 5)
    q = torch.Tensor(np.ones((36, 100, 15)))
    k = torch.Tensor(np.ones((24, 100, 15)))
    v = torch.Tensor(np.ones((24, 100, 15)))

    a, aw = x(q, k, v)

    assert a.shape == (36, 100, 15)