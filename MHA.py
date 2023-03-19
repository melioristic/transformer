#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Created by : Mohit Anand
# Created on : Sat Mar 18 2023 at 7:57:52 PM
# ==========================================================
# Created on Sat Mar 18 2023
# __copyright__ = Copyright (c) 2023, Mohit Anand's Project
# __credits__ = [Mohit Anand,]
# __license__ = Private
# __version__ = 0.0.0
# __maintainer__ = Mohit Anand
# __email__ = itsmohitanand@gmail.com
# __status__ = Development
# ==========================================================


from torch import nn
from torch.nn import Parameter
from typing import Tuple
import torch
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    
    """MultiHeadAttention
    See "Attention is All You Need" for details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout = 0., bias=True, add_bias_kv=False, add_zero_attn = False) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.scale_factor = self.head_dim ** -0.5

        self.in_projection_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter("in_projection_bias", None)
        if bias:
            self.in_projection_bias = Parameter(torch.Tensor(3*embed_dim))
        self.out_projection = nn.Linear(embed_dim, embed_dim, bias = bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_projection_weight)
        nn.init.xavier_uniform_(self.out_projection.weight)
        if self.in_projection_bias is not None:
            nn.init.constant_(self.in_projection_bias, 0.)
            nn.init.constant_(self.out_projection.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask = None)-> Tuple:
        """
        # * Input Shape: Time x Batch x Channel

        Args:
            query (_type_): _description_
            key (_type_): _description_
            value (_type_): _description_
            attn_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple: _description_
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        n_timestep, bs, embed_dim = query.size()

        assert embed_dim == self.embed_dim
        
        if qkv_same:
            # * Self Attention
            q, k, v = self._in_projection_qkv(query)

        elif kv_same:
            if key == None:
                assert value is None
                k = v = None
            else:
                k, v = self._in_projection_kv(key)
        
        else:
            q = self._in_projection_q(query)
            k = self._in_projection_k(key)
            v = self._in_projection_v(value)

        q = q * self.scale_factor

        if self.bias_k is not None:
            assert self.bias_v is not None

            k = torch.cat([k, self.bias_k.repeat(1, bs, 1)])
            v = torch.cat([v, self.bias_k.repeat(1, bs, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(n_timestep, bs * self.num_heads, self.head_dim).transpose(0,1)

        if k is not None:
            k = k.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0,1)

        if v is not None:
            v = v.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0,1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len+=1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        
        assert list(attn_weights.size()) == [bs * self.num_heads, n_timestep, src_len]
        
        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bs * self.num_heads, n_timestep, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(n_timestep, bs, embed_dim)
        attn = self.out_projection(attn)
        # average attention weights over heads
        attn_weights = attn_weights.view(bs, self.num_heads, n_timestep, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights
    

    def _in_projection_qkv(self, query):
        return self._in_projection(query).chunk(3, dim=-1)
    
    def _in_projection_kv(self, key):
        return self._in_projection(key, start = self.embed_dim).chunk(2, dim=-1)

    def _in_projection_q(self, query):
        return self._in_projection(query, end=self.embed_dim)
    
    def _in_projection_k(self, key):
        return self._in_projection(key, start=self.embed_dim, end = 2 * self.embed_dim)
    
    def _in_projection_v(self, value):
        return self._in_projection(value, start = 2 * self.embed_dim)

    def _in_projection(self, input, start = 0, end=None, **kwargs):
        
        # ! How kwargs work when not called in the function
        
        weight = kwargs.get('weight', self.in_projection_weight)
        bias = kwargs.get('bias', self.in_projection_bias)
        weight = weight[start:end, :]

        if bias is not None:
            bias=bias[start:end]

        return F.linear(input, weight, bias)