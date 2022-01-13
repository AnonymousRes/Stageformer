# -*- coding: utf-8 -*-
# Author: ???
import os
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import pickle
import warnings
warnings.filterwarnings('ignore')
from torch.nn.modules.activation import MultiheadAttention
import copy

class Attention(nn.Module):
    """
    Compute Self-Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Compute Multi-Headed Self-Attention
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        # self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        #                      for l, x in zip(self.linear_layers, (query, key, value))]
        query = self.linear_q(query)
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class ChunkedConvblock(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 output_dim=128,
                 n_stages=0,
                 n_steps=0,
                 device=None
                 ):
        super(ChunkedConvblock, self).__init__()
        # assert input_size != None and isinstance(input_size, int), 'fill in correct input_size'
        self.n_stages = n_stages
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.conv1d_temporal1 = torch.nn.Conv1d(in_channels=self.hidden_dim,
                                                out_channels=self.output_dim,
                                                kernel_size=2,
                                                dilation=2,
                                                stride=1,
                                                padding=1,
                                                )
        self.dotatt = Attention()
        self.norm = torch.nn.LayerNorm(normalized_shape=self.hidden_dim)
        # self.max_pooling = torch.nn.AvgPool1d(kernel_size=int(self.n_steps / self.n_stages), stride=1)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout()

    def forward(self, x):
        """
        x, shape (batchsize, n_steps, outputdim)
        output, shape (batchsize, n_stages, outputdim)
        """
        x_size = x.size()
        n_batchsize, n_timestep, n_orifeatdim = x_size[0], x_size[1], x_size[2]
        assert n_timestep % self.n_stages == 0, "length of the sequence must be divisible by n_stages"
        chunked_x = torch.reshape(x, [n_batchsize*self.n_stages, int(n_timestep/self.n_stages), n_orifeatdim])
        c_temporal = chunked_x.permute(0, 2, 1)
        # c_temporal = self.conv1d_temporal2(self.drop(self.relu(self.conv1d_temporal1(c_temporal))))
        # c_temporal = self.relu(c_temporal)
        # print('c_temporal', c_temporal.size())
        c_temporal = self.drop(self.relu(self.conv1d_temporal1(c_temporal)))
        # print('c_temporal', c_temporal.size())
        # c_temporal = self.max_pooling(c_temporal)
        # c_temporal = torch.mean(c_temporal, dim=-1, keepdim=True)
        # print('c_temporal', c_temporal.size())
        c_temporal = c_temporal.permute(0, 2, 1)
        c_temporal = self.norm(self.dotatt(chunked_x, c_temporal, c_temporal)[0] + c_temporal)
        c_temporal = torch.mean(c_temporal, dim=-2, keepdim=True)
        # c_temporal = self.norm(self.dotatt(chunked_x, c_temporal, c_temporal)[0] + chunked_x)
        # print('c_temporal',c_temporal.size())
        # exit(0)
        c_temporal = c_temporal.reshape(n_batchsize, -1, self.hidden_dim)
        # c_temporal = self.norm(c_temporal)
        # c_temporal = self.dotatt(c_temporal, c_temporal, c_temporal)[0]
        # c_temporal = self.norm(c_temporal)
        # c_temporal = self.norm(c_temporal)
        # c_temporal = self.dotatt(c_temporal, c_temporal, c_temporal)[0]
        # print('c_temporal', c_temporal.size())
        # exit(0)

        return c_temporal


class ChunkedConvblock_C(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 output_dim=128,
                 n_stages=0,
                 n_steps=0,
                 device=None
                 ):
        super(ChunkedConvblock_C, self).__init__()
        # assert input_size != None and isinstance(input_size, int), 'fill in correct input_size'
        self.n_stages = n_stages
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        # self.conv1d_temporal1 = torch.nn.Conv1d(in_channels=self.hidden_dim,
        #                                         out_channels=self.output_dim,
        #                                         kernel_size=2,
        #                                         dilation=2,
        #                                         stride=1,
        #                                         padding=1,
        #                                         )
        self.dotatt = Attention()
        self.norm = torch.nn.LayerNorm(normalized_shape=self.hidden_dim)
        # self.max_pooling = torch.nn.AvgPool1d(kernel_size=int(self.n_steps / self.n_stages), stride=1)
        # self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout()

    def forward(self, x):
        """
        x, shape (batchsize, n_steps, outputdim)
        output, shape (batchsize, n_stages, outputdim)
        """
        x_size = x.size()
        n_batchsize, n_timestep, n_orifeatdim = x_size[0], x_size[1], x_size[2]
        assert n_timestep % self.n_stages == 0, "length of the sequence must be divisible by n_stages"
        chunked_x = torch.reshape(x, [n_batchsize*self.n_stages, int(n_timestep/self.n_stages), n_orifeatdim])
        # c_temporal = chunked_x.permute(0, 2, 1)
        # c_temporal = self.conv1d_temporal2(self.drop(self.relu(self.conv1d_temporal1(c_temporal))))
        # c_temporal = self.relu(c_temporal)
        # print('c_temporal', c_temporal.size())
        # c_temporal = self.drop(self.relu(self.conv1d_temporal1(c_temporal)))
        # print('c_temporal', c_temporal.size())
        # c_temporal = self.max_pooling(c_temporal)
        # c_temporal = torch.mean(c_temporal, dim=-1, keepdim=True)
        # print('c_temporal', c_temporal.size())
        # c_temporal = c_temporal.permute(0, 2, 1)
        c_temporal = self.norm(self.dotatt(chunked_x, chunked_x, chunked_x)[0] + chunked_x)
        c_temporal = torch.mean(c_temporal, dim=-2, keepdim=True)
        # c_temporal = self.norm(self.dotatt(chunked_x, c_temporal, c_temporal)[0] + chunked_x)
        # print('c_temporal',c_temporal.size())
        # exit(0)
        c_temporal = c_temporal.reshape(n_batchsize, -1, self.hidden_dim)
        # c_temporal = self.norm(c_temporal)
        # c_temporal = self.dotatt(c_temporal, c_temporal, c_temporal)[0]
        # c_temporal = self.norm(c_temporal)
        # c_temporal = self.norm(c_temporal)
        # c_temporal = self.dotatt(c_temporal, c_temporal, c_temporal)[0]
        # print('c_temporal', c_temporal.size())
        # exit(0)

        return c_temporal


class ChunkedConvblock_A(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 output_dim=128,
                 n_stages=0,
                 n_steps=0,
                 device=None
                 ):
        super(ChunkedConvblock_A, self).__init__()
        # assert input_size != None and isinstance(input_size, int), 'fill in correct input_size'
        self.n_stages = n_stages
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.conv1d_temporal1 = torch.nn.Conv1d(in_channels=self.hidden_dim,
                                                out_channels=self.output_dim,
                                                kernel_size=2,
                                                dilation=2,
                                                stride=1,
                                                padding=1,
                                                )
        # self.dotatt = Attention()
        self.norm = torch.nn.LayerNorm(normalized_shape=self.hidden_dim)
        # self.max_pooling = torch.nn.AvgPool1d(kernel_size=int(self.n_steps / self.n_stages), stride=1)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout()

    def forward(self, x):
        """
        x, shape (batchsize, n_steps, outputdim)
        output, shape (batchsize, n_stages, outputdim)
        """
        x_size = x.size()
        n_batchsize, n_timestep, n_orifeatdim = x_size[0], x_size[1], x_size[2]
        assert n_timestep % self.n_stages == 0, "length of the sequence must be divisible by n_stages"
        chunked_x = torch.reshape(x, [n_batchsize*self.n_stages, int(n_timestep/self.n_stages), n_orifeatdim])
        c_temporal = chunked_x.permute(0, 2, 1)
        # c_temporal = self.conv1d_temporal2(self.drop(self.relu(self.conv1d_temporal1(c_temporal))))
        # c_temporal = self.relu(c_temporal)
        # print('c_temporal', c_temporal.size())
        c_temporal = self.drop(self.relu(self.conv1d_temporal1(c_temporal)))
        # print('c_temporal', c_temporal.size())
        # c_temporal = self.max_pooling(c_temporal)
        # c_temporal = torch.mean(c_temporal, dim=-1, keepdim=True)
        # print('c_temporal', c_temporal.size())
        c_temporal = c_temporal.permute(0, 2, 1)
        c_temporal = self.norm(c_temporal)
        c_temporal = torch.mean(c_temporal, dim=-2, keepdim=True)
        # c_temporal = self.norm(self.dotatt(chunked_x, c_temporal, c_temporal)[0] + chunked_x)
        # print('c_temporal',c_temporal.size())
        # exit(0)
        c_temporal = c_temporal.reshape(n_batchsize, -1, self.hidden_dim)
        # c_temporal = self.norm(c_temporal)
        # c_temporal = self.dotatt(c_temporal, c_temporal, c_temporal)[0]
        # c_temporal = self.norm(c_temporal)
        # c_temporal = self.norm(c_temporal)
        # c_temporal = self.dotatt(c_temporal, c_temporal, c_temporal)[0]
        # print('c_temporal', c_temporal.size())
        # exit(0)

        return c_temporal


class ChunkedConvblock_CA(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 output_dim=128,
                 n_stages=0,
                 n_steps=0,
                 device=None
                 ):
        super(ChunkedConvblock_CA, self).__init__()
        # assert input_size != None and isinstance(input_size, int), 'fill in correct input_size'
        self.n_stages = n_stages
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        # self.conv1d_temporal1 = torch.nn.Conv1d(in_channels=self.hidden_dim,
        #                                         out_channels=self.output_dim,
        #                                         kernel_size=2,
        #                                         dilation=2,
        #                                         stride=1,
        #                                         padding=1,
        #                                         )
        # self.dotatt = Attention()
        self.norm = torch.nn.LayerNorm(normalized_shape=self.hidden_dim)
        # self.max_pooling = torch.nn.AvgPool1d(kernel_size=int(self.n_steps / self.n_stages), stride=1)
        # self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout()

    def forward(self, x):
        """
        x, shape (batchsize, n_steps, outputdim)
        output, shape (batchsize, n_stages, outputdim)
        """
        x_size = x.size()
        n_batchsize, n_timestep, n_orifeatdim = x_size[0], x_size[1], x_size[2]
        assert n_timestep % self.n_stages == 0, "length of the sequence must be divisible by n_stages"
        chunked_x = torch.reshape(x, [n_batchsize*self.n_stages, int(n_timestep/self.n_stages), n_orifeatdim])
        # c_temporal = chunked_x.permute(0, 2, 1)
        # c_temporal = self.conv1d_temporal2(self.drop(self.relu(self.conv1d_temporal1(c_temporal))))
        # c_temporal = self.relu(c_temporal)
        # print('c_temporal', c_temporal.size())
        # c_temporal = self.drop(self.relu(self.conv1d_temporal1(c_temporal)))
        # print('c_temporal', c_temporal.size())
        # c_temporal = self.max_pooling(c_temporal)
        # c_temporal = torch.mean(c_temporal, dim=-1, keepdim=True)
        # print('c_temporal', c_temporal.size())
        # c_temporal = c_temporal.permute(0, 2, 1)
        c_temporal = self.norm(chunked_x)
        c_temporal = torch.mean(c_temporal, dim=-2, keepdim=True)
        # c_temporal = self.norm(self.dotatt(chunked_x, c_temporal, c_temporal)[0] + chunked_x)
        # print('c_temporal',c_temporal.size())
        # exit(0)
        c_temporal = c_temporal.reshape(n_batchsize, -1, self.hidden_dim)
        # c_temporal = self.norm(c_temporal)
        # c_temporal = self.dotatt(c_temporal, c_temporal, c_temporal)[0]
        # c_temporal = self.norm(c_temporal)
        # c_temporal = self.norm(c_temporal)
        # c_temporal = self.dotatt(c_temporal, c_temporal, c_temporal)[0]
        # print('c_temporal', c_temporal.size())
        # exit(0)

        return c_temporal


class StageEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=128,
                 nhead=8,
                 device=None
                 ):
        super(StageEncoderLayer, self).__init__()
        # assert input_size != None and isinstance(input_size, int), 'fill in correct input_size'
        self.hidden_dim = d_model
        self.nhead = nhead
        self.device = device
        self.qkvatt = MultiHeadedAttention(d_model=self.hidden_dim, h=self.nhead, dropout=0.5)
        self.mlp1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm1 = torch.nn.LayerNorm(normalized_shape=self.hidden_dim)
        self.norm2 = torch.nn.LayerNorm(normalized_shape=self.hidden_dim)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, query, key, value):
        out = self.qkvatt(query=query, key=key, value=value)
        src = query + self.dropout1(out)
        src = self.norm1(src)
        src2 = self.dropout2(self.relu(self.mlp1(src)))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class StageEncoder(nn.Module):
    def __init__(self,
                 input_dim=None,
                 hidden_dim=128,
                 output_dim=128,
                 n_stages=0,
                 n_steps=0,
                 n_layers=6,
                 device=None
                 ):
        super(StageEncoder, self).__init__()
        self.n_stages = n_stages
        self.n_steps = n_steps
        self.n_layers = n_layers
        self.input_size = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        self.embed_func = torch.nn.Linear(in_features=self.input_size, out_features=self.hidden_dim)
        self.cv_block_k = ChunkedConvblock(hidden_dim=self.hidden_dim,
                                 output_dim=self.output_dim,
                                 n_steps=self.n_steps,
                                 n_stages=self.n_stages)

        self.encoderlayer = StageEncoderLayer(d_model=self.hidden_dim, nhead=8)
        self.core = torch.nn.ModuleList([copy.deepcopy(self.encoderlayer) for i in range(self.n_layers)])


    def forward(self, input_data):

        """

        Parameters

        ----------
        input_data = {
                      'X': shape (batchsize, n_timestep, n_featdim)
                      'M': shape (batchsize, n_timestep)
                     }

        Return

        ----------

        all_output, shape (batchsize, n_timestep, n_labels)

            predict output of each time step

        cur_output, shape (batchsize, n_labels)

            predict output of last time step


        """
        X = input_data['X']
        M = input_data['M']
        cur_M = input_data['cur_M']
        n_batchsize, n_timestep, n_orifeatdim = X.shape

        assert n_timestep % self.n_stages == 0, "length of the sequence must be divisible by n_stages"

        _ori_X = X.view(-1, self.input_size)
        X = self.embed_func(_ori_X)
        X = X.reshape(n_batchsize, n_timestep, self.hidden_dim)

        c_temporal_k = self.cv_block_k(X)
        c_temporal_v = c_temporal_k
        outputs = self.core[0](X, c_temporal_k, c_temporal_v)
        for i in range(1, self.n_layers, 1):
            outputs = self.core[i](outputs, c_temporal_k, c_temporal_v)

        all_output = outputs
        cur_output = (outputs * cur_M.unsqueeze(-1)).sum(dim=1)
        return all_output, cur_output


class StageEncoder_C(nn.Module):
    def __init__(self,
                 input_dim=None,
                 hidden_dim=128,
                 output_dim=128,
                 n_stages=0,
                 n_steps=0,
                 n_layers=6,
                 device=None
                 ):
        super(StageEncoder_C, self).__init__()
        self.n_stages = n_stages
        self.n_steps = n_steps
        self.n_layers = n_layers
        self.input_size = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        self.embed_func = torch.nn.Linear(in_features=self.input_size, out_features=self.hidden_dim)
        self.cv_block_k = ChunkedConvblock_C(hidden_dim=self.hidden_dim,
                                 output_dim=self.output_dim,
                                 n_steps=self.n_steps,
                                 n_stages=self.n_stages)

        self.encoderlayer = StageEncoderLayer(d_model=self.hidden_dim, nhead=8)
        self.core = torch.nn.ModuleList([copy.deepcopy(self.encoderlayer) for i in range(self.n_layers)])


    def forward(self, input_data):

        """

        Parameters

        ----------
        input_data = {
                      'X': shape (batchsize, n_timestep, n_featdim)
                      'M': shape (batchsize, n_timestep)
                     }

        Return

        ----------

        all_output, shape (batchsize, n_timestep, n_labels)

            predict output of each time step

        cur_output, shape (batchsize, n_labels)

            predict output of last time step


        """
        X = input_data['X']
        M = input_data['M']
        cur_M = input_data['cur_M']
        n_batchsize, n_timestep, n_orifeatdim = X.shape

        assert n_timestep % self.n_stages == 0, "length of the sequence must be divisible by n_stages"

        _ori_X = X.view(-1, self.input_size)
        X = self.embed_func(_ori_X)
        X = X.reshape(n_batchsize, n_timestep, self.hidden_dim)

        c_temporal_k = self.cv_block_k(X)
        c_temporal_v = c_temporal_k
        outputs = self.core[0](X, c_temporal_k, c_temporal_v)
        for i in range(1, self.n_layers, 1):
            outputs = self.core[i](outputs, c_temporal_k, c_temporal_v)

        all_output = outputs
        cur_output = (outputs * cur_M.unsqueeze(-1)).sum(dim=1)
        return all_output, cur_output


class StageEncoder_A(nn.Module):
    def __init__(self,
                 input_dim=None,
                 hidden_dim=128,
                 output_dim=128,
                 n_stages=0,
                 n_steps=0,
                 n_layers=6,
                 device=None
                 ):
        super(StageEncoder_A, self).__init__()
        self.n_stages = n_stages
        self.n_steps = n_steps
        self.n_layers = n_layers
        self.input_size = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        self.embed_func = torch.nn.Linear(in_features=self.input_size, out_features=self.hidden_dim)
        self.cv_block_k = ChunkedConvblock_A(hidden_dim=self.hidden_dim,
                                 output_dim=self.output_dim,
                                 n_steps=self.n_steps,
                                 n_stages=self.n_stages)

        self.encoderlayer = StageEncoderLayer(d_model=self.hidden_dim, nhead=8)
        self.core = torch.nn.ModuleList([copy.deepcopy(self.encoderlayer) for i in range(self.n_layers)])


    def forward(self, input_data):

        """

        Parameters

        ----------
        input_data = {
                      'X': shape (batchsize, n_timestep, n_featdim)
                      'M': shape (batchsize, n_timestep)
                     }

        Return

        ----------

        all_output, shape (batchsize, n_timestep, n_labels)

            predict output of each time step

        cur_output, shape (batchsize, n_labels)

            predict output of last time step


        """
        X = input_data['X']
        M = input_data['M']
        cur_M = input_data['cur_M']
        n_batchsize, n_timestep, n_orifeatdim = X.shape

        assert n_timestep % self.n_stages == 0, "length of the sequence must be divisible by n_stages"

        _ori_X = X.view(-1, self.input_size)
        X = self.embed_func(_ori_X)
        X = X.reshape(n_batchsize, n_timestep, self.hidden_dim)

        c_temporal_k = self.cv_block_k(X)
        c_temporal_v = c_temporal_k
        outputs = self.core[0](X, c_temporal_k, c_temporal_v)
        for i in range(1, self.n_layers, 1):
            outputs = self.core[i](outputs, c_temporal_k, c_temporal_v)

        all_output = outputs
        cur_output = (outputs * cur_M.unsqueeze(-1)).sum(dim=1)
        return all_output, cur_output


class StageEncoder_CA(nn.Module):
    def __init__(self,
                 input_dim=None,
                 hidden_dim=128,
                 output_dim=128,
                 n_stages=0,
                 n_steps=0,
                 n_layers=6,
                 device=None
                 ):
        super(StageEncoder_CA, self).__init__()
        self.n_stages = n_stages
        self.n_steps = n_steps
        self.n_layers = n_layers
        self.input_size = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        self.embed_func = torch.nn.Linear(in_features=self.input_size, out_features=self.hidden_dim)
        self.cv_block_k = ChunkedConvblock_CA(hidden_dim=self.hidden_dim,
                                 output_dim=self.output_dim,
                                 n_steps=self.n_steps,
                                 n_stages=self.n_stages)

        self.encoderlayer = StageEncoderLayer(d_model=self.hidden_dim, nhead=8)
        self.core = torch.nn.ModuleList([copy.deepcopy(self.encoderlayer) for i in range(self.n_layers)])


    def forward(self, input_data):

        """

        Parameters

        ----------
        input_data = {
                      'X': shape (batchsize, n_timestep, n_featdim)
                      'M': shape (batchsize, n_timestep)
                     }

        Return

        ----------

        all_output, shape (batchsize, n_timestep, n_labels)

            predict output of each time step

        cur_output, shape (batchsize, n_labels)

            predict output of last time step


        """
        X = input_data['X']
        M = input_data['M']
        cur_M = input_data['cur_M']
        n_batchsize, n_timestep, n_orifeatdim = X.shape

        assert n_timestep % self.n_stages == 0, "length of the sequence must be divisible by n_stages"

        _ori_X = X.view(-1, self.input_size)
        X = self.embed_func(_ori_X)
        X = X.reshape(n_batchsize, n_timestep, self.hidden_dim)

        c_temporal_k = self.cv_block_k(X)
        c_temporal_v = c_temporal_k
        outputs = self.core[0](X, c_temporal_k, c_temporal_v)
        for i in range(1, self.n_layers, 1):
            outputs = self.core[i](outputs, c_temporal_k, c_temporal_v)

        all_output = outputs
        cur_output = (outputs * cur_M.unsqueeze(-1)).sum(dim=1)
        return all_output, cur_output
