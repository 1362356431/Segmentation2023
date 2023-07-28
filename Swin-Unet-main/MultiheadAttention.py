import numpy as np
import torch
import math


import clones
from d2l.torch import d2l
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

input = torch.randn(3, 4)
print("input=", input)
b = F.softmax(input, dim=0)  # 按列SoftMax,列和为1（即0维度进行归一化）
print("b=", b)

c = F.softmax(input, dim=1)  # 按行SoftMax,行和为1（即1维度进行归一化）
print("c=", c)


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        # d_model:词嵌入的维度 vocab:词表的大小
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


d_model = 512
vocab = 1000

x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 444, 333, 222]]))

emb = Embedding(d_model, vocab)
embr = emb(x)
print(("embr:", embr))
print(embr.shape)


class PositionEncodeing(nn.Module):
    def __init__(self, d_modle, drop_out, max_len):
        super(PositionEncodeing, self).__init__()

        # 实例化drop_out
        self.dropout = nn.Dropout(p=drop_out)
        pe = torch.zeros(max_len, d_model)  # 位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1)  # 绝对位置矩阵
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_modle))  # 变化矩阵div_term，跳跃式初始化

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 扩张为三维张量

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


d_model = 512
drop_out = 0.1
max_len = 60

x = embr
pe = PositionEncodeing(d_modle=d_model, drop_out=drop_out, max_len=max_len)
pe_result = pe(x)
print(pe_result)
print(pe_result.shape)


def subsequent_mask(size):
    attn_shape = (1, size, size)

    # 形成上三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(1 - subsequent_mask)


size = 5
sm = subsequent_mask(size)
print("sm", sm)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


query = key = value = pe_result
attn, p_attn = attention(query, key, value)
print('attn', attn)
print(attn.shape)
print('pattn:', p_attn)
print(p_attn.shape)


# @save
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


# @save
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


# @save
class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


# 前馈全连接层

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


# 规范化层
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        # features代表词嵌入的维度
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(features=size)
        self.dropout = nn.Dropout(p=dropout)
        self.size = size

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))





class EncoderLayer(nn.Module):
    def __init__(self, self_attn, feed_forward, drop_out):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, drop_out), 3)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder, self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)

        return self.norm(x)


