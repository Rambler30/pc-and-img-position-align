import torch
import tifffile
import torch.nn as nn
import math
from pathlib import Path

__all__ = ["MultiHeadAttention", "SelfAttention"]

class SelfAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # `queries` 的形状：(`batch_size`, 查询的个数, `d`)
    # `keys` 的形状：(`batch_size`, “键－值”对的个数, `d`)
    # `values` 的形状：(`batch_size`, “键－值”对的个数, 值的维度)
    # `valid_lens` 的形状: (`batch_size`,) 或者 (`batch_size`, 查询的个数)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # 设置 `transpose_b=True` 为了交换 `keys` 的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = torch.softmax(scores, dim=2)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = SelfAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)

        output = self.attention(queries, keys, values)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

    @staticmethod
    def transpose_qkv(X, num_heads):
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    @staticmethod
    def transpose_output(X, num_heads):
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

batch_size, num_queries, num_hiddens, num_heads  = 1, 4, 100, 1
root = Path(r"C:\data\GRSS\2018IEEE_Contest\Phase2\TrainingGT\hs")
path = root / "hs_label.tif"
img = tifffile.imread(path)
hs, label = img[:10,:,:48],img[:,:,-1]
flatten_hs = hs.reshape((-1, hs.shape[-1]))
flatten_label = label.reshape((-1, label.shape[-1]))
num_queries, num_hiddens = flatten_hs.shape
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
X = torch.ones((batch_size, num_queries, num_hiddens))
flatten_hs = flatten_hs.astype('float32')
flatten_hs = torch.from_numpy(flatten_hs)
ans = attention(flatten_hs, X, X)
print(ans.shape)
