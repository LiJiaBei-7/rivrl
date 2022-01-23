import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, opt, dim_in_q, dim_in, dim_k, dim_v, dropout=0.2):
        super(SelfAttention, self).__init__()
        self.dim_in_q = dim_in_q
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(self.dim_in_q, self.dim_k, bias=False)
        self.linear_k = nn.Linear(self.dim_in, self.dim_k, bias=False)
        self.linear_v = nn.Linear(self.dim_in, self.dim_v, bias=False)
        self.fc_v = nn.Linear(self.dim_v, self.dim_in)
        self.layer_norm = nn.LayerNorm(dim_v, eps=1e-6)
        self._norm_fact = 1 / math.sqrt(self.dim_k)
        self.dropout = nn.Dropout(dropout)
        self.pooling = opt.pooling

    def forward(self, mask, x, y):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        if self.pooling == 'mean':
            residual = F.avg_pool1d(x.permute(0, 2, 1), x.size(1)).squeeze(2)
        else:
            residual = F.max_pool1d(x.permute(0, 2, 1), x.size(1)).squeeze(2)


        q = self.linear_q(y)
        k = self.linear_k(x)
        v = self.linear_v(x)

        attention_mask = mask.unsqueeze(1)
        attention_mask = (1.0 - attention_mask) * -10000.0  # padding的token置为-10000，exp(-1w)=0

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact

        attention_scores = dist + attention_mask
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        att = torch.bmm(attention_probs, v).squeeze(1)
        att = self.layer_norm(self.fc_v(att) + residual)

        return att



class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        del residual

        x = self.layer_norm(x)

        return x




class qkv_layer(nn.Module):
    def __init__(self, opt, q_input_dim, kv_input_dim, qkv_out_dim):
        super(qkv_layer, self).__init__()
        self.qk_dim = 512
        self.qkv = SelfAttention(opt, q_input_dim, kv_input_dim, self.qk_dim, qkv_out_dim)
        self.ffn = PositionwiseFeedForward(qkv_out_dim, qkv_out_dim * 2)

    def forward(self, mask, kv_data, q_data):

        qkv_out = self.qkv(mask, kv_data, q_data)
        qkv_out = self.ffn(qkv_out)

        return qkv_out


