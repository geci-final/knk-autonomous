import torch
import torch.nn as nn


class MhAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MhAttention, self).__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        batch_size = query.size(0)

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        query = query.view(batch_size, -1, self.num_heads,
                           self.depth).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.num_heads,
                       self.depth).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.num_heads,
                           self.depth).permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.permute(
            0, 1, 3, 2)) / (self.depth ** 0.5)
        scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.matmul(attention, value)

        out = out.permute(0, 2, 1, 3).contiguous().view(
            batch_size, -1, self.d_model)

        out = self.dense(out)
        return out
