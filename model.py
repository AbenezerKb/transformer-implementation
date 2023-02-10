import torch
import math
import torch.nn as nn


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, dim_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(seq_len, dim_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, dim_model, 2).
                             float()*(-math.log(10000) / dim_model))
        positional_encoding[:, 0::2] = torch.sin(position*div_term)
        positional_encoding[:, 0::2] = torch.sin(position*div_term)

        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        x = x+(self.
               positional_encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(features))  # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std+self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, dim_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dim_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, dim_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_Len, d_model) --> (batch, seq_Len, d_ff)
        #  --> (batch, Seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, dim_model: int, heads: int,
                 dropout_prob: float) -> None:
        super().__init__()
        self.dim_model = dim_model // heads
        self.dim_model = dim_model
        self.heads = heads
        self.query = nn.Linear(dim_model, dim_model)
        self.key = nn.Linear(dim_model, dim_model)
        self.value = nn.Linear(dim_model, dim_model)
        self.output = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout_prob)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        dim_key = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0,
                                                            float('-inf'))
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, query, key, value, mask):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = query.view(query.shape[0], query.shape[1],
                           self.heads, self.dim_key).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads,
                       self.dim_key).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1],
                           self.heads, self.dim_key).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contigous().view(x.shape[0], -1,
                                               self.heads * self.dim_key)


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
