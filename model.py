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
