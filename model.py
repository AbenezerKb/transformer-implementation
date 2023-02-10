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


class EncoderBlock(nn.Module):

    def __init__(self, features: int, 
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](
            x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, features: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module(
            [ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.
                                         self_attention_block(x, x, x,
                                                              tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.
                                         self_attention_block(x,
                                                              encoder_output,
                                                              encoder_output,
                                                              src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.Module) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, dim_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layers = projection_layer

    def encode(self, src, src_mask):
        src = self.embed(src)
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)
        return self.encoder(src, src_mask)

    def decode(self, encode_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encode_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layers(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                      src_seq_len: int, tgt_seq_len: int, dim_model: int = 512,
                      N: int = 6, h: int = 8, dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
    # Embedding layers
    src_embed = InputEmbeddings(dim_model, src_vocab_size)
    tgt_embed = InputEmbeddings(dim_model, tgt_vocab_size)

    # Positional Encoding Layers
    src_pos = PositionalEncoding(dim_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(dim_model, tgt_seq_len, dropout)

    # Encoder Blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(dim_model,
                                                               h, dropout)
        feed_forward_block = FeedForwardBlock(dim_model, d_ff,
                                              dropout)
        encoder_block = EncoderBlock(dim_model, encoder_self_attention_block,
                                     feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(dim_model,
                                                               h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(dim_model,
                                                                h, dropout)
        feed_forward_block = FeedForwardBlock(dim_model, d_ff, dropout)
        decoder_blocks = DecoderBlock(dim_model, decoder_self_attention_block,
                                      decoder_cross_attention_block,
                                      feed_forward_block, dropout)
        decoder_blocks.append(decoder_blocks)

    # Encoder and Decoder
    encoder = Encoder(dim_model, nn.ModuleList(encoder_block))
    decoder = Decoder(dim_model, nn.ModuleList(decoder_blocks))

    # Projection Layer
    projection_layer = ProjectionLayer(dim_model, tgt_vocab_size)

    # Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed,
                              src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
