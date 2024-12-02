import torch
import torch.nn as nn

from traingpt.config.model_config import GPT2Config

from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.attn = MultiHeadAttention(
            dim_in=cfg.embedding_dim,
            dim_out=cfg.embedding_dim,
            context_length=cfg.context_length,
            dropout=cfg.drop_rate,
            num_heads=cfg.num_heads,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.embedding_dim)
        self.norm2 = LayerNorm(cfg.embedding_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        shortcut = x  # identity shortcut for residual connection
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # residual connection

        shortcut = x  # identity shortcut for residual connection
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class FeedForward(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.embedding_dim, 4 * cfg.embedding_dim),
            GELU(),
            nn.Linear(4 * cfg.embedding_dim, cfg.embedding_dim),
        )

    def forward(self, x):
        return self.layers(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.bias
