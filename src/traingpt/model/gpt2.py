import torch
import torch.nn as nn
from traingpt.config.model_config import GPT2Config
from traingpt.model.blocks import TransformerBlock, LayerNorm

class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.embedding_dim)

        self.dropout = nn.Dropout(cfg.drop_rate)

        self.transformer_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.num_layers)])

        self.final_norm = LayerNorm(cfg.embedding_dim)

        self.out_head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)