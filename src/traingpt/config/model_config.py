from typing import Dict, Any
from pydantic import BaseModel, Field, computed_field
import json
from pathlib import Path

# defaults to 124M size
class GPT2Config(BaseModel):
    vocab_size: int = Field(default=50257, description="Vocabulary size (num unique tokens) of the model", gt=0)
    context_length: int = Field(default=1024, description="Maximum sequence length")  
    embedding_dim: int = Field(default=768, description="Embedding dimension", gt=0)
    num_heads: int = Field(default=12, description="Number of attention heads", gt=0)
    num_layers: int = Field(default=12, description="Number of transformer block layers", gt=0)
    drop_rate: float = Field(default=0.1, description="Dropout rate", ge=0.0, le=1.0)
    qkv_bias: bool = Field(default=False, description="Whether to add bias to the qkv linear layers")

    @computed_field
    def head_dim(self) -> int:
        assert self.embedding_dim % self.num_heads == 0
        return self.embedding_dim // self.num_heads
    
    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f)
    
    @classmethod
    def load(cls, path: Path | str) -> "GPT2Config":
        with open(path) as f:
            return cls.model_validate(json.load(f))
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

# Pre-defined model sizes

GPT_CONFIG_124M = GPT2Config(
    vocab_size=50257,
    context_length=1024,
    embedding_dim=768,
    num_heads=12,
    num_layers=12,
    drop_rate=0.1,
    qkv_bias=False,
)

GPT_CONFIG_355M = GPT2Config(
    vocab_size=50257,
    context_length=1024,
    embedding_dim=1024,
    num_heads=16,
    num_layers=24,
    drop_rate=0.1,
    qkv_bias=False,
)