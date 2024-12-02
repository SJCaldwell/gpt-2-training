import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TrainingConfig(BaseModel):
    # Experiment tracking
    project_name: str = Field(default="traingpt", description="W&B project name")
    experiment_name: Optional[str] = Field(
        default=None, description="W&B run name. If None, will be auto-generated"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for W&B run")

    # Training hyperparameters
    learning_rate: float = Field(default=3e-4, gt=0.0)
    batch_size: int = Field(default=32, gt=0)
    grad_accum_steps: int = Field(
        default=1, description="Gradient accumulation steps", gt=0
    )
    max_steps: int = Field(default=50000, gt=0)
    warmup_steps: int = Field(default=1000, ge=0)
    eval_every: int = Field(default=500, gt=0)
    save_every: int = Field(default=1000, gt=0)

    # Optimizer settings
    optimizer: Literal["adamw", "adam"] = Field(default="adamw")
    weight_decay: float = Field(default=0.1, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)

    # Data settings
    train_path: Path = Field(description="Path to training data")
    val_path: Optional[Path] = Field(
        default=None,
        description="Path to validation data. If None, will use train/val split",
    )
    val_split: float = Field(
        default=0.1,
        description="Validation split if val_path not provided",
        ge=0.0,
        le=1.0,
    )

    # System settings
    mixed_precision: bool = Field(default=True)
    seed: int = Field(default=1337)

    @field_validator("train_path", "val_path")
    def check_path_exists(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            config_dict = self.model_dump()
            # Convert paths to strings
            config_dict["train_path"] = str(config_dict["train_path"])
            if config_dict["val_path"]:
                config_dict["val_path"] = str(config_dict["val_path"])
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        with open(path) as f:
            return cls.model_validate(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for W&B logging"""
        config_dict = self.model_dump()
        config_dict["train_path"] = str(config_dict["train_path"])
        if config_dict["val_path"]:
            config_dict["val_path"] = str(config_dict["val_path"])
        return config_dict
