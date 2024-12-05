import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from traingpt.config.model_config import GPT_CONFIG_124M, GPT2Config
from traingpt.config.training_config import TrainingConfig
from traingpt.data.dataset import create_dataloaders
from traingpt.model.gpt2 import GPT2
from traingpt.training.trainer import Trainer

logger = logging.getLogger(__name__)


def get_accelerator_config(train_config: TrainingConfig) -> dict:
    """Configure accelerator based on available hardware"""
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        # CPU only - no mixed precision
        logger.info("Running on CPU - disabling mixed precision")
        return {"mixed_precision": "no"}

    if torch.cuda.is_available():
        # NVIDIA GPU - can use mixed precision
        logger.info("Running on CUDA - enabling mixed precision")
        return {"mixed_precision": "fp16" if train_config.mixed_precision else "no"}

    if torch.backends.mps.is_available():
        # Apple Silicon - use default precision
        logger.info("Running on MPS - using default precision")
        return {"mixed_precision": "no"}

    return {"mixed_precision": "no"}


def train(
    train_config_path: Path,
    model_config_path: Optional[Path] = None,
    output_dir: Path = Path("outputs"),
) -> None:
    """Train a GPT model using config files"""

    # Load configs
    model_config = (
        GPT2Config.load(model_config_path)
        if model_config_path is not None
        else GPT_CONFIG_124M
    )
    train_config = TrainingConfig.load(train_config_path)

    # Setup accelerator based on hardware
    accelerator_config = get_accelerator_config(train_config)
    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.grad_accum_steps, **accelerator_config
    )

    # Log device info
    logger.info(f"Using device: {accelerator.device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")

    # Initialize model
    model = GPT2(model_config)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config=train_config, batch_size=train_config.batch_size
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_config=train_config,
        accelerator=accelerator,
        output_dir=output_dir,
    )

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument(
        "--train-config", "-t",
        type=Path,
        required=True,
        help="Path to training config JSON"
    )
    parser.add_argument(
        "--model-config", "-m",
        type=Path,
        help="Path to model config JSON. If not provided, uses GPT_CONFIG_124M"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("outputs"),
        help="Directory for checkpoints and logs"
    )
    
    args = parser.parse_args()
    train(
        train_config_path=args.train_config,
        model_config_path=args.model_config,
        output_dir=args.output_dir
    )
