from datetime import datetime
from pathlib import Path

import typer
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from rich.console import Console

from traingpt.config.model_config import GPT_CONFIG_124M, GPT2Config
from traingpt.config.training_config import TrainingConfig
from traingpt.data.dataset import GPTDataset
from traingpt.model.gpt2 import GPT2
from traingpt.training.trainer import Trainer

console = Console()
logger = get_logger(__name__)


def train(
    model_config_path: Path = typer.Argument(..., help="Path to model config JSON"),
    train_config_path: Path = typer.Argument(..., help="Path to training config JSON"),
    output_dir: Path = typer.Option(Path("outputs"), help="Base output directory"),
) -> None:
    """Train a GPT model with specified configs"""

    # Load configs
    if not model_config_path.exists():
        model_config = GPT_CONFIG_124M
    else:
        model_config = GPT2Config.load(model_config_path)
    train_config = TrainingConfig.load(train_config_path)

    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if train_config.mixed_precision else "no",
        gradient_accumulation_steps=train_config.grad_accum_steps,
    )

    # Set random seed
    set_seed(train_config.seed)

    # Initialize wandb on main process only
    if accelerator.is_main_process:
        # Generate run name if not provided
        if not train_config.experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            train_config.experiment_name = (
                f"gpt2_{model_config.embedding_dim}_{timestamp}"
            )

        # Create output directory
        run_dir = output_dir / train_config.experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)
        # Save configs
        model_config.save(run_dir / "model_config.json")
        train_config.save(run_dir / "train_config.json")

        # Start wandb run
        wandb.init(
            project=train_config.project_name,
            name=train_config.experiment_name,
            tags=train_config.tags,
            config={
                "model": model_config.to_dict(),
                "training": train_config.to_dict(),
            },
        )

    # Initialize model and dataset
    model = GPT2(model_config)
    dataset = GPTDataset(data_path=train_config.train_path)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_config=train_config,
        accelerator=accelerator,
        output_dir=run_dir,
    )

    # Train
    try:
        trainer.train(dataset)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if accelerator.is_main_process:
            wandb.finish(exit_code=1)
        raise

    # Finish wandb run
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    typer.run(train)
