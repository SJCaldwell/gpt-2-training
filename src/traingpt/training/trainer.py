from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm.auto import tqdm

import wandb

from ..config.training_config import TrainingConfig
from .metrics import TrainingMetrics
from .utils import create_optimizer, get_scheduler, save_checkpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_config: TrainingConfig,
        accelerator: Accelerator,
        output_dir: Path,
    ):
        self.model = model
        self.config = train_config
        self.accelerator = accelerator
        self.output_dir = output_dir
        self.step = 0

        if self.accelerator.is_main_process:
            wandb.init(
                project=train_config.project_name,
                name=train_config.experiment_name,
                config={
                    "model_config": model.config.__dict__
                    if hasattr(model, "config")
                    else {},
                    "train_config": train_config.model_dump(),
                },
            )

        # Initialize metrics tracker
        self.metrics = TrainingMetrics()

        # Setup optimizer
        self.optimizer = create_optimizer(
            model=model,
            optimizer_name=train_config.optimizer,
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        # Setup scheduler
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            warmup_steps=train_config.warmup_steps,
            total_steps=train_config.max_steps,
        )

        # Setup criterion
        self.criterion = nn.CrossEntropyLoss()

        # Prepare everything with accelerator
        (
            self.model,
            self.optimizer,
            self.scheduler,
        ) = accelerator.prepare(self.model, self.optimizer, self.scheduler)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()

        # Get inputs and targets on gpu
        input_ids = batch["input_ids"].to(self.accelerator.device)
        targets = batch["labels"].to(self.accelerator.device)

        # Forward pass
        with self.accelerator.accumulate(self.model):
            logits = self.model(input_ids)

            # Track memory after forward pass
            if torch.cuda.is_available() and self.step % 10 == 0:
                forward_memory = torch.cuda.memory_allocated()
                print(f"Memory after forward pass: {(forward_memory - initial_memory) / 1024**3:.2f}GB")
                forward_memory_in_gb = (forward_memory - initial_memory) / 1024**3
                wandb.log({"memory_after_forward_in_gb": forward_memory_in_gb, "batch_size": input_ids.size(0), "context_length": input_ids.size(1)}, step=self.step)


            # Calculate loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

            # Backward pass
            self.accelerator.backward(loss)

            if torch.cuda.is_available() and self.step % 10 == 0:
                backward_memory = torch.cuda.memory_allocated()
                print(f"Memory after backward pass: {(backward_memory - initial_memory) / 1024**3:.2f}GB")
                backward_memory_in_gb = (backward_memory - initial_memory) / 1024**3
                wandb.log({"memory_after_backward_in_gb": backward_memory_in_gb, "batch_size": input_ids.size(0), "context_length": input_ids.size(1)}, step=self.step)

            if self.config.grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # Calculate metrics
        with torch.no_grad():
            metrics = self.metrics.compute_training_metrics(
                logits=logits, targets=targets, loss=loss.item()
            )

        return metrics

    @torch.no_grad()
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluation loop"""
        self.model.eval()
        total_metrics = {}

        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(self.accelerator.device)
            targets = batch["labels"].to(self.accelerator.device)

            logits = self.model(input_ids)

            loss = self.criterion(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

            batch_metrics = self.metrics.compute_training_metrics(
                logits=logits, targets=targets, loss=loss.item()
            )

            # Accumulate metrics
            for k, v in batch_metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = []
                total_metrics[k].append(v)

        # Average metrics
        avg_metrics = {f"eval_{k}": sum(v) / len(v) for k, v in total_metrics.items()}

        self.model.train()
        return avg_metrics

    def train(self, train_dataloader, eval_dataloader=None):
        """Main training loop"""
        self.model.train()
        progress_bar = tqdm(
            range(self.config.max_steps),
            disable=not self.accelerator.is_local_main_process,
        )

        while self.step < self.config.max_steps:
            for batch in train_dataloader:
                # Training step
                metrics = self.train_step(batch)

                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix(**metrics)  # type: ignore

                # Log metrics
                if self.accelerator.is_main_process:
                    wandb.log(metrics, step=self.step)

                # Evaluation
                if (
                    eval_dataloader is not None
                    and self.step > 0
                    and self.step % self.config.eval_every == 0
                ):
                    eval_metrics = self.evaluate(eval_dataloader)
                    if self.accelerator.is_main_process:
                        wandb.log(eval_metrics, step=self.step)

                # Save checkpoint
                if (
                    self.step > 0
                    and self.step % self.config.save_every == 0
                    and self.accelerator.is_main_process
                ):
                    save_checkpoint(
                        self.accelerator.unwrap_model(self.model),
                        self.optimizer,
                        self.scheduler,
                        self.step,
                        self.metrics.best_loss,
                        self.output_dir / f"checkpoint-{self.step}",
                    )

                self.step += 1
                if self.step >= self.config.max_steps:
                    break

        # Save final checkpoint
        if self.accelerator.is_main_process:
            save_checkpoint(
                self.accelerator.unwrap_model(self.model),
                self.optimizer,
                self.scheduler,
                self.step,
                self.metrics.best_loss,
                self.output_dir / "checkpoint-final",
            )
