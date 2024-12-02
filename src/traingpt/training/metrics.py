from typing import Dict

import torch


class TrainingMetrics:
    def __init__(self):
        self.best_loss = float("inf")

    def compute_training_metrics(
        self, logits: torch.Tensor, targets: torch.Tensor, loss: float
    ) -> Dict[str, float]:
        """Compute training metrics"""
        with torch.no_grad():
            # Convert logits to predictions
            preds = torch.argmax(logits, dim=-1)

            # Calculate accuracy
            accuracy = (preds == targets).float().mean().item()

            # Calculate perplexity
            perplexity = torch.exp(torch.tensor(loss)).item()

            # Update best loss
            self.best_loss = min(self.best_loss, loss)

            # Calculate token-level metrics
            non_pad_mask = targets != -100  # Assuming -100 is pad token
            n_correct = (preds == targets)[non_pad_mask].sum().item()
            n_tokens = non_pad_mask.sum().item()

            return {
                "loss": loss,
                "perplexity": perplexity,
                "accuracy": accuracy,
                "tokens_per_sec": n_tokens,  # Will be averaged over logging interval
                "learning_rate": self._get_last_lr(),
                "best_loss": self.best_loss,
            }

    @staticmethod
    def _get_last_lr():
        """Helper to get last learning rate for logging"""
        try:
            import wandb

            if wandb.run is not None:
                return wandb.run.summary.get("learning_rate", 0.0)
            else:
                return 0.0
        except:  # noqa: E722
            return 0.0
