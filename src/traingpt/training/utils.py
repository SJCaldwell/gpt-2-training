import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(
    model: nn.Module, optimizer_name: str, lr: float, weight_decay: float
) -> Optimizer:
    """Create optimizer with proper weight decay handling"""
    # Separate parameters that should and shouldn't use weight decay
    decay = set()
    no_decay = set()

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn  # Full param name

            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (nn.Linear, nn.Embedding)):
                decay.add(fpn)
            else:
                no_decay.add(fpn)

    # Validate all params are accounted for
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), f"Parameters {inter_params} made it into both decay/no_decay sets!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"Parameters {param_dict.keys() - union_params} were not separated into decay/no_decay set!"

    # Create optimizer groups
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]

    # Create optimizer
    if optimizer_name == "adamw":
        optimizer = AdamW(optim_groups, lr=lr)
    elif optimizer_name == "adam":
        optimizer = Adam(optim_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def get_scheduler(
    optimizer: Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    """Create learning rate scheduler with linear warmup and cosine decay"""

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 0.5 * (
            1.0
            + math.cos(
                math.pi * float(step - warmup_steps) / float(total_steps - warmup_steps)
            )
        )

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    step: int,
    best_loss: float,
    output_dir: Path,
) -> None:
    """Save training checkpoint"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    # Save optimizer
    optim_path = output_dir / "optimizer.pt"
    torch.save(optimizer.state_dict(), optim_path)

    # Save scheduler
    sched_path = output_dir / "scheduler.pt"
    torch.save(scheduler.state_dict(), sched_path)

    # Save training state
    state = {"step": step, "best_loss": best_loss}
    state_path = output_dir / "training_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer],
    scheduler: Optional[LambdaLR],
    checkpoint_dir: Path,
) -> tuple[int, float]:
    """Load training checkpoint"""
    # Load model
    model_path = checkpoint_dir / "model.pt"
    model.load_state_dict(torch.load(model_path))

    # Load optimizer
    if optimizer is not None:
        optim_path = checkpoint_dir / "optimizer.pt"
        optimizer.load_state_dict(torch.load(optim_path))

    # Load scheduler
    if scheduler is not None:
        sched_path = checkpoint_dir / "scheduler.pt"
        scheduler.load_state_dict(torch.load(sched_path))

    # Load training state
    state_path = checkpoint_dir / "training_state.json"
    with open(state_path) as f:
        state = json.load(f)

    return state["step"], state["best_loss"]
