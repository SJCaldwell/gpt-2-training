import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

from ..config.training_config import TrainingConfig
from .preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class GPTDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        context_length: int = 1024,
    ):
        self.data_path = Path(data_path)
        self.context_length = context_length

        # Load tokenized data
        logger.info(f"Loading dataset from {self.data_path}")
        self.sequences = torch.load(self.data_path)

        # Load stats if available
        stats_path = self.data_path.with_suffix(".json")
        if stats_path.exists():
            import json

            with open(stats_path) as f:
                self.stats = json.load(f)
            logger.info(f"Dataset stats: {self.stats}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.sequences[idx]

        # Prepare input_ids and labels for casual language modeling
        # Input is the full sequence
        input_ids = tokens
        # Labels are the same as input_ids (predict next token)
        labels = tokens.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def create_dataloaders(
    config: TrainingConfig,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create training and validation dataloaders"""

    # Create dataset
    dataset = GPTDataset(config.train_path)

    # Split into train/val if needed
    if config.val_path is None and config.val_split > 0:
        val_size = int(len(dataset) * config.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset = dataset
        val_dataset = GPTDataset(config.val_path) if config.val_path else None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


class TextDatasetBuilder:
    """Helper class to build datasets from raw text files"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_from_gutenberg(
        self,
        input_files: List[Path],
        preprocessor: TextPreprocessor,
        output_name: str = "gutenberg",
    ) -> Path:
        """Build dataset from Gutenberg text files"""
        # Combine all files into one
        combined_text = self.output_dir / f"{output_name}_combined.txt"
        with open(combined_text, "w", encoding="utf-8") as outfile:
            for file in tqdm(input_files, desc="Combining files"):
                with open(file, "r", encoding="utf-8") as infile:
                    # Skip Gutenberg header
                    text = infile.read()
                    start_idx = text.find("*** START OF THIS PROJECT GUTENBERG")
                    end_idx = text.find("*** END OF THIS PROJECT GUTENBERG")
                    if start_idx != -1 and end_idx != -1:
                        text = text[start_idx:end_idx]
                    outfile.write(text + "\n\n")

        # Preprocess combined file
        output_path = self.output_dir / f"{output_name}.pt"
        preprocessor.preprocess_file(combined_text, output_path)

        # Cleanup
        combined_text.unlink()

        return output_path
