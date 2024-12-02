import json
import logging
from pathlib import Path
from typing import List, Union

import tiktoken
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(
        self,
        context_length: int = 1024,
        encoding_name: str = "gpt2",  # or "p50k_base", "r50k_base", "cl100k_base"
    ):
        self.context_length = context_length
        self.encoding = tiktoken.get_encoding(encoding_name)

        # Special token IDs for GPT-2 encoding
        self.eos_token_id = self.encoding.eot_token
        self.special_tokens = {self.eos_token_id}

        logger.info(f"Initialized tokenizer with encoding {encoding_name}")
        logger.info(f"Vocabulary size: {self.encoding.n_vocab}")

    def preprocess_file(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        chunk_size: int = 1000,  # Process this many lines at once
    ) -> None:
        """Preprocess a text file into tokenized chunks"""
        input_file = Path(input_file)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Read and preprocess in chunks
        all_token_chunks = []
        total_tokens = 0

        with open(input_file, "r", encoding="utf-8") as f:
            # Count lines for progress bar
            total_lines = sum(1 for _ in f)
            f.seek(0)  # Reset file pointer

            # Process in chunks
            pbar = tqdm(total=total_lines, desc="Preprocessing")
            current_chunk = []

            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                current_chunk.append(line)

                if len(current_chunk) >= chunk_size:
                    tokens, n_tokens = self._process_chunk(current_chunk)
                    all_token_chunks.extend(tokens)
                    total_tokens += n_tokens
                    current_chunk = []
                    pbar.update(chunk_size)

            # Process remaining lines
            if current_chunk:
                tokens, n_tokens = self._process_chunk(current_chunk)
                all_token_chunks.extend(tokens)
                total_tokens += n_tokens
                pbar.update(len(current_chunk))

            pbar.close()

        # Save preprocessed data
        logger.info(f"Saving {len(all_token_chunks)} sequences to {output_file}")
        torch.save(all_token_chunks, output_file)

        # Calculate and save statistics
        stats = {
            "num_sequences": len(all_token_chunks),
            "total_tokens": total_tokens,
            "context_length": self.context_length,
            "vocab_size": self.encoding.n_vocab,
            "unique_tokens": len(set(torch.cat(all_token_chunks).tolist())),
        }

        with open(output_file.with_suffix(".json"), "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Processing stats: {stats}")

    def _process_chunk(self, texts: List[str]) -> tuple[List[torch.Tensor], int]:
        """Process a chunk of text into token sequences"""
        # Join texts with newline
        full_text = "\n".join(texts)

        # Tokenize with tiktoken
        tokens = self.encoding.encode(full_text, disallowed_special=())

        # Add EOS token if not present
        if tokens[-1] != self.eos_token_id:
            tokens.append(self.eos_token_id)

        # Split into context_length chunks
        chunks = []
        total_tokens = 0

        for i in range(0, len(tokens), self.context_length):
            chunk = tokens[i : i + self.context_length]
            total_tokens += len(chunk)

            # Pad if necessary
            if len(chunk) < self.context_length:
                chunk = chunk + [self.eos_token_id] * (self.context_length - len(chunk))

            chunks.append(torch.tensor(chunk, dtype=torch.long))

        return chunks, total_tokens

    def decode(self, tokens: Union[torch.Tensor, List[int]]) -> str:
        """Decode token IDs back to text"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.encoding.decode(tokens)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        return self.encoding.encode(text, disallowed_special=())

    def token_count(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encode(text))


class TextDatasetBuilder:
    """Helper class to build datasets from raw text files"""

    def __init__(
        self, output_dir: Path, context_length: int = 1024, encoding_name: str = "gpt2"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = TextPreprocessor(
            context_length=context_length, encoding_name=encoding_name
        )

    def build_from_gutenberg(
        self,
        input_files: List[Path],
        output_name: str = "gutenberg",
        skip_existing: bool = True,
    ) -> Path:
        """Build dataset from Gutenberg text files"""
        output_path = self.output_dir / f"{output_name}.pt"
        if skip_existing and output_path.exists():
            logger.info(f"Dataset already exists at {output_path}")
            return output_path

        # Combine all files into one
        combined_text = self.output_dir / f"{output_name}_combined.txt"
        with open(combined_text, "w", encoding="utf-8") as outfile:
            for file in tqdm(input_files, desc="Combining files"):
                with open(file, "r", encoding="utf-8") as infile:
                    # Skip Gutenberg header/footer
                    text = infile.read()
                    start_idx = text.find("*** START OF THIS PROJECT GUTENBERG")
                    end_idx = text.find("*** END OF THIS PROJECT GUTENBERG")
                    if start_idx != -1 and end_idx != -1:
                        text = text[start_idx:end_idx]
                    outfile.write(text + "\n\n")

        # Preprocess combined file
        self.preprocessor.preprocess_file(combined_text, output_path)

        # Cleanup
        combined_text.unlink()

        return output_path


def get_tokenizer(encoding_name: str = "gpt2") -> tiktoken.Encoding:
    """Helper function to get tokenizer"""
    return tiktoken.get_encoding(encoding_name)
