from pathlib import Path
from typing import List

import tiktoken
import torch


def clean_gutenberg_text(text: str) -> str:
    """Remove Gutenberg header and footer from text."""
    start_idx = text.find("*** START OF THIS PROJECT GUTENBERG")
    end_idx = text.find("*** END OF THIS PROJECT GUTENBERG")
    if start_idx != -1 and end_idx != -1:
        return text[start_idx:end_idx]
    return text


def process_file(file_path: Path, encoding) -> List[int]:
    """Process a single file and return its tokens with EOS token."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Clean text
    text = clean_gutenberg_text(text)

    # Encode and add EOS token
    tokens = encoding.encode(text)
    print(f"  File has {len(tokens)} tokens")
    return tokens + [50256]  # Explicit GPT-2 EOT token


def prepare_data(input_dir: str, output_file: str, context_length: int = 1024):
    """Process all text files in a directory and save as concatenated chunks."""
    # Initialize tokenizer
    encoding = tiktoken.get_encoding("gpt2")

    # Get all text files in directory
    input_path = Path(input_dir)
    text_files = list(input_path.glob("*.txt"))

    if not text_files:
        raise ValueError(f"No .txt files found in {input_dir}")

    # Process all files and create chunks respecting document boundaries
    chunks = []
    current_chunk = []
    chunks_per_file = []  # Track number of chunks per file

    for file_path in text_files:
        print(f"\nProcessing {file_path.name}...")
        tokens = process_file(file_path, encoding)
        start_chunk_count = len(chunks)

        # Handle current tokens
        token_position = 0
        while token_position < len(tokens):
            # Fill the current chunk
            while len(current_chunk) < context_length and token_position < len(tokens):
                current_chunk.append(tokens[token_position])
                token_position += 1

            # If chunk is full, save it
            if len(current_chunk) == context_length:
                chunks.append(torch.tensor(current_chunk, dtype=torch.long))
                current_chunk = []

        # After each file, if we have a partial chunk, pad it
        if current_chunk:
            original_len = len(current_chunk)
            while len(current_chunk) < context_length:
                current_chunk.append(50256)  # Explicit GPT-2 EOT token
            chunks.append(torch.tensor(current_chunk, dtype=torch.long))
            current_chunk = []

        chunks_per_file.append(len(chunks) - start_chunk_count)
        print(f"  Created {chunks_per_file[-1]} chunks")

    # Save
    torch.save(chunks, output_file)
    print("\nProcessing complete:")
    for file_path, chunk_count in zip(text_files, chunks_per_file):
        print(f"{file_path.name}: {chunk_count} chunks")
    print(f"Total chunks: {len(chunks)}")

    # Print statistics for interesting chunks
    def print_chunk_stats(chunk, index, description=""):
        eot_count = (chunk == 50256).sum().item()
        print(f"\n{description} (Chunk {index}):")
        print(f"  Total tokens: {len(chunk)}")
        print(f"  EOT tokens: {eot_count}")
        if eot_count > 0:
            eot_positions = (chunk == 50256).nonzero().tolist()
            print(f"  EOT positions: {eot_positions}")
        print(f"  First 5 tokens: {chunk[:5].tolist()}")
        print(f"  Last 5 tokens: {chunk[-5:].tolist()}")

    print("\nChunk Statistics:")

    # First chunk
    print_chunk_stats(chunks[0], 0, "First chunk")

    # Last chunk of first file
    first_file_end = chunks_per_file[0] - 1
    print_chunk_stats(
        chunks[first_file_end], first_file_end, "Last chunk of first file"
    )

    # First chunk of second file
    print_chunk_stats(
        chunks[first_file_end + 1], first_file_end + 1, "First chunk of second file"
    )

    # Last chunk
    print_chunk_stats(chunks[-1], len(chunks) - 1, "Final chunk")


if __name__ == "__main__":
    input_dir = "data/raw"
    output_file = "data/processed/train.pt"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    prepare_data(input_dir, output_file)
