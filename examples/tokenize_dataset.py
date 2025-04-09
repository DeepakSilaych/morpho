#!/usr/bin/env python
"""
Example script for tokenizing and encoding a dataset using a pre-trained tokenizer.
"""

import sys
import os
import argparse
import json
from typing import List, Dict, Any, Optional

# Add parent directory to path to allow importing from the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vocab.trainer import TokenizerTrainer
from encoders.encoder import Encoder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tokenize and encode a dataset using a pre-trained tokenizer."
    )
    parser.add_argument(
        "--tokenizer", 
        "-t", 
        type=str, 
        required=True, 
        help="Path to tokenizer model file")
    parser.add_argument(
        "--input", 
        "-i", 
        type=str, 
        required=True, 
        help="Path to input dataset file (JSON or JSONL)")
    parser.add_argument(
        "--output", 
        "-o", 
        type=str, 
        required=True, 
        help="Path to output encoded dataset file")
    parser.add_argument(
        "--text-field", 
        "-f", 
        type=str, 
        default="text", 
        help="Field name for text in input dataset")
    parser.add_argument(
        "--max-length", 
        "-m", 
        type=int, 
        default=512, 
        help="Maximum sequence length")
    parser.add_argument(
        "--add-special-tokens", 
        action="store_true", 
        help="Add special tokens (BOS/EOS)")
    parser.add_argument(
        "--truncate", 
        action="store_true", 
        help="Truncate sequences to max length")
    return parser.parse_args()


def load_dataset(path: str, text_field: str = "text") -> List[Dict[str, Any]]:
    """Load dataset from JSON or JSONL file."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        if content.startswith('[') and content.endswith(']'):
            # JSON array format
            data = json.loads(content)
        else:
            # JSONL format (one JSON object per line)
            data = [json.loads(line) for line in content.split('\n') if line.strip()]
    
    # Validate dataset
    for item in data:
        if text_field not in item:
            print(f"Warning: Missing text field '{text_field}' in some items.")
            break
    
    return data


def save_dataset(data: List[Dict[str, Any]], path: str, format: str = "json"):
    """Save dataset to JSON or JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        if format.lower() == "jsonl":
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)


def process_dataset(
    dataset: List[Dict[str, Any]],
    encoder: Encoder,
    text_field: str = "text",
    max_length: Optional[int] = None,
    add_special_tokens: bool = False,
    truncate: bool = False
) -> List[Dict[str, Any]]:
    """Process dataset by tokenizing and encoding text."""
    processed_dataset = []
    
    for item in dataset:
        if text_field not in item:
            # Skip items without text field
            processed_dataset.append(item.copy())
            continue
        
        text = item[text_field]
        
        # Encode text
        ids = encoder.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncate,
            max_length=max_length
        )
        
        # Tokenize text (for reference)
        tokens = encoder.tokenizer.tokenize(text)
        
        # Create processed item
        processed_item = item.copy()
        processed_item['input_ids'] = ids
        processed_item['tokens'] = tokens
        processed_item['length'] = len(ids)
        
        processed_dataset.append(processed_item)
    
    return processed_dataset


def print_stats(dataset: List[Dict[str, Any]]):
    """Print statistics about the processed dataset."""
    if not dataset:
        print("Empty dataset.")
        return
    
    # Calculate sequence length statistics
    lengths = [item.get('length', 0) for item in dataset if 'length' in item]
    
    if not lengths:
        print("No encoded sequences found.")
        return
    
    min_length = min(lengths)
    max_length = max(lengths)
    avg_length = sum(lengths) / len(lengths)
    
    print(f"Dataset size: {len(dataset)} items")
    print(f"Sequence lengths: min={min_length}, max={max_length}, avg={avg_length:.1f}")
    
    # Print an example
    example = next((item for item in dataset if 'tokens' in item), None)
    if example:
        print("\nExample item:")
        print(f"Text: {example.get('text', '')[:50]}...")
        print(f"Tokens: {example.get('tokens', [])[:10]}...")
        print(f"Input IDs: {example.get('input_ids', [])[:10]}...")


def main():
    """Main function."""
    args = parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = TokenizerTrainer.load_tokenizer(args.tokenizer)
    
    # Create encoder
    encoder = Encoder(
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Load dataset
    print(f"Loading dataset from {args.input}")
    dataset = load_dataset(args.input, args.text_field)
    print(f"Loaded {len(dataset)} items")
    
    # Process dataset
    print("Processing dataset...")
    processed_dataset = process_dataset(
        dataset,
        encoder,
        text_field=args.text_field,
        max_length=args.max_length,
        add_special_tokens=args.add_special_tokens,
        truncate=args.truncate
    )
    
    # Print stats
    print_stats(processed_dataset)
    
    # Save processed dataset
    print(f"Saving processed dataset to {args.output}")
    
    # Determine output format from file extension
    output_format = "jsonl" if args.output.endswith(".jsonl") else "json"
    save_dataset(processed_dataset, args.output, format=output_format)
    
    print("Done!")


if __name__ == "__main__":
    main() 