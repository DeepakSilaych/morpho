#!/usr/bin/env python
"""
Command-line interface for the morpho tokenizer library.

Usage:
    morpho train --type bpe --file data.txt --vocab-size 8000
    morpho encode --tokenizer model.json --text "hello world"
    morpho decode --tokenizer model.json --ids 12 15 18
"""

import sys
import os
import argparse
import re
from typing import List, Dict, Optional, Any

# Add parent directory to path to allow importing from the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vocab.trainer import TokenizerTrainer
from encoders.encoder import Encoder
from preprocessing.normalizer import TextNormalizer
from preprocessing.tokenizer_helpers import basic_clean


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Morpho Tokenization Library CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a tokenizer")
    train_parser.add_argument(
        "--type", 
        "-t", 
        type=str, 
        required=True, 
        choices=["whitespace", "character", "bpe", "wordpiece", "unigram", "sentencepiece"],
        help="Type of tokenizer to train")
    train_parser.add_argument(
        "--file", 
        "-f", 
        type=str, 
        required=True,
        nargs='+',
        help="Path to training file(s)")
    train_parser.add_argument(
        "--output", 
        "-o", 
        type=str, 
        required=True, 
        help="Path to save the trained tokenizer")
    train_parser.add_argument(
        "--vocab-size", 
        "-v", 
        type=int, 
        default=30000, 
        help="Vocabulary size")
    train_parser.add_argument(
        "--min-frequency", 
        "-m", 
        type=int, 
        default=2, 
        help="Minimum token frequency")
    train_parser.add_argument(
        "--max-lines", 
        type=int, 
        default=None, 
        help="Maximum number of lines to read from each file")
    train_parser.add_argument(
        "--pre-tokenize-pattern", 
        type=str, 
        default=None, 
        help="Regex pattern for pre-tokenization (BPE only)")
    train_parser.add_argument(
        "--use-gpt2-pattern", 
        action="store_true", 
        help="Use GPT-2 pre-tokenization pattern (BPE only)")
    
    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode text to token IDs")
    encode_parser.add_argument(
        "--tokenizer", 
        "-t", 
        type=str, 
        required=True, 
        help="Path to tokenizer model file")
    encode_parser.add_argument(
        "--text", 
        "-i", 
        type=str, 
        required=True, 
        help="Text to encode")
    encode_parser.add_argument(
        "--special-tokens", 
        "-s", 
        action="store_true", 
        help="Add special tokens (BOS/EOS)")
    encode_parser.add_argument(
        "--output-format", 
        "-f", 
        type=str, 
        default="ids", 
        choices=["ids", "tokens", "both"], 
        help="Output format")
    
    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode token IDs to text")
    decode_parser.add_argument(
        "--tokenizer", 
        "-t", 
        type=str, 
        required=True, 
        help="Path to tokenizer model file")
    decode_parser.add_argument(
        "--ids", 
        "-i", 
        type=int, 
        nargs='+', 
        required=True, 
        help="Token IDs to decode")
    decode_parser.add_argument(
        "--skip-special", 
        "-s", 
        action="store_true", 
        help="Skip special tokens")
    
    # Tokenize command
    tokenize_parser = subparsers.add_parser("tokenize", help="Tokenize text without encoding")
    tokenize_parser.add_argument(
        "--tokenizer", 
        "-t", 
        type=str, 
        required=True, 
        help="Path to tokenizer model file")
    tokenize_parser.add_argument(
        "--text", 
        "-i", 
        type=str, 
        required=True, 
        help="Text to tokenize")
    tokenize_parser.add_argument(
        "--special-tokens", 
        "-s", 
        action="store_true", 
        help="Add special tokens (BOS/EOS)")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get information about a trained tokenizer")
    info_parser.add_argument(
        "--tokenizer", 
        "-t", 
        type=str, 
        required=True, 
        help="Path to tokenizer model file")
    
    return parser.parse_args()


def train_tokenizer(args):
    """Train a tokenizer."""
    print(f"Training {args.type} tokenizer on files: {args.file}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Minimum token frequency: {args.min_frequency}")
    
    # Set pre-tokenization pattern
    pre_tokenize_pattern = None
    if args.type == 'bpe':
        if args.use_gpt2_pattern:
            print("Using GPT-2 pre-tokenization pattern")
            pre_tokenize_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        elif args.pre_tokenize_pattern:
            print(f"Using custom pre-tokenization pattern: {args.pre_tokenize_pattern}")
            pre_tokenize_pattern = args.pre_tokenize_pattern
    
    # Create trainer
    trainer = TokenizerTrainer(
        tokenizer_type=args.type,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        pre_tokenize_pattern=pre_tokenize_pattern
    )
    
    # Train on files
    tokenizer = trainer.train_from_files(
        files=args.file,
        max_lines=args.max_lines,
        verbose=True
    )
    
    # Save tokenizer
    print(f"Saving tokenizer to {args.output}")
    trainer.save_tokenizer(args.output)
    
    # Print statistics
    vocab_size = len(getattr(tokenizer, 'vocab', {}))
    print(f"Trained tokenizer with vocabulary size: {vocab_size}")
    
    # Print an example
    example_text = "Hello, world! This is an example text to tokenize."
    print("\nExample:")
    print(f"Text: {example_text}")
    
    tokens = tokenizer.tokenize(example_text)
    print(f"Tokens: {tokens}")
    
    ids = tokenizer.encode(example_text)
    print(f"Token IDs: {ids}")


def encode_text(args):
    """Encode text to token IDs."""
    # Load tokenizer
    tokenizer = TokenizerTrainer.load_tokenizer(args.tokenizer)
    
    # Clean text
    text = basic_clean(args.text)
    
    # Create encoder
    encoder = Encoder(tokenizer=tokenizer)
    
    # Tokenize and encode
    tokens = tokenizer.tokenize(text)
    ids = encoder.encode(text, add_special_tokens=args.special_tokens)
    
    # Print output based on format
    if args.output_format == "ids":
        print(" ".join(map(str, ids)))
    elif args.output_format == "tokens":
        print(" ".join(tokens))
    else:  # both
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {ids}")


def decode_ids(args):
    """Decode token IDs to text."""
    # Load tokenizer
    tokenizer = TokenizerTrainer.load_tokenizer(args.tokenizer)
    
    # Create encoder
    encoder = Encoder(tokenizer=tokenizer)
    
    # Decode
    text = encoder.decode(args.ids, skip_special_tokens=args.skip_special)
    
    # Print output
    print(text)


def tokenize_text(args):
    """Tokenize text without encoding."""
    # Load tokenizer
    tokenizer = TokenizerTrainer.load_tokenizer(args.tokenizer)
    
    # Clean text
    text = basic_clean(args.text)
    
    # Tokenize
    tokens = tokenizer.tokenize(text)
    
    # Add special tokens if requested
    if args.special_tokens:
        from vocab.special_tokens import SpecialTokens
        tokens = SpecialTokens.add_special_tokens(tokens, bos=True, eos=True)
    
    # Print output
    print(" ".join(tokens))


def show_tokenizer_info(args):
    """Show information about a trained tokenizer."""
    # Load tokenizer
    try:
        tokenizer = TokenizerTrainer.load_tokenizer(args.tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Load metadata
    metadata_path = f"{os.path.splitext(args.tokenizer)[0]}_metadata.json"
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Print information
    print(f"Tokenizer path: {args.tokenizer}")
    print(f"Tokenizer type: {metadata.get('tokenizer_type', 'Unknown')}")
    
    vocab = getattr(tokenizer, 'vocab', {})
    print(f"Vocabulary size: {len(vocab)}")
    
    # Print special tokens
    special_tokens = metadata.get('special_tokens', {})
    if special_tokens:
        print("\nSpecial tokens:")
        for name, token in special_tokens.items():
            token_id = vocab.get(token, 'N/A')
            print(f"  {name}: {token} (ID: {token_id})")
    
    # Print example tokens
    if vocab:
        print("\nExample vocabulary items:")
        items = list(vocab.items())
        for token, id in sorted(items[:10], key=lambda x: x[1]):
            print(f"  {token}: {id}")
        print(f"  ... and {len(vocab) - 10} more tokens")


def main():
    """Main function."""
    args = parse_args()
    
    if args.command == "train":
        train_tokenizer(args)
    elif args.command == "encode":
        encode_text(args)
    elif args.command == "decode":
        decode_ids(args)
    elif args.command == "tokenize":
        tokenize_text(args)
    elif args.command == "info":
        show_tokenizer_info(args)
    else:
        print("Please specify a command. Use --help for more information.")
        sys.exit(1)


if __name__ == "__main__":
    main() 