#!/usr/bin/env python
"""
Example script for training a BPE tokenizer with custom regex pre-tokenization.
This example uses the same regex pattern as GPT-2 tokenizer.
"""

import sys
import os
import argparse
import re
from typing import List

# Add parent directory to path to allow importing from the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizers.bpe import BPETokenizer
from preprocessing.normalizer import TextNormalizer
from utils.visualize_merges import plot_merge_graph, visualize_token_segmentation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer with regex pre-tokenization on a text file."
    )
    parser.add_argument(
        "--input", 
        "-i", 
        type=str, 
        required=True, 
        help="Path to input text file(s)",
        nargs='+')
    parser.add_argument(
        "--vocab-size", 
        "-v", 
        type=int, 
        default=10000, 
        help="Vocabulary size")
    parser.add_argument(
        "--min-freq", 
        "-f", 
        type=int, 
        default=2, 
        help="Minimum token frequency")
    parser.add_argument(
        "--output", 
        "-o", 
        type=str, 
        required=True, 
        help="Path to output model file")
    parser.add_argument(
        "--max-lines", 
        "-m", 
        type=int, 
        default=None, 
        help="Maximum number of lines to read from each file")
    parser.add_argument(
        "--test-text", 
        "-t", 
        type=str, 
        default=None, 
        help="Test text to tokenize after training")
    parser.add_argument(
        "--visualize", 
        action="store_true", 
        help="Visualize the BPE merges and token segmentation")
    return parser.parse_args()


def normalize_corpus(files: List[str], max_lines: int = None) -> List[str]:
    """Load and normalize a text corpus."""
    normalizer = TextNormalizer(
        unicode_normalization='NFC',
        replace_control_chars=True
    )
    
    corpus = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            if max_lines:
                lines = [next(f).strip() for _ in range(max_lines) if f]
            else:
                lines = [line.strip() for line in f]
            corpus.extend(normalizer.batch_normalize(lines))
    
    return corpus


def main():
    """Main function."""
    args = parse_args()
    
    print(f"Processing input files: {args.input}")
    print(f"Training BPE tokenizer with vocab size {args.vocab_size}")
    
    # Load and normalize corpus
    corpus = normalize_corpus(args.input, args.max_lines)
    print(f"Loaded {len(corpus)} lines of text")
    
    # Create GPT-2 style regex pattern for pre-tokenization
    # This pattern handles contractions, words, numbers, and other symbols
    gpt2_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    # Create BPE tokenizer with the regex pattern
    tokenizer = BPETokenizer(vocab_size=args.vocab_size, pre_tokenize_pattern=gpt2_pattern)
    
    # Train tokenizer
    print("Training tokenizer...")
    tokenizer.train(corpus, min_frequency=args.min_freq)
    
    # Save tokenizer
    print(f"Saving tokenizer to {args.output}")
    tokenizer.save(args.output)
    
    # Test tokenizer if test text is provided
    if args.test_text:
        print("\nTesting tokenizer on:", args.test_text)
        tokens = tokenizer.tokenize(args.test_text)
        print("Tokens:", tokens)
        
        ids = tokenizer.encode(args.test_text)
        print("Token IDs:", ids)
        
        decoded = tokenizer.decode(ids)
        print("Decoded:", decoded)
        
        # Visualize if requested
        if args.visualize:
            try:
                # Visualize merge operations
                plot_merge_graph(tokenizer.merges, max_nodes=30, 
                                title="BPE Merge Operations")
                
                # Visualize token segmentation
                visualize_token_segmentation(args.test_text, tokens,
                                           title=f"GPT-2 Style Tokenization: {args.test_text}")
            except ImportError:
                print("Visualization requires matplotlib and networkx. Install with:")
                print("pip install matplotlib networkx")


if __name__ == "__main__":
    main() 