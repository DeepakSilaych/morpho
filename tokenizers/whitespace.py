from typing import List, Dict, Optional
import re
import json
import os
from .base import BaseTokenizer

class WhitespaceTokenizer(BaseTokenizer):
    """A simple whitespace-based tokenizer with special token handling.
    
    Implements a basic but robust whitespace tokenizer with support for:
    - Special token handling (pad, unk, etc.)
    - Saving/loading vocabulary
    - Handling unknown tokens
    """
    
    def __init__(self, 
                 unk_token: str = "<unk>",
                 pad_token: str = "<pad>", 
                 bos_token: str = "<s>", 
                 eos_token: str = "</s>",
                 lowercase: bool = False):
        super().__init__()
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.lowercase = lowercase
        
        # Define special tokens
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]
        
        # Initialize vocabulary
        self.vocab = {}  # token -> id mapping
        self.id_to_token = {}  # id -> token mapping
        self.next_id = 0
        
        # Add special tokens to vocabulary
        for token in self.special_tokens:
            self._add_to_vocab(token)
            
        # Custom whitespace pattern for tokenization
        self.pattern = re.compile(r'\s+')
    
    def _add_to_vocab(self, token: str) -> int:
        """Add a token to vocabulary if not present and return its ID."""
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.vocab[token]
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text before tokenization."""
        if self.lowercase:
            text = text.lower()
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Split text on whitespace."""
        text = self._preprocess(text)
        return self.pattern.split(text.strip())
    
    def detokenize(self, tokens: List[str]) -> str:
        """Join tokens with spaces, handling special tokens properly."""
        # Filter out special tokens for detokenization
        filtered_tokens = [t for t in tokens if t not in self.special_tokens]
        return " ".join(filtered_tokens)
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs, handling unknown tokens."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.id_to_token.get(id, self.unk_token) for id in ids]
        return self.detokenize(tokens)
    
    def build_vocab_from_texts(self, texts: List[str], min_freq: int = 1):
        """Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for a token to be included
        """
        # Count token frequencies
        token_counts = {}
        for text in texts:
            for token in self.tokenize(text):
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Filter by minimum frequency
        for token, count in token_counts.items():
            if count >= min_freq and token not in self.vocab:
                self._add_to_vocab(token)
    
    def save(self, path: str):
        """Save tokenizer vocabulary to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'special_tokens': self.special_tokens,
                'lowercase': self.lowercase
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'WhitespaceTokenizer':
        """Load tokenizer from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        lowercase = data.get('lowercase', False)
        special_tokens = data.get('special_tokens', [])
        
        # Get special token values or use defaults
        unk_token = special_tokens[1] if len(special_tokens) > 1 else "<unk>"
        pad_token = special_tokens[0] if len(special_tokens) > 0 else "<pad>"
        bos_token = special_tokens[2] if len(special_tokens) > 2 else "<s>"
        eos_token = special_tokens[3] if len(special_tokens) > 3 else "</s>"
        
        # Create tokenizer with saved settings
        tokenizer = cls(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            lowercase=lowercase
        )
        
        # Load vocabulary, skipping special tokens (already added in __init__)
        tokenizer.vocab = {k: int(v) for k, v in data['vocab'].items()}
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.vocab.items()}
        tokenizer.next_id = max(tokenizer.vocab.values()) + 1 if tokenizer.vocab else 0
        
        return tokenizer 