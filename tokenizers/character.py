from typing import List, Dict, Optional, Set
import re
import json
import os
import unicodedata
from .base import BaseTokenizer

class CharacterTokenizer(BaseTokenizer):
    """A character-level tokenizer with unicode normalization and special token support.
    
    Features:
    - Unicode normalization (NFKC by default)
    - Special token handling
    - Optional character filtering
    - Handles control characters
    """
    
    def __init__(self, 
                 unk_token: str = "<unk>",
                 pad_token: str = "<pad>", 
                 bos_token: str = "<s>", 
                 eos_token: str = "</s>",
                 control_token: str = "<control>",
                 normalize_unicode: bool = True,
                 normalization_form: str = 'NFKC',
                 filter_chars: Optional[Set[str]] = None):
        super().__init__()
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.control_token = control_token
        self.normalize_unicode = normalize_unicode
        self.normalization_form = normalization_form
        self.filter_chars = filter_chars or set()
        
        # Define special tokens
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token, control_token]
        
        # Initialize vocabulary with special tokens
        self.vocab = {}  # char -> id mapping
        self.id_to_char = {}  # id -> char mapping
        self.next_id = 0
        
        # Add special tokens to vocabulary
        for token in self.special_tokens:
            self._add_to_vocab(token)
    
    def _add_to_vocab(self, char: str) -> int:
        """Add a character to vocabulary if not present and return its ID."""
        if char not in self.vocab:
            self.vocab[char] = self.next_id
            self.id_to_char[self.next_id] = char
            self.next_id += 1
        return self.vocab[char]
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text before tokenization."""
        # Apply unicode normalization if enabled
        if self.normalize_unicode:
            text = unicodedata.normalize(self.normalization_form, text)
        return text
    
    def _is_control(self, char: str) -> bool:
        """Check if a character is a control character."""
        # Check if character is a control character (e.g. newline, tab)
        cat = unicodedata.category(char)
        return cat.startswith('C')
    
    def tokenize(self, text: str) -> List[str]:
        """Split text into individual characters."""
        text = self._preprocess(text)
        tokens = []
        
        for char in text:
            # Handle control characters
            if self._is_control(char):
                tokens.append(self.control_token)
            # Filter out unwanted characters
            elif char in self.filter_chars:
                continue
            else:
                tokens.append(char)
                
        return tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """Join characters back into text, handling special tokens."""
        # Filter out special tokens for detokenization
        chars = []
        for token in tokens:
            if token in self.special_tokens:
                # Handle control token - convert to space
                if token == self.control_token:
                    chars.append(' ')
            else:
                chars.append(token)
                
        return "".join(chars)
    
    def encode(self, text: str) -> List[int]:
        """Convert text to character IDs."""
        chars = self.tokenize(text)
        return [self.vocab.get(char, self.vocab[self.unk_token]) for char in chars]
    
    def decode(self, ids: List[int]) -> str:
        """Convert character IDs back to text."""
        chars = [self.id_to_char.get(id, self.unk_token) for id in ids]
        return self.detokenize(chars)
    
    def build_vocab_from_texts(self, texts: List[str]):
        """Build vocabulary from a list of texts.
        
        For character tokenizers, this adds all unique characters to the vocabulary.
        
        Args:
            texts: List of text strings
        """
        for text in texts:
            # Process each text to handle control characters and filtering
            for token in self.tokenize(text):
                if token not in self.vocab:
                    self._add_to_vocab(token)
    
    def save(self, path: str):
        """Save tokenizer vocabulary to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'special_tokens': self.special_tokens,
                'normalize_unicode': self.normalize_unicode,
                'normalization_form': self.normalization_form,
                'filter_chars': list(self.filter_chars)
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'CharacterTokenizer':
        """Load tokenizer from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract configuration
        normalize_unicode = data.get('normalize_unicode', True)
        normalization_form = data.get('normalization_form', 'NFKC')
        filter_chars = set(data.get('filter_chars', []))
        special_tokens = data.get('special_tokens', [])
        
        # Get special token values or use defaults
        unk_token = special_tokens[1] if len(special_tokens) > 1 else "<unk>"
        pad_token = special_tokens[0] if len(special_tokens) > 0 else "<pad>"
        bos_token = special_tokens[2] if len(special_tokens) > 2 else "<s>"
        eos_token = special_tokens[3] if len(special_tokens) > 3 else "</s>"
        control_token = special_tokens[4] if len(special_tokens) > 4 else "<control>"
        
        # Create tokenizer with saved settings
        tokenizer = cls(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            control_token=control_token,
            normalize_unicode=normalize_unicode,
            normalization_form=normalization_form,
            filter_chars=filter_chars
        )
        
        # Load vocabulary
        tokenizer.vocab = {k: int(v) for k, v in data['vocab'].items()}
        tokenizer.id_to_char = {int(v): k for k, v in tokenizer.vocab.items()}
        tokenizer.next_id = max(tokenizer.vocab.values()) + 1 if tokenizer.vocab else 0
        
        return tokenizer 