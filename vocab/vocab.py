from typing import Dict, List, Optional, Set, Union, Iterable
import json
import os
from collections import Counter

class Vocabulary:
    """Vocabulary class for mapping between tokens and ids.
    
    This class handles the mapping between tokens and their corresponding ids,
    and provides methods for vocabulary building, special token handling,
    and serialization.
    """
    
    def __init__(self,
                 unk_token: str = "<unk>",
                 pad_token: str = "<pad>",
                 bos_token: str = "<s>",
                 eos_token: str = "</s>",
                 mask_token: Optional[str] = None,
                 special_tokens: Optional[List[str]] = None):
        """Initialize a vocabulary.
        
        Args:
            unk_token: Token to use for unknown tokens
            pad_token: Token to use for padding
            bos_token: Token to use for beginning of sequence
            eos_token: Token to use for end of sequence
            mask_token: Token to use for masked tokens (e.g., in BERT)
            special_tokens: Additional special tokens
        """
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        
        # Initialize token-to-id and id-to-token mappings
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Add default special tokens
        self.default_special_tokens = [pad_token, unk_token, bos_token, eos_token]
        if mask_token:
            self.default_special_tokens.append(mask_token)
        
        # Add user-provided special tokens
        self.additional_special_tokens = special_tokens or []
        
        # Add all special tokens to vocabulary
        self.next_id = 0
        for token in self.default_special_tokens + self.additional_special_tokens:
            if token and token not in self.token_to_id:
                self.add_token(token)
    
    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary and return its ID.
        
        Args:
            token: The token to add
            
        Returns:
            The ID of the token
        """
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.token_to_id[token]
    
    def add_tokens(self, tokens: Iterable[str]) -> List[int]:
        """Add multiple tokens to the vocabulary.
        
        Args:
            tokens: An iterable of tokens to add
            
        Returns:
            A list of token IDs
        """
        return [self.add_token(token) for token in tokens]
    
    def token_to_index(self, token: str) -> int:
        """Convert a token to its ID.
        
        Args:
            token: The token to convert
            
        Returns:
            The ID of the token, or the unknown token ID if not found
        """
        return self.token_to_id.get(token, self.token_to_id.get(self.unk_token, 0))
    
    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """Convert a list of tokens to their IDs.
        
        Args:
            tokens: The tokens to convert
            
        Returns:
            A list of token IDs
        """
        return [self.token_to_index(token) for token in tokens]
    
    def index_to_token(self, index: int) -> str:
        """Convert an ID to its token.
        
        Args:
            index: The ID to convert
            
        Returns:
            The token for the ID, or the unknown token if not found
        """
        return self.id_to_token.get(index, self.unk_token)
    
    def indices_to_tokens(self, indices: List[int]) -> List[str]:
        """Convert a list of IDs to their tokens.
        
        Args:
            indices: The IDs to convert
            
        Returns:
            A list of tokens
        """
        return [self.index_to_token(index) for index in indices]
    
    def build_from_counter(self, counter: Counter, min_freq: int = 1, max_size: Optional[int] = None):
        """Build vocabulary from a counter of tokens.
        
        Args:
            counter: A counter of tokens
            min_freq: Minimum frequency for a token to be included
            max_size: Maximum vocabulary size (not including special tokens)
        """
        # Sort tokens by frequency, then alphabetically for ties
        tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        
        # Filter by minimum frequency
        tokens = [(token, freq) for token, freq in tokens if freq >= min_freq]
        
        # Limit vocabulary size if specified
        if max_size is not None:
            tokens = tokens[:max_size]
        
        # Add tokens to vocabulary
        for token, _ in tokens:
            if token not in self.token_to_id:
                self.add_token(token)
    
    def build_from_texts(self, texts: List[str], tokenize_fn=None, min_freq: int = 1, max_size: Optional[int] = None):
        """Build vocabulary from texts.
        
        Args:
            texts: A list of text strings
            tokenize_fn: A function to tokenize each text (default: split on whitespace)
            min_freq: Minimum frequency for a token to be included
            max_size: Maximum vocabulary size (not including special tokens)
        """
        # Default tokenization function
        if tokenize_fn is None:
            tokenize_fn = lambda text: text.split()
        
        # Count token frequencies
        counter = Counter()
        for text in texts:
            tokens = tokenize_fn(text)
            counter.update(tokens)
        
        # Build vocabulary from counter
        self.build_from_counter(counter, min_freq, max_size)
    
    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.token_to_id)
    
    def __contains__(self, token: str) -> bool:
        """Check if a token is in the vocabulary."""
        return token in self.token_to_id
    
    def get_special_tokens_mask(self, token_ids: List[int]) -> List[int]:
        """Create a mask of 1s for special tokens and 0s for regular tokens.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            A list of 1s and 0s (1 for special tokens)
        """
        special_token_ids = {self.token_to_id[token] for token in 
                           self.default_special_tokens + self.additional_special_tokens
                           if token in self.token_to_id}
        
        return [1 if token_id in special_token_ids else 0 for token_id in token_ids]
    
    def save(self, path: str):
        """Save vocabulary to a file.
        
        Args:
            path: Path to save the vocabulary
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'token_to_id': self.token_to_id,
            'special_tokens': {
                'unk_token': self.unk_token,
                'pad_token': self.pad_token,
                'bos_token': self.bos_token,
                'eos_token': self.eos_token,
                'mask_token': self.mask_token,
                'additional_special_tokens': self.additional_special_tokens
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from a file.
        
        Args:
            path: Path to load the vocabulary from
            
        Returns:
            A new Vocabulary instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get special tokens or use defaults
        special_tokens = data.get('special_tokens', {})
        unk_token = special_tokens.get('unk_token', '<unk>')
        pad_token = special_tokens.get('pad_token', '<pad>')
        bos_token = special_tokens.get('bos_token', '<s>')
        eos_token = special_tokens.get('eos_token', '</s>')
        mask_token = special_tokens.get('mask_token', None)
        additional_special_tokens = special_tokens.get('additional_special_tokens', [])
        
        # Create a new vocabulary
        vocab = cls(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            special_tokens=additional_special_tokens
        )
        
        # Clear initial vocabulary (which has special tokens) and load the saved one
        vocab.token_to_id = {}
        vocab.id_to_token = {}
        vocab.next_id = 0
        
        # Load token to id mapping
        for token, idx in data['token_to_id'].items():
            vocab.token_to_id[token] = int(idx)
            vocab.id_to_token[int(idx)] = token
        
        # Update next_id
        if vocab.token_to_id:
            vocab.next_id = max(vocab.token_to_id.values()) + 1
        
        return vocab 