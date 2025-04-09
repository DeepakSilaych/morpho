from typing import List, Dict, Tuple, Optional, Counter as CounterType, Pattern
from collections import defaultdict, Counter
import re
import json
import os
from tqdm import tqdm
from .base import BaseTokenizer

class BPETokenizer(BaseTokenizer):
    """Byte-Pair Encoding (BPE) tokenizer.
    
    Implementation follows OpenAI's GPT-2/GPT-3 BPE algorithm with slight modifications:
    1. Character-level initial vocabulary
    2. Frequency-based merge operations
    3. Special token handling
    4. Optional regex pre-tokenization pattern
    """
    
    def __init__(self, vocab_size: int = 30000, pre_tokenize_pattern: Optional[Pattern] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> id mapping
        self.id_to_token = {}  # id -> token mapping
        self.merges = {}  # (token1, token2) -> merged_token
        self.next_id = 0
        # Special tokens
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.special_tokens = [self.unk_token, self.pad_token, self.bos_token, self.eos_token]
        
        # Pre-tokenization pattern
        self.pre_tokenize_pattern = pre_tokenize_pattern
        
        # Initialize special tokens
        for token in self.special_tokens:
            self._add_to_vocab(token)
    
    def _add_to_vocab(self, token: str) -> int:
        """Add a token to vocabulary if not present and return its ID."""
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.vocab[token]
    
    def _get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> CounterType:
        """Count frequency of consecutive token pairs in all words."""
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs
    
    def _merge_pair(self, pair: Tuple[str, str], word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of a given pair in the vocabulary."""
        new_word_freqs = {}
        bigram = pair[0] + pair[1]
        self.merges[pair] = bigram
        
        for word, freq in word_freqs.items():
            word_list = list(word)
            i = 0
            
            while i < len(word_list) - 1:
                if word_list[i] == pair[0] and word_list[i+1] == pair[1]:
                    word_list[i] = bigram
                    word_list.pop(i+1)
                else:
                    i += 1
            
            new_word = tuple(word_list)
            new_word_freqs[new_word] = new_word_freqs.get(new_word, 0) + freq
        
        return new_word_freqs
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """Apply pre-tokenization to text if a pattern is provided.
        
        Args:
            text: Input text
            
        Returns:
            List of pre-tokenized segments
        """
        if self.pre_tokenize_pattern is None:
            # Default: split on whitespace
            return text.split()
        else:
            # Apply regex pattern
            return [match for match in self.pre_tokenize_pattern.findall(text) if match]
    
    def train(self, texts: List[str], min_frequency: int = 2, verbose: bool = True):
        """Train BPE tokenizer on a corpus of texts.
        
        Args:
            texts: List of training texts
            min_frequency: Minimum frequency for a merge rule to be considered
            verbose: Whether to show progress bar
        """
        # 1. Initialize character-level vocabulary
        word_freqs = defaultdict(int)
        
        # Pre-tokenize and initialize with character sequences
        for text in texts:
            pre_tokens = self._pre_tokenize(text)
            for token in pre_tokens:
                # Convert token to character tuples
                char_tuple = tuple(token)
                word_freqs[char_tuple] += 1
        
        # Remove low frequency words
        word_freqs = {word: freq for word, freq in word_freqs.items() if freq >= min_frequency}
        
        # Create initial vocabulary
        for word in word_freqs:
            for char in word:
                self._add_to_vocab(char)
        
        # Initialize progress bar if verbose
        num_merges = self.vocab_size - len(self.vocab)
        
        if verbose:
            pbar = tqdm(total=num_merges, desc="Training BPE tokenizer")
        
        # Iteratively merge most frequent pairs
        while len(self.vocab) < self.vocab_size:
            # Get pair frequencies
            pair_freqs = self._get_stats(word_freqs)
            
            if not pair_freqs:
                break
                
            # Find the most frequent pair
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            
            # Merge the best pair in all words
            word_freqs = self._merge_pair(best_pair, word_freqs)
            
            # Add the new token to the vocabulary
            merged_token = ''.join(best_pair)
            self._add_to_vocab(merged_token)
            
            if verbose:
                pbar.update(1)
                
            # Break if vocab size is reached
            if len(self.vocab) >= self.vocab_size:
                break
        
        if verbose:
            pbar.close()
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using learned BPE merge rules."""
        if not word:
            return []
            
        # Start with character-level tokens
        tokens = list(word)
        
        # Apply merge rules until no more merges can be done
        while len(tokens) > 1:
            # Find all consecutive pairs
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
            
            # Find the first merge rule that can be applied
            for pair in pairs:
                if pair in self.merges:
                    merged_token = self.merges[pair]
                    # Find the position to apply the merge
                    i = 0
                    while i < len(tokens) - 1:
                        if tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                            tokens[i] = merged_token
                            tokens.pop(i+1)
                        else:
                            i += 1
                    # Restart from the beginning
                    break
            else:
                # No merge rule could be applied
                break
        
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using learned BPE merge rules."""
        # Pre-tokenize text
        pre_tokens = self._pre_tokenize(text)
        tokens = []
        
        for i, token in enumerate(pre_tokens):
            word_tokens = self._tokenize_word(token)
            tokens.extend(word_tokens)
            # Add space token between words (except the last word)
            if i < len(pre_tokens) - 1 and token != ' ' and pre_tokens[i+1] != ' ':
                tokens.append(" ")
        
        return tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """Join tokens back into text."""
        return "".join(tokens)
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs using BPE tokenization."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.id_to_token.get(id, self.unk_token) for id in ids]
        return self.detokenize(tokens)
    
    def save(self, path: str):
        """Save tokenizer vocabulary and merge rules to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'merges': {' '.join(k): v for k, v in self.merges.items()},
                'special_tokens': self.special_tokens,
                'pre_tokenize_pattern': self.pre_tokenize_pattern.pattern if self.pre_tokenize_pattern else None
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            pre_tokenize_pattern = data.get('pre_tokenize_pattern')
            
            tokenizer = cls(
                pre_tokenize_pattern=re.compile(pre_tokenize_pattern) if pre_tokenize_pattern else None
            )
            
            tokenizer.vocab = {k: int(v) for k, v in data['vocab'].items()}
            tokenizer.merges = {tuple(k.split(' ')): v for k, v in data['merges'].items()}
            tokenizer.special_tokens = data['special_tokens']
            tokenizer.id_to_token = {int(v): k for k, v in tokenizer.vocab.items()}
            tokenizer.next_id = max(tokenizer.vocab.values()) + 1 if tokenizer.vocab else 0
        return tokenizer
        
    @classmethod
    def from_pretrained(cls, path: str, pre_tokenize_pattern: Optional[Pattern] = None) -> 'BPETokenizer':
        """Load tokenizer from disk with an option to override the pre-tokenize pattern."""
        tokenizer = cls.load(path)
        if pre_tokenize_pattern is not None:
            tokenizer.pre_tokenize_pattern = pre_tokenize_pattern
        return tokenizer 