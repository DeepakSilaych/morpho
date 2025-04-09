from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import re
import json
import os
from tqdm import tqdm
from .base import BaseTokenizer

class WordPieceTokenizer(BaseTokenizer):
    """WordPiece tokenizer (used in BERT).
    
    Implementation follows the algorithm described in the BERT paper and used in
    HuggingFace's transformers library for BERT, RoBERTa, etc.
    """
    
    def __init__(self, vocab_size: int = 30000, min_frequency: int = 2, 
                 unk_token: str = "[UNK]", sep_token: str = "[SEP]", 
                 cls_token: str = "[CLS]", pad_token: str = "[PAD]",
                 mask_token: str = "[MASK]", prefix: str = "##"):
        super().__init__()
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.prefix = prefix
        
        # Special tokens go at the beginning of the vocabulary
        self.special_tokens = [pad_token, unk_token, cls_token, sep_token, mask_token]
        
        # Initialize vocabulary with special tokens
        self.vocab = {}
        self.id_to_token = {}
        self.next_id = 0
        
        # Add special tokens to vocabulary
        for token in self.special_tokens:
            self._add_to_vocab(token)
    
    def _add_to_vocab(self, token: str) -> int:
        """Add a token to vocabulary if not present and return its ID."""
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.vocab[token]
    
    def _split_into_wordpieces(self, word: str) -> List[str]:
        """Split a word into WordPiece subwords."""
        if len(word) <= 1:
            return [word] if word in self.vocab else [self.unk_token]
        
        # Try to split the word into subwords
        tokens = []
        start = 0
        while start < len(word):
            # Find the longest subword starting from start
            end = len(word)
            cur_substr = None
            
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = self.prefix + substr
                
                if substr in self.vocab:
                    cur_substr = substr
                    break
                
                end -= 1
            
            # If no subword is found, use UNK
            if cur_substr is None:
                tokens.append(self.unk_token)
                break
            
            tokens.append(cur_substr)
            start = end
        
        return tokens
    
    def _compute_word_frequencies(self, texts: List[str]) -> Dict[str, int]:
        """Compute word frequencies from texts."""
        word_frequencies = Counter()
        for text in texts:
            for word in text.split():
                word_frequencies[word] += 1
        return word_frequencies
    
    def _get_initial_alphabet(self, word_freqs: Dict[str, int]) -> Set[str]:
        """Get initial alphabet from word frequencies."""
        alphabet = set()
        for word in word_freqs:
            for char in word:
                alphabet.add(char)
        return alphabet
    
    def train(self, texts: List[str], verbose: bool = True):
        """Train WordPiece tokenizer on a corpus of texts.
        
        Args:
            texts: List of training texts
            verbose: Whether to show progress with tqdm
        """
        # 1. Compute word frequencies
        word_freqs = self._compute_word_frequencies(texts)
        
        # 2. Filter words with frequency below threshold
        word_freqs = {word: freq for word, freq in word_freqs.items() 
                    if freq >= self.min_frequency}
        
        # 3. Initialize vocabulary with characters
        alphabet = self._get_initial_alphabet(word_freqs)
        for char in sorted(alphabet):
            self._add_to_vocab(char)
            self._add_to_vocab(self.prefix + char)
        
        # 4. Initialize progress bar if verbose
        n_iters = self.vocab_size - len(self.vocab)
        if verbose:
            pbar = tqdm(total=n_iters, desc="Training WordPiece tokenizer")
        
        # 5. Main training loop - iteratively add subwords to vocabulary
        subword_freqs = defaultdict(int)
        current_subwords = {word: list(word) for word in word_freqs}
        
        while len(self.vocab) < self.vocab_size:
            # Calculate frequencies of subword pairs
            subword_freqs.clear()
            
            for word, freq in word_freqs.items():
                subwords = current_subwords[word]
                
                if len(subwords) == 1:
                    continue
                
                for i in range(len(subwords) - 1):
                    pair = (subwords[i], subwords[i + 1])
                    subword_freqs[pair] += freq
            
            if not subword_freqs:
                break
            
            # Find the most frequent pair
            best_pair, best_freq = max(subword_freqs.items(), key=lambda x: x[1])
            best_pair_str = ''.join(best_pair)
            
            # Add the new subword to the vocabulary
            self._add_to_vocab(best_pair_str if best_pair[0] == self.prefix else 
                               self.prefix + best_pair_str)
            
            # Update current_subwords
            for word in word_freqs:
                subwords = current_subwords[word]
                i = 0
                
                while i < len(subwords) - 1:
                    if i + 1 < len(subwords) and subwords[i] == best_pair[0] and subwords[i + 1] == best_pair[1]:
                        subwords[i] = best_pair_str
                        subwords.pop(i + 1)
                    else:
                        i += 1
                
                current_subwords[word] = subwords
            
            if verbose:
                pbar.update(1)
            
            if len(self.vocab) >= self.vocab_size:
                break
        
        if verbose:
            pbar.close()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using WordPiece algorithm."""
        if not self.vocab:
            raise ValueError("Tokenizer vocabulary is empty. Train the tokenizer first.")
        
        tokens = []
        for word in text.split():
            word_tokens = self._split_into_wordpieces(word)
            tokens.extend(word_tokens)
        
        return tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """Join tokens back into text, handling the prefix."""
        text = ""
        for token in tokens:
            if token.startswith(self.prefix):
                text += token[len(self.prefix):]
            else:
                if text and not (token in self.special_tokens or text.endswith(" ")):
                    text += " "
                text += token
        
        return text
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs using WordPiece tokenization."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.id_to_token.get(id, self.unk_token) for id in ids]
        return self.detokenize(tokens)
    
    def save(self, path: str):
        """Save tokenizer vocabulary to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'special_tokens': self.special_tokens,
                'prefix': self.prefix
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'WordPieceTokenizer':
        """Load tokenizer from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tokenizer = cls(prefix=data.get('prefix', '##'))
        tokenizer.vocab = {k: int(v) for k, v in data['vocab'].items()}
        tokenizer.special_tokens = data.get('special_tokens', tokenizer.special_tokens)
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.vocab.items()}
        tokenizer.next_id = max(tokenizer.vocab.values()) + 1 if tokenizer.vocab else 0
        
        return tokenizer 