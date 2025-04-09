from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import math
import json
import os
import re
import numpy as np
from tqdm import tqdm
from .base import BaseTokenizer

class UnigramTokenizer(BaseTokenizer):
    """Unigram Language Model tokenizer.
    
    Implementation follows the algorithm described in the SentencePiece paper:
    "Subword Regularization: Improving Neural Network Translation Models with 
    Multiple Subword Candidates" (Kudo, 2018)
    """
    
    def __init__(self, vocab_size: int = 8000, 
                 unk_token: str = "<unk>", 
                 bos_token: str = "<s>", 
                 eos_token: str = "</s>", 
                 pad_token: str = "<pad>"):
        super().__init__()
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]
        
        # Initialize vocabulary with special tokens
        self.vocab = {}
        self.id_to_token = {}
        self.token_scores = {}  # token -> log probability
        self.next_id = 0
        
        # Add special tokens to vocabulary
        for token in self.special_tokens:
            self._add_to_vocab(token, 0.0)  # Special tokens have score 0.0
    
    def _add_to_vocab(self, token: str, score: float = float("-inf")) -> int:
        """Add a token to vocabulary if not present and return its ID."""
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.token_scores[token] = score
            self.next_id += 1
        return self.vocab[token]
    
    def _compute_token_frequencies(self, texts: List[str]) -> Dict[str, int]:
        """Compute character/substring frequencies from texts."""
        char_freq = Counter()
        for text in texts:
            for char in text:
                char_freq[char] += 1
        return char_freq
    
    def _calculate_loss(self, word: str, model: Dict[str, float]) -> float:
        """Calculate negative log likelihood of a segmentation."""
        segmentation = self._segment_word(word, model)
        return -sum(model.get(token, float("-inf")) for token in segmentation)
    
    def _segment_word(self, word: str, model: Dict[str, float]) -> List[str]:
        """Find the best segmentation using the Viterbi algorithm."""
        n = len(word)
        
        # Best score (log probability) for segmentation up to position i
        best_scores = [float("-inf")] * (n + 1)
        best_scores[0] = 0.0
        
        # Best token ending at position i
        best_tokens = [None] * (n + 1)
        
        # Dynamic programming to find the best segmentation
        for i in range(n):
            if best_scores[i] == float("-inf"):
                continue
                
            for j in range(i + 1, min(n + 1, i + 20)):  # Limit max token length to 20
                token = word[i:j]
                score = model.get(token, float("-inf"))
                
                if score == float("-inf"):
                    continue
                
                new_score = best_scores[i] + score
                if new_score > best_scores[j]:
                    best_scores[j] = new_score
                    best_tokens[j] = token
        
        # If no valid segmentation found, return unknown token
        if best_scores[n] == float("-inf"):
            return [self.unk_token]
            
        # Reconstruct the best segmentation
        tokens = []
        pos = n
        while pos > 0:
            token = best_tokens[pos]
            tokens.append(token)
            pos -= len(token)
            
        return tokens[::-1]  # Reverse tokens to get correct order
    
    def _initialize_seed_vocab(self, word_freqs: Dict[str, int]) -> Dict[str, float]:
        """Initialize seed vocabulary for Unigram model."""
        # Start with character vocabulary
        chars = set()
        for word in word_freqs:
            for char in word:
                chars.add(char)
        
        # Add all substrings of length <= 3 as candidates
        vocab = {}
        for word, freq in word_freqs.items():
            for i in range(len(word)):
                for j in range(i + 1, min(i + 4, len(word) + 1)):
                    substr = word[i:j]
                    vocab[substr] = vocab.get(substr, 0) + freq
        
        # Calculate initial log probabilities
        total_count = sum(vocab.values())
        return {token: math.log(count / total_count) for token, count in vocab.items()}
    
    def _compute_expected_frequencies(self, word_freqs: Dict[str, int], model: Dict[str, float]) -> Dict[str, float]:
        """Compute expected frequencies of tokens using forward-backward algorithm."""
        token_expected_freqs = defaultdict(float)
        
        for word, freq in word_freqs.items():
            # Run the forward algorithm
            n = len(word)
            forward = [float("-inf")] * (n + 1)
            forward[0] = 0.0
            
            for i in range(n):
                if forward[i] == float("-inf"):
                    continue
                    
                for j in range(i + 1, min(n + 1, i + 20)):  # Limit max token length to 20
                    token = word[i:j]
                    score = model.get(token, float("-inf"))
                    
                    if score == float("-inf"):
                        continue
                        
                    forward[j] = np.logaddexp(forward[j], forward[i] + score)
            
            # If word can't be tokenized, skip it
            if forward[n] == float("-inf"):
                continue
                
            # Run the backward algorithm
            backward = [float("-inf")] * (n + 1)
            backward[n] = 0.0
            
            for i in range(n - 1, -1, -1):
                for j in range(i + 1, min(n + 1, i + 20)):
                    token = word[i:j]
                    score = model.get(token, float("-inf"))
                    
                    if score == float("-inf"):
                        continue
                        
                    backward[i] = np.logaddexp(backward[i], backward[j] + score)
            
            # Compute expected counts
            for i in range(n):
                for j in range(i + 1, min(n + 1, i + 20)):
                    token = word[i:j]
                    score = model.get(token, float("-inf"))
                    
                    if score == float("-inf"):
                        continue
                    
                    expected_count = forward[i] + score + backward[j] - forward[n]
                    expected_count = math.exp(expected_count) * freq
                    token_expected_freqs[token] += expected_count
        
        return token_expected_freqs
    
    def train(self, texts: List[str], num_iterations: int = 10, verbose: bool = True):
        """Train Unigram LM tokenizer on a corpus of texts.
        
        Args:
            texts: List of training texts
            num_iterations: Number of EM iterations
            verbose: Whether to show progress with tqdm
        """
        # 1. Compute word frequencies
        word_freqs = Counter()
        for text in texts:
            for word in text.split():
                word_freqs[word] += 1
        
        # 2. Initialize seed vocabulary with substrings
        model = self._initialize_seed_vocab(word_freqs)
        
        # 3. EM algorithm iterations
        if verbose:
            print(f"Initial model size: {len(model)}")
            pbar = tqdm(range(num_iterations), desc="Training Unigram tokenizer")
        else:
            pbar = range(num_iterations)
            
        for _ in pbar:
            # E-step: Compute expected frequencies
            token_expected_freqs = self._compute_expected_frequencies(word_freqs, model)
            
            # M-step: Re-estimate parameters
            total_freq = sum(token_expected_freqs.values())
            model = {token: math.log(freq / total_freq) for token, freq in token_expected_freqs.items()}
            
            # Prune the vocabulary
            if len(model) > self.vocab_size:
                sorted_tokens = sorted(model.items(), key=lambda x: x[1], reverse=True)
                model = dict(sorted_tokens[:self.vocab_size])
                
            if verbose:
                pbar.set_postfix({"vocab_size": len(model)})
        
        if verbose:
            print(f"Final model size: {len(model)}")
        
        # 4. Update tokenizer vocabulary with final model
        self.token_scores = model
        for token, score in model.items():
            self._add_to_vocab(token, score)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using Viterbi algorithm."""
        tokens = []
        for word in text.split():
            word_tokens = self._segment_word(word, self.token_scores)
            tokens.extend(word_tokens)
            # Add space token between words (except the last word)
            if word != text.split()[-1]:
                tokens.append(" ")
        
        return tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """Join tokens back into text."""
        return "".join(tokens)
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs using Unigram tokenization."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.id_to_token.get(id, self.unk_token) for id in ids]
        return self.detokenize(tokens)
    
    def save(self, path: str):
        """Save tokenizer vocabulary and scores to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'token_scores': {k: float(v) for k, v in self.token_scores.items()},
                'special_tokens': self.special_tokens
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'UnigramTokenizer':
        """Load tokenizer from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.vocab = {k: int(v) for k, v in data['vocab'].items()}
        tokenizer.token_scores = {k: float(v) for k, v in data['token_scores'].items()}
        tokenizer.special_tokens = data.get('special_tokens', tokenizer.special_tokens)
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.vocab.items()}
        tokenizer.next_id = max(tokenizer.vocab.values()) + 1 if tokenizer.vocab else 0
        
        return tokenizer 