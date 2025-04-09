from typing import List, Dict, Optional, Union, Type, Any
import os
import json
import re
from tqdm import tqdm
import logging
from collections import Counter

from ..tokenizers.base import BaseTokenizer
from ..tokenizers.whitespace import WhitespaceTokenizer
from ..tokenizers.character import CharacterTokenizer
from ..tokenizers.bpe import BPETokenizer
from ..tokenizers.wordpiece import WordPieceTokenizer
from ..tokenizers.unigram import UnigramTokenizer
from ..tokenizers.sentencepiece_wrapper import SentencePieceWrapper

logger = logging.getLogger(__name__)

class TokenizerTrainer:
    """Trainer class for different tokenizers.
    
    This class provides a unified interface for training different types of
    tokenizers, with support for data loading, preprocessing, and evaluation.
    """
    
    TOKENIZER_TYPES = {
        'whitespace': WhitespaceTokenizer,
        'character': CharacterTokenizer,
        'bpe': BPETokenizer,
        'wordpiece': WordPieceTokenizer,
        'unigram': UnigramTokenizer,
        'sentencepiece': SentencePieceWrapper
    }
    
    def __init__(self, 
                 tokenizer_type: str, 
                 vocab_size: int = 30000,
                 min_frequency: int = 2,
                 special_tokens: Optional[Dict[str, str]] = None,
                 pre_tokenize_pattern: Optional[str] = None,
                 **kwargs):
        """Initialize a tokenizer trainer.
        
        Args:
            tokenizer_type: Type of tokenizer to train ('whitespace', 'character', 'bpe', etc.)
            vocab_size: Maximum vocabulary size
            min_frequency: Minimum frequency for a token to be included
            special_tokens: Dictionary of special tokens (unk_token, pad_token, etc.)
            pre_tokenize_pattern: Regex pattern for pre-tokenization (BPE only)
            **kwargs: Additional arguments for the tokenizer
        """
        if tokenizer_type not in self.TOKENIZER_TYPES:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. " 
                           f"Available types: {list(self.TOKENIZER_TYPES.keys())}")
        
        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.pre_tokenize_pattern = pre_tokenize_pattern
        
        # Initialize special tokens with defaults
        self.special_tokens = {
            'unk_token': '<unk>',
            'pad_token': '<pad>',
            'bos_token': '<s>',
            'eos_token': '</s>',
            'mask_token': '[MASK]' if tokenizer_type == 'wordpiece' else None
        }
        
        # Update with user-provided special tokens
        if special_tokens:
            self.special_tokens.update(special_tokens)
        
        # Filter out None values
        self.special_tokens = {k: v for k, v in self.special_tokens.items() if v is not None}
        
        # Store additional kwargs
        self.kwargs = kwargs
        
        # Initialize tokenizer
        self.tokenizer = self._create_tokenizer()
    
    def _create_tokenizer(self) -> BaseTokenizer:
        """Create a tokenizer instance based on the specified type."""
        tokenizer_class = self.TOKENIZER_TYPES[self.tokenizer_type]
        
        # Add appropriate special tokens based on tokenizer type
        tokenizer_kwargs = self.kwargs.copy()
        
        if self.tokenizer_type == 'wordpiece':
            tokenizer_kwargs.update({
                'vocab_size': self.vocab_size,
                'min_frequency': self.min_frequency,
                'unk_token': self.special_tokens.get('unk_token', '[UNK]'),
                'pad_token': self.special_tokens.get('pad_token', '[PAD]'),
                'cls_token': self.special_tokens.get('cls_token', '[CLS]'),
                'sep_token': self.special_tokens.get('sep_token', '[SEP]'),
                'mask_token': self.special_tokens.get('mask_token', '[MASK]')
            })
        elif self.tokenizer_type == 'bpe':
            tokenizer_kwargs.update({
                'vocab_size': self.vocab_size,
                'unk_token': self.special_tokens.get('unk_token', '<unk>'),
                'pad_token': self.special_tokens.get('pad_token', '<pad>'),
                'bos_token': self.special_tokens.get('bos_token', '<s>'),
                'eos_token': self.special_tokens.get('eos_token', '</s>')
            })
            
            # Add pre-tokenize pattern if provided
            if self.pre_tokenize_pattern:
                tokenizer_kwargs['pre_tokenize_pattern'] = re.compile(self.pre_tokenize_pattern)
                
        elif self.tokenizer_type == 'character':
            tokenizer_kwargs.update({
                'unk_token': self.special_tokens.get('unk_token', '<unk>'),
                'pad_token': self.special_tokens.get('pad_token', '<pad>'),
                'bos_token': self.special_tokens.get('bos_token', '<s>'),
                'eos_token': self.special_tokens.get('eos_token', '</s>')
            })
        
        return tokenizer_class(**tokenizer_kwargs)
    
    def train_from_files(self, 
                        files: List[str], 
                        max_lines: Optional[int] = None,
                        verbose: bool = True) -> BaseTokenizer:
        """Train tokenizer on a list of text files.
        
        Args:
            files: List of file paths
            max_lines: Maximum number of lines to read from each file
            verbose: Whether to show progress bar
            
        Returns:
            Trained tokenizer
        """
        texts = []
        
        # Load text from files
        for file_path in files:
            if verbose:
                logger.info(f"Loading text from {file_path}")
            
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if max_lines:
                    file_texts = [next(f).strip() for _ in range(max_lines) if f]
                else:
                    file_texts = [line.strip() for line in f]
                texts.extend(file_texts)
        
        return self.train_from_texts(texts, verbose)
    
    def train_from_texts(self, 
                         texts: List[str], 
                         verbose: bool = True) -> BaseTokenizer:
        """Train tokenizer on a list of texts.
        
        Args:
            texts: List of text strings
            verbose: Whether to show progress bar
            
        Returns:
            Trained tokenizer
        """
        if not texts:
            raise ValueError("No texts provided for training.")
        
        if verbose:
            logger.info(f"Training {self.tokenizer_type} tokenizer on {len(texts)} texts " +
                       f"with vocab size {self.vocab_size}")
        
        # Special handling for SentencePiece which requires a file
        if self.tokenizer_type == 'sentencepiece':
            # Create a temporary file
            temp_file = 'temp_corpus.txt'
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    for text in texts:
                        f.write(text + '\n')
                
                # Train the SentencePiece model
                self.tokenizer.train(
                    input_file=temp_file,
                    vocab_size=self.vocab_size,
                    model_type=self.kwargs.get('model_type', 'unigram')
                )
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        # For other tokenizer types
        elif hasattr(self.tokenizer, 'train'):
            # Different tokenizers have different training methods
            if self.tokenizer_type == 'bpe':
                self.tokenizer.train(texts, min_frequency=self.min_frequency, verbose=verbose)
            elif self.tokenizer_type == 'wordpiece':
                self.tokenizer.train(texts, verbose=verbose)
            elif self.tokenizer_type == 'unigram':
                self.tokenizer.train(texts, num_iterations=self.kwargs.get('num_iterations', 10), verbose=verbose)
            else:
                # Generic training method fallback
                if hasattr(self.tokenizer, 'build_vocab_from_texts'):
                    self.tokenizer.build_vocab_from_texts(texts, min_freq=self.min_frequency)
        
        # For tokenizers without a train method, build vocabulary directly
        elif hasattr(self.tokenizer, 'build_vocab_from_texts'):
            self.tokenizer.build_vocab_from_texts(texts, min_freq=self.min_frequency)
        
        if verbose:
            vocab_size = len(getattr(self.tokenizer, 'vocab', {}))
            logger.info(f"Tokenizer trained. Vocabulary size: {vocab_size}")
        
        return self.tokenizer
    
    def save_tokenizer(self, path: str):
        """Save tokenizer to disk.
        
        Args:
            path: Path to save the tokenizer
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save tokenizer
        if hasattr(self.tokenizer, 'save'):
            self.tokenizer.save(path)
        else:
            # Fallback for tokenizers without save method
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'vocab': getattr(self.tokenizer, 'vocab', {})}, f)
        
        # Save metadata
        metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"
        metadata = {
            'tokenizer_type': self.tokenizer_type,
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'special_tokens': self.special_tokens,
            'pre_tokenize_pattern': self.pre_tokenize_pattern
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_tokenizer(cls, path: str) -> BaseTokenizer:
        """Load a trained tokenizer from disk.
        
        Args:
            path: Path to the saved tokenizer
            
        Returns:
            Loaded tokenizer
        """
        # Load metadata
        metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            tokenizer_type = metadata.get('tokenizer_type')
            pre_tokenize_pattern = metadata.get('pre_tokenize_pattern')
        else:
            # Try to infer tokenizer type from filename
            tokenizer_type = None
            for t in cls.TOKENIZER_TYPES:
                if t in path.lower():
                    tokenizer_type = t
                    break
            
            if tokenizer_type is None:
                tokenizer_type = 'bpe'  # Default to BPE if unknown
                
            pre_tokenize_pattern = None
        
        # Create tokenizer
        if tokenizer_type == 'sentencepiece':
            # SentencePiece has a different loading mechanism
            tokenizer = SentencePieceWrapper.load(path)
        else:
            tokenizer_class = cls.TOKENIZER_TYPES.get(tokenizer_type)
            if not tokenizer_class:
                raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
            
            if hasattr(tokenizer_class, 'load'):
                # For tokenizers with a load method
                if tokenizer_type == 'bpe' and pre_tokenize_pattern:
                    # First load the tokenizer
                    tokenizer = tokenizer_class.load(path)
                    # Then set the pre-tokenization pattern if needed
                    tokenizer.pre_tokenize_pattern = re.compile(pre_tokenize_pattern)
                else:
                    tokenizer = tokenizer_class.load(path)
            else:
                # Fallback for tokenizers without load method
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                tokenizer = tokenizer_class()
                if hasattr(tokenizer, 'vocab'):
                    tokenizer.vocab = data.get('vocab', {})
        
        return tokenizer
    
    def evaluate(self, texts: List[str], metrics: List[str] = None) -> Dict[str, float]:
        """Evaluate tokenizer on a list of texts.
        
        Args:
            texts: List of text strings
            metrics: List of metrics to compute (vocabulary coverage, compression ratio, etc.)
            
        Returns:
            Dictionary of metric names and values
        """
        if not metrics:
            metrics = ['vocab_coverage', 'compression_ratio', 'token_frequency']
        
        results = {}
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            all_tokens.extend(tokens)
        
        # Compute metrics
        if 'vocab_coverage' in metrics:
            # Percentage of tokens that are in the vocabulary
            vocab = getattr(self.tokenizer, 'vocab', {})
            
            if vocab:
                unknown_count = sum(1 for token in all_tokens 
                                 if token not in vocab and token != self.special_tokens.get('unk_token'))
                results['vocab_coverage'] = 1.0 - (unknown_count / len(all_tokens)) if all_tokens else 0.0
        
        if 'compression_ratio' in metrics:
            # Average number of characters per token
            char_count = sum(len(text) for text in texts)
            token_count = len(all_tokens)
            results['compression_ratio'] = char_count / token_count if token_count else 0.0
        
        if 'token_frequency' in metrics:
            # Distribution of token frequencies
            token_counter = Counter(all_tokens)
            token_counts = list(token_counter.values())
            results['token_frequency_mean'] = sum(token_counts) / len(token_counts) if token_counts else 0.0
            results['token_frequency_median'] = sorted(token_counts)[len(token_counts) // 2] if token_counts else 0.0
        
        return results 