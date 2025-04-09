"""
Morpho: A modular, extensible, and diverse tokenization library for NLP and LLMs.

This library provides implementations of multiple tokenization algorithms including:
- Whitespace tokenizer
- Character tokenizer
- Byte-Pair Encoding (BPE) tokenizer
- WordPiece tokenizer (used in BERT)
- Unigram Language Model tokenizer
- SentencePiece wrapper
"""

from tokenizers.base import BaseTokenizer
from tokenizers.whitespace import WhitespaceTokenizer
from tokenizers.character import CharacterTokenizer
from tokenizers.bpe import BPETokenizer
from tokenizers.wordpiece import WordPieceTokenizer
from tokenizers.unigram import UnigramTokenizer
from tokenizers.sentencepiece_wrapper import SentencePieceWrapper

from vocab.vocab import Vocabulary
from vocab.trainer import TokenizerTrainer
from vocab.special_tokens import SpecialTokens

from encoders.encoder import Encoder

__version__ = "0.1.0"

__all__ = [
    'BaseTokenizer',
    'WhitespaceTokenizer',
    'CharacterTokenizer',
    'BPETokenizer',
    'WordPieceTokenizer',
    'UnigramTokenizer',
    'SentencePieceWrapper',
    'Vocabulary',
    'TokenizerTrainer',
    'SpecialTokens',
    'Encoder',
] 