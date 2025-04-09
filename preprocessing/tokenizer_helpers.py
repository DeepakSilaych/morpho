import re
from typing import List, Dict, Optional, Union, Callable, Set, Tuple
import unicodedata
from .normalizer import TextNormalizer

def get_word_boundaries(text: str) -> List[Tuple[int, int]]:
    """Get word boundaries in text.
    
    Args:
        text: Input text
        
    Returns:
        List of (start, end) tuples for word boundaries
    """
    # Use regex to find word boundaries
    word_pattern = re.compile(r'\w+', re.UNICODE)
    return [(match.start(), match.end()) for match in word_pattern.finditer(text)]

def split_on_whitespace(text: str) -> List[str]:
    """Split text on whitespace.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    return re.split(r'\s+', text.strip())

def split_on_punctuation(text: str) -> List[str]:
    """Split text on punctuation and whitespace.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    # First replace punctuation with space + punctuation + space
    for punct in r'.,;:!?()[]{}<>':
        text = text.replace(punct, f' {punct} ')
    
    # Then split on whitespace and filter empty tokens
    return [token for token in re.split(r'\s+', text.strip()) if token]

def preprocess_for_wordpiece(text: str) -> str:
    """Preprocess text for WordPiece tokenization.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    # Apply BERT-style normalization
    normalizer = TextNormalizer(
        lowercase=True,
        unicode_normalization='NFKC',
        replace_control_chars=True
    )
    return normalizer.normalize(text)

def preprocess_for_bpe(text: str) -> str:
    """Preprocess text for BPE tokenization.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    # Apply GPT-style normalization
    normalizer = TextNormalizer(
        lowercase=False,
        unicode_normalization='NFC',
        replace_control_chars=True
    )
    return normalizer.normalize(text)

def add_bos_eos(tokens: List[str], bos_token: str = "<s>", eos_token: str = "</s>") -> List[str]:
    """Add beginning and end of sequence tokens.
    
    Args:
        tokens: List of tokens
        bos_token: Beginning of sequence token
        eos_token: End of sequence token
        
    Returns:
        Tokens with BOS and EOS added
    """
    return [bos_token] + tokens + [eos_token]

def truncate_sequences(tokens_a: List[str], 
                      tokens_b: Optional[List[str]] = None, 
                      max_length: int = 512,
                      truncation_strategy: str = 'longest_first') -> Tuple[List[str], Optional[List[str]]]:
    """Truncate sequences to fit within max_length.
    
    Args:
        tokens_a: First sequence of tokens
        tokens_b: Optional second sequence of tokens
        max_length: Maximum combined length
        truncation_strategy: Strategy for truncation
            ('longest_first', 'only_first', 'only_second')
            
    Returns:
        Tuple of truncated sequences
    """
    if tokens_b is None:
        if len(tokens_a) > max_length:
            tokens_a = tokens_a[:max_length]
        return tokens_a, None
    
    if truncation_strategy == 'only_first':
        if len(tokens_a) > max_length:
            tokens_a = tokens_a[:max_length]
        return tokens_a, tokens_b
    
    if truncation_strategy == 'only_second':
        if len(tokens_b) > max_length:
            tokens_b = tokens_b[:max_length]
        return tokens_a, tokens_b
    
    # Default: 'longest_first'
    total_length = len(tokens_a) + len(tokens_b)
    
    if total_length <= max_length:
        return tokens_a, tokens_b
    
    # Truncate the longer sequence
    if len(tokens_a) > len(tokens_b):
        tokens_a = tokens_a[:max_length - len(tokens_b)]
    else:
        tokens_b = tokens_b[:max_length - len(tokens_a)]
    
    # Double check that we're within the limit
    if len(tokens_a) + len(tokens_b) > max_length:
        # If still too long, truncate both
        tokens_a = tokens_a[:max_length//2]
        tokens_b = tokens_b[:max_length - len(tokens_a)]
    
    return tokens_a, tokens_b

def is_control(char: str) -> bool:
    """Check if a character is a control character.
    
    Args:
        char: Character to check
        
    Returns:
        True if character is a control character
    """
    # Check if character is a control character (C category in Unicode)
    return unicodedata.category(char).startswith('C')

def is_whitespace(char: str) -> bool:
    """Check if a character is whitespace.
    
    Args:
        char: Character to check
        
    Returns:
        True if character is whitespace
    """
    # Check if character is whitespace (Z category in Unicode or tab, newline, etc.)
    return char.isspace() or unicodedata.category(char).startswith('Z')

def is_punctuation(char: str) -> bool:
    """Check if a character is punctuation.
    
    Args:
        char: Character to check
        
    Returns:
        True if character is punctuation
    """
    # Check if character is punctuation (P category in Unicode)
    return unicodedata.category(char).startswith('P')

def basic_clean(text: str) -> str:
    """Basic cleaning of text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Replace repeated whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Replace control characters
    text = ''.join([' ' if is_control(c) else c for c in text])
    
    # Normalize whitespace again
    text = re.sub(r'\s+', ' ', text)
    
    return text 