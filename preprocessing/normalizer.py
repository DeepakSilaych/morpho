import unicodedata
import re
from typing import List, Dict, Union, Optional, Pattern, Set


class TextNormalizer:
    """Text normalizer for preprocessing text before tokenization.
    
    Features:
    - Unicode normalization
    - Case normalization
    - Whitespace normalization
    - Punctuation handling
    - Control character handling
    """
    
    def __init__(self, 
                 lowercase: bool = False,
                 unicode_normalization: Optional[str] = 'NFKC',
                 strip_accents: bool = False,
                 strip_punctuation: bool = False,
                 replace_control_chars: bool = True,
                 control_char_replacement: str = ' ',
                 custom_replacements: Optional[Dict[str, str]] = None):
        """Initialize text normalizer.
        
        Args:
            lowercase: Whether to convert text to lowercase
            unicode_normalization: Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD', or None)
            strip_accents: Whether to strip accents from text
            strip_punctuation: Whether to strip punctuation from text
            replace_control_chars: Whether to replace control characters
            control_char_replacement: String to replace control characters with
            custom_replacements: Dictionary of custom replacements
        """
        self.lowercase = lowercase
        self.unicode_normalization = unicode_normalization
        self.strip_accents = strip_accents
        self.strip_punctuation = strip_punctuation
        self.replace_control_chars = replace_control_chars
        self.control_char_replacement = control_char_replacement
        self.custom_replacements = custom_replacements or {}
        
        # Compile regex patterns
        self.punctuation_pattern = re.compile(r'[\p{P}]', re.UNICODE)
        self.whitespace_pattern = re.compile(r'\s+')
        self.control_char_pattern = re.compile(r'[\p{C}]', re.UNICODE)
    
    def normalize(self, text: str) -> str:
        """Normalize text according to the configured options.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Apply custom replacements first
        for old, new in self.custom_replacements.items():
            text = text.replace(old, new)
        
        # Apply unicode normalization
        if self.unicode_normalization:
            text = unicodedata.normalize(self.unicode_normalization, text)
        
        # Convert to lowercase if enabled
        if self.lowercase:
            text = text.lower()
        
        # Strip accents if enabled
        if self.strip_accents:
            text = self._strip_accents(text)
        
        # Replace control characters if enabled
        if self.replace_control_chars:
            text = self.control_char_pattern.sub(self.control_char_replacement, text)
        
        # Strip punctuation if enabled
        if self.strip_punctuation:
            text = self.punctuation_pattern.sub('', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def _strip_accents(self, text: str) -> str:
        """Strip accents from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with accents removed
        """
        text = unicodedata.normalize('NFD', text)
        return ''.join([c for c in text if not unicodedata.combining(c)])
    
    def batch_normalize(self, texts: List[str]) -> List[str]:
        """Normalize a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of normalized texts
        """
        return [self.normalize(text) for text in texts]


class RegexTokenizerMixin:
    """Mixin class for regex-based tokenization.
    
    This class provides methods for tokenizing text using regular expressions,
    which can be used by various tokenizers.
    """
    
    def __init__(self, pattern: Optional[Union[str, Pattern]] = None):
        """Initialize regex tokenizer.
        
        Args:
            pattern: Regex pattern for tokenization (defaults to whitespace)
        """
        self.pattern = pattern or re.compile(r'\s+')
        
        # Ensure pattern is compiled
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern)
    
    def split(self, text: str) -> List[str]:
        """Split text using the pattern.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self.pattern.split(text.strip())
    
    def findall(self, text: str) -> List[str]:
        """Find all matches in text.
        
        Args:
            text: Input text
            
        Returns:
            List of matched tokens
        """
        return self.pattern.findall(text)


def normalize_for_tokenizer(text: str, tokenizer_type: str = 'standard') -> str:
    """Normalize text for a specific tokenizer type.
    
    Args:
        text: Input text
        tokenizer_type: Type of tokenizer
        
    Returns:
        Normalized text
    """
    if tokenizer_type.lower() in ['bert', 'wordpiece']:
        # BERT-style normalization
        normalizer = TextNormalizer(
            lowercase=True,
            unicode_normalization='NFKC',
            strip_accents=False,
            strip_punctuation=False,
            replace_control_chars=True
        )
    elif tokenizer_type.lower() in ['bpe', 'gpt', 'gpt2']:
        # BPE-style normalization
        normalizer = TextNormalizer(
            lowercase=False,
            unicode_normalization='NFC',
            strip_accents=False,
            strip_punctuation=False,
            replace_control_chars=True
        )
    elif tokenizer_type.lower() == 'unigram':
        # Unigram-style normalization
        normalizer = TextNormalizer(
            lowercase=False,
            unicode_normalization='NFKC',
            strip_accents=False,
            strip_punctuation=False,
            replace_control_chars=True
        )
    else:
        # Standard normalization
        normalizer = TextNormalizer(
            lowercase=False,
            unicode_normalization='NFKC',
            strip_accents=False,
            strip_punctuation=False,
            replace_control_chars=True
        )
    
    return normalizer.normalize(text) 