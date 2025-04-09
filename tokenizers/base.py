from typing import List
from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    """Base class for all tokenizers defining the common interface."""
    
    def __init__(self):
        """Initialize the tokenizer."""
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Convert text into a list of tokens.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        pass
    
    @abstractmethod
    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back into text.
        
        Args:
            tokens: List of tokens to combine
            
        Returns:
            Reconstructed text
        """
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text into token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back into text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Reconstructed text
        """
        pass 