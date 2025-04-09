from typing import Dict, List, Optional, Set

class SpecialTokens:
    """Class to handle special tokens consistently across tokenizers.
    
    This class defines the standard special tokens used by different tokenizers
    and provides methods for consistent special token handling.
    """
    
    # Default special tokens
    PAD = "<pad>"  # Padding token
    UNK = "<unk>"  # Unknown token
    BOS = "<s>"    # Begin of sequence token
    EOS = "</s>"   # End of sequence token
    MASK = "<mask>"  # Mask token for masked language modeling
    
    # Special tokens for BERT-style models
    CLS = "[CLS]"  # Classification token
    SEP = "[SEP]"  # Separator token
    BERT_PAD = "[PAD]"
    BERT_UNK = "[UNK]"
    BERT_MASK = "[MASK]"
    
    # Additional special tokens
    CONTROL = "<control>"  # Used for control characters
    
    @classmethod
    def get_default_special_tokens(cls, tokenizer_type: str = "standard") -> Dict[str, str]:
        """Get the default special tokens for a given tokenizer type.
        
        Args:
            tokenizer_type: Type of tokenizer ('standard', 'bert', 'gpt', etc.)
            
        Returns:
            Dictionary of special token names and values
        """
        if tokenizer_type.lower() in ["bert", "wordpiece"]:
            return {
                "pad_token": cls.BERT_PAD,
                "unk_token": cls.BERT_UNK,
                "cls_token": cls.CLS,
                "sep_token": cls.SEP,
                "mask_token": cls.BERT_MASK
            }
        elif tokenizer_type.lower() in ["gpt", "gpt2", "bpe"]:
            return {
                "pad_token": cls.PAD,
                "unk_token": cls.UNK,
                "bos_token": cls.BOS,
                "eos_token": cls.EOS
            }
        else:  # standard
            return {
                "pad_token": cls.PAD,
                "unk_token": cls.UNK,
                "bos_token": cls.BOS,
                "eos_token": cls.EOS,
                "mask_token": cls.MASK
            }
    
    @classmethod
    def is_special_token(cls, token: str, special_tokens: Optional[List[str]] = None) -> bool:
        """Check if a token is a special token.
        
        Args:
            token: Token to check
            special_tokens: Optional list of special tokens to check against
            
        Returns:
            True if the token is a special token, False otherwise
        """
        if special_tokens:
            return token in special_tokens
            
        # Check against all known special tokens
        all_special_tokens = {
            cls.PAD, cls.UNK, cls.BOS, cls.EOS, cls.MASK,
            cls.CLS, cls.SEP, cls.BERT_PAD, cls.BERT_UNK, cls.BERT_MASK,
            cls.CONTROL
        }
        return token in all_special_tokens
    
    @classmethod
    def get_special_tokens_mask(cls, token_list: List[str], special_tokens: Optional[List[str]] = None) -> List[int]:
        """Create a mask of 1s for special tokens and 0s for regular tokens.
        
        Args:
            token_list: List of tokens
            special_tokens: Optional list of special tokens to check against
            
        Returns:
            A list of 1s and 0s (1 for special tokens)
        """
        return [1 if cls.is_special_token(token, special_tokens) else 0 for token in token_list]
    
    @classmethod
    def add_special_tokens(cls, tokens: List[str], bos: bool = False, eos: bool = False, 
                          tokenizer_type: str = "standard") -> List[str]:
        """Add special tokens to a list of tokens.
        
        Args:
            tokens: List of tokens
            bos: Whether to add begin of sequence token
            eos: Whether to add end of sequence token
            tokenizer_type: Type of tokenizer
            
        Returns:
            Tokens with special tokens added
        """
        special_tokens = cls.get_default_special_tokens(tokenizer_type)
        result = tokens.copy()
        
        if bos:
            bos_token = special_tokens.get("bos_token", cls.BOS)
            if tokenizer_type.lower() in ["bert", "wordpiece"]:
                bos_token = special_tokens.get("cls_token", cls.CLS)
            result.insert(0, bos_token)
            
        if eos:
            eos_token = special_tokens.get("eos_token", cls.EOS)
            if tokenizer_type.lower() in ["bert", "wordpiece"]:
                eos_token = special_tokens.get("sep_token", cls.SEP)
            result.append(eos_token)
            
        return result
    
    @classmethod
    def remove_special_tokens(cls, tokens: List[str], special_tokens: Optional[List[str]] = None) -> List[str]:
        """Remove special tokens from a list of tokens.
        
        Args:
            tokens: List of tokens
            special_tokens: Optional list of special tokens to remove
            
        Returns:
            Tokens with special tokens removed
        """
        if special_tokens:
            return [token for token in tokens if token not in special_tokens]
        
        # Remove all known special tokens
        all_special_tokens = {
            cls.PAD, cls.UNK, cls.BOS, cls.EOS, cls.MASK,
            cls.CLS, cls.SEP, cls.BERT_PAD, cls.BERT_UNK, cls.BERT_MASK,
            cls.CONTROL
        }
        return [token for token in tokens if token not in all_special_tokens] 