from typing import List, Dict, Optional, Union, Any
import json
import os
from ..tokenizers.base import BaseTokenizer

class Encoder:
    """Encoder class for encoding and decoding between text and token IDs.
    
    This class provides a unified interface for encoding texts using different
    tokenizers, with support for batched encoding, padding, truncation, and
    other common operations.
    """
    
    def __init__(self, 
                 tokenizer: BaseTokenizer,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None,
                 bos_token_id: Optional[int] = None,
                 mask_token_id: Optional[int] = None,
                 max_length: Optional[int] = None):
        """Initialize an encoder with a tokenizer.
        
        Args:
            tokenizer: The tokenizer to use for encoding
            pad_token_id: ID of the padding token
            eos_token_id: ID of the end of sequence token
            bos_token_id: ID of the beginning of sequence token
            mask_token_id: ID of the mask token
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        
        # Try to get token IDs from tokenizer if not provided
        vocab = getattr(tokenizer, 'vocab', {})
        
        self.pad_token_id = pad_token_id
        if self.pad_token_id is None and hasattr(tokenizer, 'pad_token'):
            self.pad_token_id = vocab.get(tokenizer.pad_token, None)
        
        self.eos_token_id = eos_token_id
        if self.eos_token_id is None and hasattr(tokenizer, 'eos_token'):
            self.eos_token_id = vocab.get(tokenizer.eos_token, None)
        
        self.bos_token_id = bos_token_id
        if self.bos_token_id is None and hasattr(tokenizer, 'bos_token'):
            self.bos_token_id = vocab.get(tokenizer.bos_token, None)
        
        self.mask_token_id = mask_token_id
        if self.mask_token_id is None and hasattr(tokenizer, 'mask_token'):
            self.mask_token_id = vocab.get(tokenizer.mask_token, None)
        
        self.max_length = max_length
    
    def encode(self, 
               text: str, 
               add_special_tokens: bool = True,
               truncation: bool = False,
               max_length: Optional[int] = None) -> List[int]:
        """Encode text into token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            truncation: Whether to truncate to max_length
            max_length: Maximum sequence length (overrides self.max_length)
            
        Returns:
            List of token IDs
        """
        # Use tokenizer.encode if available
        if hasattr(self.tokenizer, 'encode'):
            ids = self.tokenizer.encode(text)
        else:
            # Fallback to tokenize and manual conversion
            tokens = self.tokenizer.tokenize(text)
            ids = [self.tokenizer.vocab.get(token, self.tokenizer.vocab.get('<unk>', 0)) 
                  for token in tokens]
        
        # Add special tokens if requested
        if add_special_tokens:
            if self.bos_token_id is not None:
                ids = [self.bos_token_id] + ids
            if self.eos_token_id is not None:
                ids = ids + [self.eos_token_id]
        
        # Apply truncation if requested
        max_len = max_length or self.max_length
        if truncation and max_len and len(ids) > max_len:
            ids = ids[:max_len]
        
        return ids
    
    def decode(self, 
               ids: List[int], 
               skip_special_tokens: bool = True,
               clean_up_tokenization_spaces: bool = True) -> str:
        """Decode token IDs into text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces
            
        Returns:
            Decoded text
        """
        # Skip special tokens if requested
        if skip_special_tokens:
            special_ids = []
            if self.pad_token_id is not None:
                special_ids.append(self.pad_token_id)
            if self.bos_token_id is not None:
                special_ids.append(self.bos_token_id)
            if self.eos_token_id is not None:
                special_ids.append(self.eos_token_id)
            if self.mask_token_id is not None:
                special_ids.append(self.mask_token_id)
            
            ids = [id for id in ids if id not in special_ids]
        
        # Use tokenizer.decode if available
        if hasattr(self.tokenizer, 'decode'):
            text = self.tokenizer.decode(ids)
        else:
            # Fallback to manual conversion
            tokens = [self.tokenizer.id_to_token.get(id, '<unk>') for id in ids]
            text = self.tokenizer.detokenize(tokens)
        
        # Clean up tokenization spaces
        if clean_up_tokenization_spaces:
            text = text.replace(' ##', '')  # For WordPiece
            text = text.replace('▁', ' ')   # For SentencePiece
            text = ' '.join(text.split())   # Normalize whitespace
        
        return text
    
    def encode_batch(self, 
                    texts: List[str], 
                    add_special_tokens: bool = True,
                    padding: bool = False,
                    truncation: bool = False,
                    max_length: Optional[int] = None,
                    return_tensors: Optional[str] = None) -> Dict[str, Any]:
        """Encode a batch of texts.
        
        Args:
            texts: List of input texts
            add_special_tokens: Whether to add special tokens
            padding: Whether to pad sequences to the same length
            truncation: Whether to truncate to max_length
            max_length: Maximum sequence length (overrides self.max_length)
            return_tensors: Optional tensor format ('pt' for PyTorch, 'tf' for TensorFlow)
            
        Returns:
            Dictionary with encoded data
        """
        # Encode each text
        batch_ids = [self.encode(text, add_special_tokens, truncation, max_length) 
                    for text in texts]
        
        # Get max length in batch
        batch_max_length = max(len(ids) for ids in batch_ids)
        max_len = max_length or self.max_length or batch_max_length
        
        # Apply padding if requested
        attention_mask = None
        if padding:
            if self.pad_token_id is None:
                raise ValueError("Padding requires pad_token_id to be set")
            
            attention_mask = []
            padded_ids = []
            
            for ids in batch_ids:
                # Create attention mask (1 for real tokens, 0 for padding)
                mask = [1] * len(ids) + [0] * (max_len - len(ids))
                attention_mask.append(mask[:max_len])
                
                # Pad sequence
                padded = ids + [self.pad_token_id] * (max_len - len(ids))
                padded_ids.append(padded[:max_len])
            
            batch_ids = padded_ids
        
        # Convert to tensors if requested
        if return_tensors:
            if return_tensors == 'pt':
                try:
                    import torch
                    batch_ids = torch.tensor(batch_ids)
                    if attention_mask:
                        attention_mask = torch.tensor(attention_mask)
                except ImportError:
                    raise ImportError("PyTorch not installed. Install it with 'pip install torch'")
            elif return_tensors == 'tf':
                try:
                    import tensorflow as tf
                    batch_ids = tf.constant(batch_ids)
                    if attention_mask:
                        attention_mask = tf.constant(attention_mask)
                except ImportError:
                    raise ImportError("TensorFlow not installed. Install it with 'pip install tensorflow'")
            else:
                raise ValueError(f"Unsupported tensor format: {return_tensors}")
        
        # Prepare output
        output = {'input_ids': batch_ids}
        if attention_mask:
            output['attention_mask'] = attention_mask
        
        return output
    
    def decode_batch(self, 
                    batch_ids: List[List[int]], 
                    skip_special_tokens: bool = True,
                    clean_up_tokenization_spaces: bool = True) -> List[str]:
        """Decode a batch of token IDs.
        
        Args:
            batch_ids: List of lists of token IDs
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces
            
        Returns:
            List of decoded texts
        """
        return [self.decode(ids, skip_special_tokens, clean_up_tokenization_spaces) 
                for ids in batch_ids]
    
    def save_pretrained(self, path: str):
        """Save encoder configuration to disk.
        
        Args:
            path: Path to save the encoder
        """
        os.makedirs(path, exist_ok=True)
        
        # Save encoder configuration
        config = {
            'pad_token_id': self.pad_token_id,
            'eos_token_id': self.eos_token_id,
            'bos_token_id': self.bos_token_id,
            'mask_token_id': self.mask_token_id,
            'max_length': self.max_length
        }
        
        with open(os.path.join(path, 'encoder_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # Save tokenizer if it has a save method
        if hasattr(self.tokenizer, 'save'):
            self.tokenizer.save(os.path.join(path, 'tokenizer.json'))
    
    @classmethod
    def from_pretrained(cls, path: str, tokenizer: Optional[BaseTokenizer] = None) -> 'Encoder':
        """Load encoder from disk.
        
        Args:
            path: Path to load the encoder from
            tokenizer: Optional tokenizer (loaded from path if not provided)
            
        Returns:
            Loaded encoder
        """
        # Load configuration
        with open(os.path.join(path, 'encoder_config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Load tokenizer if not provided
        if tokenizer is None:
            tokenizer_path = os.path.join(path, 'tokenizer.json')
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
            
            # Try to determine tokenizer type
            metadata_path = os.path.join(path, 'tokenizer_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                tokenizer_type = metadata.get('tokenizer_type')
                
                # Import and load the tokenizer
                if tokenizer_type == 'whitespace':
                    from ..tokenizers.whitespace import WhitespaceTokenizer
                    tokenizer = WhitespaceTokenizer.load(tokenizer_path)
                elif tokenizer_type == 'character':
                    from ..tokenizers.character import CharacterTokenizer
                    tokenizer = CharacterTokenizer.load(tokenizer_path)
                elif tokenizer_type == 'bpe':
                    from ..tokenizers.bpe import BPETokenizer
                    tokenizer = BPETokenizer.load(tokenizer_path)
                elif tokenizer_type == 'wordpiece':
                    from ..tokenizers.wordpiece import WordPieceTokenizer
                    tokenizer = WordPieceTokenizer.load(tokenizer_path)
                elif tokenizer_type == 'unigram':
                    from ..tokenizers.unigram import UnigramTokenizer
                    tokenizer = UnigramTokenizer.load(tokenizer_path)
                elif tokenizer_type == 'sentencepiece':
                    from ..tokenizers.sentencepiece_wrapper import SentencePieceWrapper
                    tokenizer = SentencePieceWrapper.load(tokenizer_path)
                else:
                    raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
            else:
                # Try to load with each tokenizer type
                for tokenizer_type in ['bpe', 'wordpiece', 'unigram', 'character', 'whitespace']:
                    try:
                        if tokenizer_type == 'whitespace':
                            from ..tokenizers.whitespace import WhitespaceTokenizer
                            tokenizer = WhitespaceTokenizer.load(tokenizer_path)
                        elif tokenizer_type == 'character':
                            from ..tokenizers.character import CharacterTokenizer
                            tokenizer = CharacterTokenizer.load(tokenizer_path)
                        elif tokenizer_type == 'bpe':
                            from ..tokenizers.bpe import BPETokenizer
                            tokenizer = BPETokenizer.load(tokenizer_path)
                        elif tokenizer_type == 'wordpiece':
                            from ..tokenizers.wordpiece import WordPieceTokenizer
                            tokenizer = WordPieceTokenizer.load(tokenizer_path)
                        elif tokenizer_type == 'unigram':
                            from ..tokenizers.unigram import UnigramTokenizer
                            tokenizer = UnigramTokenizer.load(tokenizer_path)
                        break
                    except:
                        continue
                
                if not tokenizer:
                    raise ValueError("Could not determine tokenizer type and load the tokenizer")
        
        # Create encoder with loaded configuration
        return cls(
            tokenizer=tokenizer,
            pad_token_id=config.get('pad_token_id'),
            eos_token_id=config.get('eos_token_id'),
            bos_token_id=config.get('bos_token_id'),
            mask_token_id=config.get('mask_token_id'),
            max_length=config.get('max_length')
        ) 