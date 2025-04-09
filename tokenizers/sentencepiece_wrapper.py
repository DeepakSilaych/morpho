from typing import List, Optional
from .base import BaseTokenizer

class SentencePieceWrapper(BaseTokenizer):
    """Wrapper for SentencePiece tokenizer."""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        self.model_path = model_path
        self.sp_model = None
        if model_path:
            try:
                import sentencepiece as spm
                self.sp_model = spm.SentencePieceProcessor()
                self.sp_model.Load(model_path)
            except ImportError:
                raise ImportError("Please install sentencepiece: pip install sentencepiece")
            except Exception as e:
                raise RuntimeError(f"Error loading SentencePiece model: {e}")
    
    def train(self, input_file: str, vocab_size: int = 8000, 
              model_type: str = "unigram", model_prefix: str = "sp"):
        """Train a new SentencePiece model.
        
        Args:
            input_file: Path to text file for training
            vocab_size: Size of vocabulary
            model_type: Type of model ('unigram' or 'bpe')
            model_prefix: Prefix for saving model files
        """
        try:
            import sentencepiece as spm
            spm.SentencePieceTrainer.Train(
                f'--input={input_file} '
                f'--model_prefix={model_prefix} '
                f'--vocab_size={vocab_size} '
                f'--model_type={model_type}'
            )
            self.model_path = f"{model_prefix}.model"
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(self.model_path)
        except ImportError:
            raise ImportError("Please install sentencepiece: pip install sentencepiece")
        except Exception as e:
            raise RuntimeError(f"Error training SentencePiece model: {e}")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using SentencePiece model."""
        if not self.sp_model:
            raise RuntimeError("No SentencePiece model loaded")
        return self.sp_model.EncodeAsPieces(text)
    
    def detokenize(self, tokens: List[str]) -> str:
        """Join tokens back into text."""
        if not self.sp_model:
            raise RuntimeError("No SentencePiece model loaded")
        return self.sp_model.DecodePieces(tokens)
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs using SentencePiece model."""
        if not self.sp_model:
            raise RuntimeError("No SentencePiece model loaded")
        return self.sp_model.EncodeAsIds(text)
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        if not self.sp_model:
            raise RuntimeError("No SentencePiece model loaded")
        return self.sp_model.DecodeIds(ids) 