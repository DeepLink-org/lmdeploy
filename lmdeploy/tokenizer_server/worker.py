"""Worker process functions for tokenizer server."""

from typing import List, Optional, Set
from transformers import PreTrainedTokenizer

def encode_text(tokenizer: PreTrainedTokenizer,
               text: str,
               add_bos: bool = True,
               add_special_tokens: bool = True) -> List[int]:
    """Encode text to token IDs.
    
    Args:
        tokenizer: The tokenizer instance
        text: Text to encode
        add_bos: Whether to add BOS token
        add_special_tokens: Whether to add special tokens
        
    Returns:
        List of token IDs
    """
    if add_bos and add_special_tokens:
        if tokenizer.bos_token_id is not None:
            token_ids = [tokenizer.bos_token_id]
        else:
            token_ids = []
        token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    else:
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        
    return [int(x) for x in token_ids]

def decode_tokens(tokenizer: PreTrainedTokenizer,
                 token_ids: List[int],
                 skip_special_tokens: bool = True) -> str:
    """Decode token IDs to text.
    
    Args:
        tokenizer: The tokenizer instance
        token_ids: List of token IDs to decode
        skip_special_tokens: Whether to skip special tokens
        
    Returns:
        Decoded text
    """
    token_ids = [int(x) for x in token_ids]
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

def find_token_indexes(tokenizer: PreTrainedTokenizer, token: str) -> List[int]:
    """Find token IDs containing the given token.
    
    Args:
        tokenizer: The tokenizer instance
        token: Token to search for
        
    Returns:
        List of token IDs
    """
    vocab = tokenizer.get_vocab()
    return [int(idx) for idx, tok in vocab.items() if token in str(tok)] 