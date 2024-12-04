"""Message class for tokenizer server communication."""
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

@dataclass
class Message:
    """Message class for communication between client and server."""
    method: str
    params: Dict[str, Any]
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create Message from JSON string."""
        data = json.loads(json_str)
        return cls(method=data['method'], params=data['params'])
    
    def to_json(self) -> str:
        """Convert Message to JSON string."""
        return json.dumps({
            'method': self.method,
            'params': self.params
        })

    @classmethod
    def encode_request(cls, text: str, add_bos: bool = True, add_special_tokens: bool = True) -> 'Message':
        """Create encode request message."""
        return cls(
            method='encode', 
            params={
                'text': text,
                'add_bos': add_bos,
                'add_special_tokens': add_special_tokens
            }
        )

    @classmethod
    def decode_request(cls, token_ids: List[int], offset: Optional[int] = None, skip_special_tokens: bool = True) -> 'Message':
        """Create decode request message."""
        return cls(
            method='decode',
            params={
                'token_ids': token_ids,
                'offset': offset,
                'skip_special_tokens': skip_special_tokens
            }
        )

    @classmethod
    def detokenize_incrementally_request(cls, token_ids: List[int], state: Dict[str, Any], **kwargs) -> 'Message':
        """Create detokenize incrementally request message."""
        return cls(
            method='detokenize_incrementally',
            params={
                'token_ids': token_ids,
                'state': state,
                **kwargs
            }
        )

    @classmethod
    def encode_response(cls, token_ids: List[int]) -> 'Message':
        """Create encode response message."""
        return cls(
            method='encode_response',
            params={'token_ids': token_ids}
        )

    @classmethod
    def decode_response(cls, text: str) -> 'Message':
        """Create decode response message."""
        return cls(
            method='decode_response',
            params={'text': text}
        )

    @classmethod
    def error_response(cls, error: str) -> 'Message':
        """Create error response message."""
        return cls(
            method='error',
            params={'error': error}
        )

    @classmethod
    def get_vocab_size_request(cls) -> 'Message':
        """Create get vocab size request message."""
        return cls(method='get_vocab_size', params={})

    @classmethod
    def get_bos_token_id_request(cls) -> 'Message':
        """Create get BOS token ID request message."""
        return cls(method='get_bos_token_id', params={})

    @classmethod
    def get_eos_token_id_request(cls) -> 'Message':
        """Create get EOS token ID request message."""
        return cls(method='get_eos_token_id', params={})

    @classmethod
    def get_vocab_size_with_added_request(cls) -> 'Message':
        """Create get vocab size with added request message."""
        return cls(method='get_vocab_size_with_added', params={})

    @classmethod
    def get_prefix_space_tokens_request(cls) -> 'Message':
        """Create get prefix space tokens request message."""
        return cls(method='get_prefix_space_tokens', params={})

    @classmethod
    def get_maybe_decode_bytes_request(cls) -> 'Message':
        """Create get maybe decode bytes request message."""
        return cls(method='get_maybe_decode_bytes', params={})

    @classmethod
    def indexes_containing_token_request(cls, token: str) -> 'Message':
        """Create indexes containing token request message."""
        return cls(method='indexes_containing_token', params={'token': token}) 