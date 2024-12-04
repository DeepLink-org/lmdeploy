"""Remote tokenizer implementation."""
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import zmq

from ..tokenizer import HuggingFaceTokenizer, DetokenizeState
from .messages import Message

logger = logging.getLogger(__name__)

class RemoteTokenizer(HuggingFaceTokenizer):
    """Remote tokenizer that communicates with tokenizer server."""

    def __init__(self, model_path: str, host: str = "127.0.0.1", port: int = 5555):
        """Initialize remote tokenizer.
        
        Args:
            model_path: Path to the model/tokenizer
            host: Server host address
            port: Server port number
        """
        super().__init__(model_path)
        self.host = host
        self.port = port
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        
    def _send_receive(self, message: Message, retries: int = 3) -> Message:
        """Send message to server and receive response."""
        for attempt in range(retries):
            try:
                self.socket.send_string(message.to_json())
                response_json = self.socket.recv_string()
                return Message.from_json(response_json)
            except zmq.error.ZMQError as e:
                if attempt == retries - 1:
                    raise ConnectionError(f"Failed to communicate with server: {e}")
                logger.warning(f"Connection attempt {attempt + 1} failed, retrying...")
                
    def encode(self,
              text: str,
              add_bos: bool = True,
              add_special_tokens: bool = True,
              **kwargs) -> List[int]:
        """Encode text to token IDs using remote server.
        
        Args:
            text: Text to encode
            add_bos: Whether to add BOS token
            add_special_tokens: Whether to add special tokens
            **kwargs: Additional arguments passed to the server
            
        Returns:
            List of token IDs
        """
        message = Message(
            "encode",
            {
                "text": text,
                "add_bos": add_bos,
                "add_special_tokens": add_special_tokens,
                **kwargs
            }
        )
        response = self._send_receive(message)
        if response.method == "error":
            raise RuntimeError(response.params["error"])
        return response.params["token_ids"]
        
    def decode(self,
              token_ids: Sequence[int],
              offset: Optional[int] = None,
              skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text using remote server.
        
        Args:
            token_ids: Token IDs to decode
            offset: Offset for incremental decoding
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        message = Message(
            "decode",
            {
                "token_ids": list(token_ids),
                "offset": offset,
                "skip_special_tokens": skip_special_tokens
            }
        )
        response = self._send_receive(message)
        if response.method == "error":
            raise RuntimeError(response.params["error"])
        return response.params["text"]
        
    def detokenize_incrementally(self,
                               all_input_ids: Sequence[int],
                               state: DetokenizeState,
                               skip_special_tokens: bool = True,
                               spaces_between_special_tokens: bool = True):
        """Incrementally detokenize the input IDs using remote server.
        
        Args:
            all_input_ids: Token IDs to decode
            state: Current detokenization state
            skip_special_tokens: Whether to skip special tokens
            spaces_between_special_tokens: Whether to add spaces between special tokens
            
        Returns:
            Tuple of (decoded text, new state)
        """
        message = Message(
            "detokenize_incrementally",
            {
                "token_ids": list(all_input_ids),
                "state": {
                    "ids_offset": state.ids_offset,
                    "prev_tokens": state.prev_tokens,
                    "prefix_offset": state.prefix_offset,
                    "read_offset": state.read_offset
                },
                "skip_special_tokens": skip_special_tokens,
                "spaces_between_special_tokens": spaces_between_special_tokens
            }
        )
        response = self._send_receive(message)
        if response.method == "error":
            raise RuntimeError(response.params["error"])
            
        new_state = DetokenizeState(
            ids_offset=response.params["state"]["ids_offset"],
            prev_tokens=response.params["state"]["prev_tokens"],
            prefix_offset=response.params["state"]["prefix_offset"],
            read_offset=response.params["state"]["read_offset"]
        )
        return response.params["text"], new_state