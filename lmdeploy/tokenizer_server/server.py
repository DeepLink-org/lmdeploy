"""Tokenizer server implementation."""
import json
import logging
import multiprocessing as mp
from typing import Any, Dict, List, Optional

import zmq

from lmdeploy.tokenizer import HuggingFaceTokenizer, DetokenizeState
from .messages import Message

logger = logging.getLogger(__name__)

def worker_process(model_path: str, host: str, port: int):
    """Worker process function."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(f"ipc:///tmp/tokenizer_server_{port}")
    
    try:
        # Initialize tokenizer in each worker process
        tokenizer = HuggingFaceTokenizer(model_path)
        logger.info(f"Worker initialized with model from {model_path}")
        
        while True:
            try:
                # Receive and parse message
                message_json = socket.recv_string()
                message = Message.from_json(message_json)
                
                # Handle different methods
                if message.method == "encode":
                    # Use HuggingFaceTokenizer encode
                    token_ids = tokenizer.encode(
                        message.params["text"],
                        add_bos=message.params.get("add_bos", True),
                        add_special_tokens=message.params.get("add_special_tokens", True)
                    )
                    response = Message.encode_response([int(x) for x in token_ids])
                    
                elif message.method == "decode":
                    # Use HuggingFaceTokenizer decode
                    text = tokenizer.decode(
                        message.params["token_ids"],
                        offset=message.params.get("offset"),
                        skip_special_tokens=message.params.get("skip_special_tokens", True)
                    )
                    response = Message.decode_response(text)
                    
                elif message.method == "detokenize_incrementally":
                    # Use HuggingFaceTokenizer detokenize_incrementally
                    try:
                        state = DetokenizeState(
                            ids_offset=message.params["state"]["ids_offset"],
                            prev_tokens=message.params["state"]["prev_tokens"],
                            prefix_offset=message.params["state"]["prefix_offset"],
                            read_offset=message.params["state"]["read_offset"]
                        )
                        text, new_state = tokenizer.detokenize_incrementally(
                            message.params["token_ids"],
                            state,
                            skip_special_tokens=message.params.get("skip_special_tokens", True),
                            spaces_between_special_tokens=message.params.get("spaces_between_special_tokens", True)
                        )
                        response = Message(
                            "detokenize_incrementally_response",
                            {
                                "text": text,
                                "state": {
                                    "ids_offset": new_state.ids_offset,
                                    "prev_tokens": new_state.prev_tokens,
                                    "prefix_offset": new_state.prefix_offset,
                                    "read_offset": new_state.read_offset
                                }
                            }
                        )
                    except Exception as e:
                        logger.error(f"Detokenize incrementally error: {e}")
                        raise RuntimeError(f"Detokenize incrementally failed: {e}")
                    
                else:
                    response = Message.error_response(f"Unknown method: {message.method}")
                
                # Send response
                socket.send_string(response.to_json())
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                error_response = Message.error_response(str(e))
                socket.send_string(error_response.to_json())
                
    except Exception as e:
        logger.error(f"Worker failed: {e}")
    finally:
        socket.close()
        context.term()

class TokenizerServer:
    """Tokenizer server using ZMQ for IPC."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, num_workers: int = 4):
        """Initialize server."""
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.workers: List[mp.Process] = []
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.model_path: Optional[str] = None

    def initialize(self, model_path: str):
        """Initialize the server with model."""
        self.model_path = model_path
        logger.info(f"Initializing server with model from {model_path}")

    def start(self):
        """Start the server."""
        if not self.model_path:
            raise ValueError("Server not initialized with model_path")

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://{self.host}:{self.port}")

        # Create IPC socket for workers
        worker_socket = self.context.socket(zmq.DEALER)
        worker_socket.bind(f"ipc:///tmp/tokenizer_server_{self.port}")

        # Start worker processes
        for _ in range(self.num_workers):
            worker = mp.Process(
                target=worker_process,
                args=(self.model_path, self.host, self.port)
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"Server started with {self.num_workers} workers")

        try:
            zmq.proxy(self.socket, worker_socket)
        except KeyboardInterrupt:
            logger.info("Server stopping...")
        finally:
            self.stop()

    def stop(self):
        """Stop the server."""
        logger.info("Stopping server...")
        
        # Stop workers
        for worker in self.workers:
            worker.terminate()
            worker.join()
        self.workers.clear()

        # Close sockets
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()

        logger.info("Server stopped")