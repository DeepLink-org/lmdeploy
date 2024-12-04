import os
import logging
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .constants import *
from .exceptions import *

logger = logging.getLogger('lmdeploy')

@dataclass
class Message:
    """Message for client-server communication"""
    type: str
    id: str
    model_path: Optional[str]
    method: str
    args: tuple
    kwargs: Dict[str, Any]

    def serialize(self) -> bytes:
        """Serialize message to bytes"""
        return pickle.dumps(self.__dict__)

    @classmethod
    def deserialize(cls, data: bytes) -> 'Message':
        """Deserialize message from bytes"""
        return cls(**pickle.loads(data))

def get_server_config():
    """Get server configuration from environment variables"""
    return {
        'host': os.environ.get(ENV_HOST, DEFAULT_HOST),
        'port': int(os.environ.get(ENV_PORT, DEFAULT_PORT)),
    }

def setup_logger():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )