#!/usr/bin/env python
"""Start script for tokenizer server."""

import os
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from lmdeploy.tokenizer_server.server_main import main

if __name__ == '__main__':
    main()
