"""Main script to start the tokenizer server."""

import argparse
import logging
import os
import sys
from .server import TokenizerServer

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main function to start the server."""
    parser = argparse.ArgumentParser(description='Start the tokenizer server')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host address to bind to')
    parser.add_argument('--port', type=int, default=5555,
                       help='Port to listen on')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker processes')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the model/tokenizer')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(level=log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting server with configuration: {args}")
    
    try:
        # Create and start server
        server = TokenizerServer(
            host=args.host,
            port=args.port,
            num_workers=args.workers
        )
        
        # Initialize with model
        server.initialize(args.model_path)
        
        # Start server
        server.start()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 