class TokenizerError(Exception):
    """Base exception for tokenizer errors"""
    pass

class ConnectionError(TokenizerError):
    """Connection related errors"""
    pass

class TimeoutError(TokenizerError):
    """Timeout related errors"""
    pass

class ServerError(TokenizerError):
    """Server side errors"""
    pass

class InitializationError(TokenizerError):
    """Initialization related errors"""
    pass