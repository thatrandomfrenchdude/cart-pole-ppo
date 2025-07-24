import logging
import os
import sys


class SafeFormatter(logging.Formatter):
    """Custom formatter that safely handles Unicode characters for Windows console"""
    
    def format(self, record):
        try:
            return super().format(record)
        except UnicodeEncodeError:
            # If encoding fails, replace problematic characters
            formatted = super().format(record)
            # Replace common emoji characters with text equivalents
            replacements = {
                '🎬': '[EXAMPLE]',
                '🚀': '[SUCCESS]',
                '✅': '[OK]',
                '⚠️': '[WARNING]',
                '❌': '[ERROR]',
                '📦': '[LOADING]',
                '🏃': '[BENCHMARK]',
                '📊': '[RESULTS]',
                '🏆': '[FASTEST]',
                '💡': '[TIP]',
                '🔍': '[DEBUG]',
                '🧪': '[TEST]',
                '🎉': '[COMPLETE]'
            }
            for emoji, replacement in replacements.items():
                formatted = formatted.replace(emoji, replacement)
            return formatted


def setup_logging(config, append_mode=False):
    """Set up logging to both console and file with proper UTF-8 encoding."""
    # Try to set console to UTF-8 mode on Windows
    try:
        if sys.platform.startswith('win'):
            # Enable UTF-8 mode for Windows console
            os.system('chcp 65001 > nul 2>&1')
    except:
        pass  # Ignore if this fails
    
    log_file = config['logging']['log_file']
    log_level = getattr(logging, config['logging']['level'])
    log_format = config['logging']['format']
    
    # Create directory for log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Clear any existing handlers on the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    # Choose file mode based on whether we're resuming training
    file_mode = 'a' if append_mode else 'w'
    mode_description = "append mode (resuming training)" if append_mode else "overwrite mode (fresh start)"
    
    # Create formatters
    safe_formatter = SafeFormatter(log_format)  # For console (safe encoding)
    utf8_formatter = logging.Formatter(log_format)  # For file (UTF-8)
    
    # Create console handler with UTF-8 encoding support
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(safe_formatter)  # Use safe formatter for console
    
    # Create file handler with explicit UTF-8 encoding
    file_handler = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(utf8_formatter)  # Use UTF-8 formatter for file
    
    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return log_file, mode_description
