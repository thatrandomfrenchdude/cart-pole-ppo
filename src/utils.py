import logging
import os


def setup_logging(config, append_mode=False):
    """Set up logging to both console and file."""
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
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode=file_mode)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return log_file, mode_description
