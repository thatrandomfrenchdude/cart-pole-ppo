import logging


def setup_logging(config, append_mode=False):
    """Set up logging to both console and file."""
    log_file = config['logging']['log_file']
    log_level = getattr(logging, config['logging']['level'])
    log_format = config['logging']['format']
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Choose file mode based on whether we're resuming training
    file_mode = 'a' if append_mode else 'w'
    mode_description = "append mode (resuming training)" if append_mode else "overwrite mode (fresh start)"
    
    # Set up logging to both console and file
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file, mode=file_mode)  # File output
        ],
        force=True
    )
    
    return log_file, mode_description
