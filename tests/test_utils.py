"""
Tests for the utility functions module.
"""
import logging
import os
import pytest
import tempfile
from utils import setup_logging


class TestUtils:
    
    def test_setup_logging_overwrite_mode(self, sample_config, temp_dir):
        """Test logging setup in overwrite mode."""
        # Set log file path to temp directory
        log_file = os.path.join(temp_dir, 'test.log')
        sample_config['logging']['log_file'] = log_file
        sample_config['logging']['level'] = 'INFO'  # Set to INFO level for this test
        
        # Setup logging in overwrite mode
        returned_log_file, mode_desc = setup_logging(sample_config, append_mode=False)
        
        # Check return values
        assert returned_log_file == log_file
        assert "overwrite mode" in mode_desc
        
        # Check that log file is created
        logger = logging.getLogger()  # Use root logger
        logger.info("Test message")
        
        # Force flush all handlers and close them to ensure data is written
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
            if hasattr(handler, 'close'):
                handler.close()
        
        # Give logging system time to write
        import time
        time.sleep(0.1)
        
        assert os.path.exists(log_file)
        
        # Read log content
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test message" in content
    
    def test_setup_logging_append_mode(self, sample_config, temp_dir):
        """Test logging setup in append mode."""
        log_file = os.path.join(temp_dir, 'test_append.log')
        sample_config['logging']['log_file'] = log_file
        sample_config['logging']['level'] = 'INFO'  # Set to INFO level for this test
        
        # Create initial log content
        with open(log_file, 'w') as f:
            f.write("Initial content\n")
        
        # Setup logging in append mode
        returned_log_file, mode_desc = setup_logging(sample_config, append_mode=True)
        
        # Check return values
        assert returned_log_file == log_file
        assert "append mode" in mode_desc
        
        # Log new message
        logger = logging.getLogger()  # Use root logger
        logger.info("Appended message")
        
        # Force flush all handlers and close them to ensure data is written
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
            if hasattr(handler, 'close'):
                handler.close()
        
        # Give logging system time to write
        import time
        time.sleep(0.1)
        
        # Check that both old and new content exist
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Initial content" in content
            assert "Appended message" in content
    
    def test_logging_level_configuration(self, sample_config, temp_dir):
        """Test that logging level is set correctly."""
        log_file = os.path.join(temp_dir, 'test_level.log')
        sample_config['logging']['log_file'] = log_file
        sample_config['logging']['level'] = 'ERROR'
        
        setup_logging(sample_config, append_mode=False)
        
        logger = logging.getLogger('test_level_logger')
        
        # INFO message should not appear in log
        logger.info("This should not appear")
        
        # ERROR message should appear
        logger.error("This should appear")
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert "This should not appear" not in content
            assert "This should appear" in content
    
    def test_logging_format_configuration(self, sample_config, temp_dir):
        """Test that logging format is applied correctly."""
        log_file = os.path.join(temp_dir, 'test_format.log')
        sample_config['logging']['log_file'] = log_file
        sample_config['logging']['format'] = 'TEST: %(message)s'
        sample_config['logging']['level'] = 'INFO'  # Set to INFO level for this test
        
        setup_logging(sample_config, append_mode=False)
        
        logger = logging.getLogger()  # Use root logger
        logger.info("Format test message")
        
        # Force flush all handlers and close them to ensure data is written
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
            if hasattr(handler, 'close'):
                handler.close()
        
        # Give logging system time to write
        import time
        time.sleep(0.1)
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert "TEST: Format test message" in content
    
    def test_handler_cleanup(self, sample_config, temp_dir):
        """Test that existing handlers are cleaned up."""
        # Add some initial handlers
        logger = logging.getLogger()
        initial_handler_count = len(logger.handlers)
        
        # Add a dummy handler
        dummy_handler = logging.StreamHandler()
        logger.addHandler(dummy_handler)
        
        log_file = os.path.join(temp_dir, 'test_cleanup.log')
        sample_config['logging']['log_file'] = log_file
        
        # Setup logging should clean up handlers
        setup_logging(sample_config, append_mode=False)
        
        # Should have exactly 2 handlers (console and file)
        assert len(logger.handlers) == 2
    
    def test_console_and_file_output(self, sample_config, temp_dir, capsys):
        """Test that logging outputs to both console and file."""
        log_file = os.path.join(temp_dir, 'test_both.log')
        sample_config['logging']['log_file'] = log_file
        sample_config['logging']['level'] = 'INFO'  # Set to INFO level for this test
        
        setup_logging(sample_config, append_mode=False)
        
        logger = logging.getLogger()  # Use root logger
        test_message = "Test both console and file"
        logger.info(test_message)
        
        # Force flush all handlers and close them to ensure data is written
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
            if hasattr(handler, 'close'):
                handler.close()
        
        # Give logging system time to write
        import time
        time.sleep(0.1)
        
        # Check console output
        captured = capsys.readouterr()
        assert test_message in captured.out or test_message in captured.err
        
        # Check file output
        with open(log_file, 'r') as f:
            content = f.read()
            assert test_message in content
    
    def test_different_log_levels(self, sample_config, temp_dir):
        """Test different logging levels."""
        log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in log_levels:
            log_file = os.path.join(temp_dir, f'test_{level.lower()}.log')
            sample_config['logging']['log_file'] = log_file
            sample_config['logging']['level'] = level
            
            setup_logging(sample_config, append_mode=False)
            
            # Should not raise any exceptions
            logger = logging.getLogger(f'test_{level.lower()}_logger')
            logger.info(f"Test message for {level}")
            
            assert os.path.exists(log_file)
    
    def test_log_file_directory_creation(self, sample_config, temp_dir):
        """Test that log file directories are created if they don't exist."""
        # Use nested directory path
        nested_log_file = os.path.join(temp_dir, 'logs', 'nested', 'test.log')
        sample_config['logging']['log_file'] = nested_log_file
        sample_config['logging']['level'] = 'INFO'  # Set to INFO level for this test
        
        # Directories should not exist initially
        assert not os.path.exists(os.path.dirname(nested_log_file))
        
        setup_logging(sample_config, append_mode=False)
        
        logger = logging.getLogger()  # Use root logger
        logger.info("Test nested directory")
        
        # Force flush all handlers and close them to ensure data is written
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
            if hasattr(handler, 'close'):
                handler.close()
        
        # Give logging system time to write
        import time
        time.sleep(0.1)
        
        # Directory and file should be created
        assert os.path.exists(nested_log_file)
        
        with open(nested_log_file, 'r') as f:
            content = f.read()
            assert "Test nested directory" in content
    
    def test_multiple_setup_calls(self, sample_config, temp_dir):
        """Test calling setup_logging multiple times."""
        log_file = os.path.join(temp_dir, 'test_multiple.log')
        sample_config['logging']['log_file'] = log_file
        sample_config['logging']['level'] = 'INFO'  # Set to INFO level for this test
        
        # Call setup multiple times
        setup_logging(sample_config, append_mode=False)
        setup_logging(sample_config, append_mode=False)
        setup_logging(sample_config, append_mode=False)
        
        # Should still work correctly
        logger = logging.getLogger()  # Use root logger
        logger.info("Multiple setup test")
        
        # Force flush all handlers and close them to ensure data is written
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
            if hasattr(handler, 'close'):
                handler.close()
        
        # Give logging system time to write
        import time
        time.sleep(0.1)
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Multiple setup test" in content
        
        # Should not have duplicate handlers
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 2  # Console and file
