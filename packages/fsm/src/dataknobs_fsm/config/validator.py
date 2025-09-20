"""Configuration validation utilities."""

from typing import Dict, Any, List
from pathlib import Path

from .schema import validate_config
from .loader import ConfigLoader


class ConfigValidator:
    """Configuration validation utility."""
    
    def __init__(self):
        self.loader = ConfigLoader()
    
    def validate_file(self, file_path: str) -> List[str]:
        """Validate configuration file.

        Args:
            file_path: Path to configuration file

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            # Try to load and validate the config - loading validates structure
            self.loader.load_from_file(Path(file_path))
            # If we get here, the config is valid
            return errors

        except Exception as e:
            errors.append(str(e))
            return errors
    
    def validate_dict(self, config_dict: Dict[str, Any]) -> List[str]:
        """Validate configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            validate_config(config_dict)
            return errors
            
        except Exception as e:
            errors.append(str(e))
            return errors
