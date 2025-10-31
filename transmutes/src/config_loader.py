"""Configuration loader for Transmutes AI."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage application configuration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration loader.
        
        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        load_dotenv()  # Load environment variables
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Configuration dictionary.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'llm.temperature')
            default: Default value if key is not found.
            
        Returns:
            Configuration value or default.
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_env(self, key: str, default: str = "") -> str:
        """Get environment variable.
        
        Args:
            key: Environment variable name.
            default: Default value if not found.
            
        Returns:
            Environment variable value or default.
        """
        return os.getenv(key, default)
    
    @property
    def data_directory(self) -> Path:
        """Get the data source directory."""
        return Path(self.get('data.source_directory', './Transmutes_RAG_data'))
    
    @property
    def persist_directory(self) -> Path:
        """Get the vector store persist directory."""
        return Path(self.get('vector_store.persist_directory', './chroma_db'))




