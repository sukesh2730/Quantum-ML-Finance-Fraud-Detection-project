"""Configuration management for quantum fraud detector.

This module provides configuration classes and loaders for managing
system parameters across different environments.
"""

import json
import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class QuantumConfig:
    """Quantum circuit configuration."""
    n_qubits: int = 4
    n_layers: int = 2
    device: str = "default.qubit"  # or "lightning.qubit", hardware backend


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 50
    learning_rate: float = 0.01
    batch_size: int = 32
    optimizer: str = "adam"


@dataclass
class SageMakerConfig:
    """AWS SageMaker deployment configuration."""
    region: str = "us-east-1"
    instance_type: str = "ml.m5.large"
    s3_bucket: str = ""
    role_arn: str = ""
    model_name: str = "quantum-fraud-detector"
    endpoint_name: str = "quantum-fraud-endpoint"


@dataclass
class PreprocessingConfig:
    """Data preprocessing configuration."""
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    handle_missing: str = "mean"  # or "median", "drop"


class ConfigLoader:
    """Load configuration from file or environment variables."""
    
    @staticmethod
    def load_from_file(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file.
        
        Supports both YAML and JSON formats. File format is determined
        by the file extension (.yaml, .yml, or .json).
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration data
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If file format is unsupported or file is malformed
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file format from extension
        suffix = path.suffix.lower()
        
        try:
            with open(path, 'r') as f:
                if suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif suffix == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(
                        f"Unsupported configuration file format: {suffix}. "
                        "Supported formats: .yaml, .yml, .json"
                    )
            
            # Handle empty files
            if config is None:
                config = {}
            
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file: {e}")
    
    @staticmethod
    def load_from_env(prefix: str = "QFRAUD_") -> Dict[str, Any]:
        """Load configuration from environment variables.
        
        Reads environment variables with the specified prefix and converts
        them to a nested configuration dictionary. Variable names are converted
        to lowercase and split by underscore to create nested structure.
        
        Example:
            QFRAUD_QUANTUM_N_QUBITS=4 -> {"quantum": {"n_qubits": 4}}
            QFRAUD_TRAINING_EPOCHS=100 -> {"training": {"epochs": 100}}
        
        Args:
            prefix: Environment variable prefix (default: "QFRAUD_")
            
        Returns:
            Dictionary containing configuration data from environment
        """
        config: Dict[str, Any] = {}
        
        for key, value in os.environ.items():
            # Check if variable starts with prefix
            if not key.startswith(prefix):
                continue
            
            # Remove prefix and convert to lowercase
            config_key = key[len(prefix):].lower()
            
            # Split by underscore to create nested structure
            parts = config_key.split('_')
            
            if len(parts) < 2:
                # Skip variables without section (e.g., QFRAUD_DEBUG)
                continue
            
            # First part is the section (quantum, training, etc.)
            section = parts[0]
            # Remaining parts form the parameter name
            param = '_'.join(parts[1:])
            
            # Initialize section if needed
            if section not in config:
                config[section] = {}
            
            # Parse value - attempt to convert to appropriate type
            parsed_value = ConfigLoader._parse_env_value(value)
            config[section][param] = parsed_value
        
        return config
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable value to appropriate type.
        
        Attempts to convert string values to int, float, bool, or list.
        Returns string if no conversion is possible.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Parsed value with appropriate type
        """
        # Handle boolean values
        if value.lower() in ['true', 'yes', '1']:
            return True
        if value.lower() in ['false', 'no', '0']:
            return False
        
        # Handle numeric values
        try:
            # Try integer first
            if '.' not in value:
                return int(value)
            # Then float
            return float(value)
        except ValueError:
            pass
        
        # Handle comma-separated lists
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # Return as string
        return value
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries with priority.
        
        Later configurations override earlier ones. Nested dictionaries
        are merged recursively, allowing fine-grained overrides.
        
        Example:
            config1 = {"quantum": {"n_qubits": 4, "n_layers": 2}}
            config2 = {"quantum": {"n_qubits": 8}}
            merge_configs(config1, config2)
            -> {"quantum": {"n_qubits": 8, "n_layers": 2}}
        
        Args:
            *configs: Variable number of configuration dictionaries
            
        Returns:
            Merged configuration dictionary
        """
        if not configs:
            return {}
        
        # Start with first config
        result = {}
        
        for config in configs:
            result = ConfigLoader._deep_merge(result, config)
        
        return result
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries.
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                # Override value
                result[key] = value
        
        return result
