import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path="configs"):
        self.config_path = config_path
        self.configs = {}
        
    def load_configs(self) -> Dict[str, Any]:
        """Load all YAML config files from config directory"""
        try:
            for file in os.listdir(self.config_path):
                if file.endswith('.yaml') or file.endswith('.yml'):
                    config_name = os.path.splitext(file)[0]
                    with open(os.path.join(self.config_path, file)) as f:
                        self.configs[config_name] = yaml.safe_load(f)
            return self.configs
        except Exception as e:
            raise RuntimeError(f"Error loading configs: {str(e)}")

    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get specific configuration by name"""
        if not self.configs:
            self.load_configs()
        return self.configs.get(config_name, {})

# Singleton instance
config_loader = ConfigLoader()