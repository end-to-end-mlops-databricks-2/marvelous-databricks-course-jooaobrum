from typing import Any, Dict, List

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    input_data: str
    test_size: float
    random_state: int
    num_features: List[str]
    cat_features: List[str]
    target: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)