from typing import List

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    input_data: str
    test_size: float
    random_state: int
    num_features: List[str]
    cat_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    experiment_name: str
    parameters: dict

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

class Tags(BaseModel):
    git_sha: str
    branch: str