from typing import List

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    input_data: str
    test_size: float
    random_state: int
    id_columns: List[str]
    num_features: List[str]
    cat_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    experiment_name: str
    parameters: dict
    endpoint_name: str

    @classmethod
    def from_yaml(cls, config_path: str, env: str = None):
        """Load configuration from a YAML file, adapting based on environment settings."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if env is not None:
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]
            config_dict["pipeline_id"] = config_dict[env]["pipeline_id"]
        else:
            config_dict["catalog_name"] = config_dict["catalog_name"]
            config_dict["schema_name"] = config_dict["schema_name"]
            config_dict["pipeline_id"] = config_dict["pipeline_id"]

        return cls(**config_dict)


class Tags(BaseModel):
    git_sha: str
    branch: str
    job_run_id: str
