
import pytest
import os
import sys
import yaml
from pydantic import ValidationError
from components.config import ProjectConfig



def test_yaml_file_exists():
    config_path = '../project_config.yml'
    assert os.path.exists(config_path), f"{config_path} does not exist."

def test_yaml_file_valid():
    config_path = '../project_config.yml'
    
    try:
        config = ProjectConfig.from_yaml(config_path)
    except ValidationError as e:
        assert False, f"Error loading {config_path}: {e}"

    assert isinstance(config.input_data, str)
    assert isinstance(config.num_features, list)
    assert all(isinstance(feat, str) for feat in config.num_features)
    assert isinstance(config.cat_features, list)
    assert all(isinstance(feat, str) for feat in config.cat_features)
    assert isinstance(config.target, str)
    assert isinstance(config.parameters, dict)



