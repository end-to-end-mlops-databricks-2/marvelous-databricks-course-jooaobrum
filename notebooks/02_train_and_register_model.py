import mlflow
from pyspark.sql import SparkSession

from components.config import ProjectConfig, Tags
from components.models.custom_model import CustomModel

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "feat/modeling"}
tags = Tags(**tags_dict)

# Initialize model
model = CustomModel(config=config, tags=tags, spark=spark, code_paths=[])

# Load Data
model.load_data()

# Prepare Features
model.prepare_features()

# Train Model
model.train_model()

# Log Experiment
model.log_model_experiment()
