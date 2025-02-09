# Databricks notebook source
import mlflow
from pyspark.sql import SparkSession

from components.config import ProjectConfig, Tags
from components.models.custom_model import CustomModel

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "feat/modeling"}
tags = Tags(**tags_dict)

# COMMAND ----------
# Initialize model
model = CustomModel(config=config, tags=tags, spark=spark, code_paths=[])

# COMMAND ----------
model.load_data()

# COMMAND ----------
model.prepare_features()

# COMMAND ----------
model.train_model()

# COMMAND ----------
model.log_model_experiment()
