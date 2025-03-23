# Databricks notebook source
# COMMAND ----------
#!pip install /Volumes/uc_dev/hotel_reservation/samples/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------
import mlflow
from pyspark.sql import SparkSession

from components.config import ProjectConfig, Tags
from components.models.custom_model import CustomModel

# COMMAND ----------
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
# Load Data
model.load_data()

# Prepare Features
model.prepare_features()

# Train Model
model.train_model()

# Log Experiment
model.log_model_experiment()
