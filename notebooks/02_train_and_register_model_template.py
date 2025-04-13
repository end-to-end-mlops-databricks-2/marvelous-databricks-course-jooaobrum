# Databricks notebook source
# COMMAND ----------
#!pip install /Volumes/uc_dev/hotel_reservation/samples/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------
import mlflow
from pyspark.sql import SparkSession
from loguru import logger
from components.config import ProjectConfig, Tags
from components.models.feature_lookup_model import FeatureLookUpModel

# COMMAND ----------
# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "feat/modeling", "job_run_id": "1234"}
tags = Tags(**tags_dict)

# COMMAND ----------
# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")

# COMMAND ----------
#fe_model.create_feature_table()
#logger.info("Feature table updated.")

# COMMAND ----------
# Load data
fe_model.load_data()
logger.info("Data loaded.")

# Retrieve Features
fe_model.retrieve_features()
logger.info("Features retrieved.")

# Train Model
fe_model.train()
logger.info("Model training completed.")

# Check if model improved and register
if fe_model.model_improved():
    # Register the model
    latest_version = fe_model.register_model()
