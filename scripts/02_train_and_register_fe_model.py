# Databricks notebook source
import mlflow
from pyspark.sql import SparkSession

from components.config import ProjectConfig, Tags
from components.models.feature_lookup_model import FeatureLookUpModel

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
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
fe_model.create_feature_table()

# COMMAND ----------
fe_model.load_data()

# COMMAND ----------
fe_model.feature_engineering()

# COMMAND ----------
fe_model.train()

# COMMAND ----------
fe_model.register_model()
