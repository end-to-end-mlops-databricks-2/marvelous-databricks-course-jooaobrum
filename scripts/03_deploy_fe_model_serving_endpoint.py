# Databricks notebook source

from loguru import logger

from components.config import ProjectConfig
from components.serving.fe_model_serving import FeatureLookupServing

# COMMAND ----------
# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "hotel-reservation-model-serving-fe"

# COMMAND ----------
# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    model_name = f"{catalog_name}.{schema_name}.hotel_reservation_model_fe",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{schema_name}.hotel_reservation_features",
)

# COMMAND ----------
# Create the online table for house features
feature_model_server.create_online_table()
logger.info("Created online table")

# COMMAND ----------
# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()
logger.info("Started deployment/update of the serving endpoint")