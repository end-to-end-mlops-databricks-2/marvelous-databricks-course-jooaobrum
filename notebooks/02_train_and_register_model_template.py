# Databricks notebook source
# COMMAND ----------
#!pip install /Volumes/uc_dev/hotel_reservation/samples/packages/hotel_reservations-latest-py3-none-any.whl
# COMMAND ----------
# Import packages
import mlflow
from pyspark.sql import SparkSession
from loguru import logger
from components.config import ProjectConfig, Tags
from components.models.feature_lookup_model import FeatureLookUpModel
# COMMAND ----------
######################## Check configuration ########################
"""
For this model, make sure you have the following parameters in the config file:
- catalog_name
- schema_name
- model_name
- experiment_name
- primary_keys
- num_features
- cat_features
- target
- random_state
"""
# COMMAND ----------
######################## Load configuration ########################
config_path = f"../project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

# Configure MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Create tags
tags_dict = {"git_sha": "abcd12345", "branch": "feat/modeling", "job_run_id": "1234"}
tags = Tags(**tags_dict)

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

logger.info("Configuration loaded:")
logger.info(f"Catalog: {config.catalog_name}")
logger.info(f"Schema: {config.schema_name}")
logger.info(f"Model name: {config.model_name}")
# COMMAND ----------
######################## Initialize Model ########################
"""
Initialize the feature lookup model with configuration and tags
"""
# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")
# COMMAND ----------
######################## Create Feature Table (Optional) ########################
"""
Uncomment this section if you need to create or update the feature table
"""
# fe_model.create_feature_table()
# logger.info("Feature table created/updated.")
# COMMAND ----------
######################## Load and Prepare Data ########################
"""
Load the training data and retrieve features
"""
# Load data from training table
fe_model.load_data()
logger.info("Data loaded.")

# Retrieve features from feature store
fe_model.retrieve_features()
logger.info("Features retrieved.")
# COMMAND ----------
######################## Train Model ########################
"""
Train the model using features from the feature store
"""
# Train the model
fe_model.train()
logger.info("Model training completed.")
# COMMAND ----------
######################## Register Model if Improved ########################
"""
Check if model performance improved and register if it did
"""
if fe_model.model_improved():
    # Register the model in Unity Catalog
    latest_version = fe_model.register_model()
    logger.info(f"Model registered with version {latest_version}")
else:
    logger.info("Model did not improve. Keeping existing model.")
# COMMAND ----------
######################## Score with Latest Model ########################
"""
Make predictions using the latest registered model
"""
# Load test data
test_data = spark.table(f"{config.catalog_name}.{config.schema_name}.test_data")

# Make predictions with the latest model
predictions = fe_model.predict(test_data)
logger.info(f"Generated predictions for {predictions.count()} rows")

# Save predictions if needed
# fe_model.save_predictions(predictions, "model_predictions")