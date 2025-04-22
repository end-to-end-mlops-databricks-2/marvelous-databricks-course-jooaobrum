# Databricks notebook source
# COMMAND ----------
# Import packages
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from components.config import ProjectConfig
from components.serving.fe_online_model_serving import FeatureLookupServing

# COMMAND ----------
######################## Check configuration ########################
"""
For this model, make sure you have the following parameters in the config file:
- catalog_name
- schema_name
- model_name
- endpoint_name
- feature_table_name
Modify fe_online_model_serving.py file and edit the functions to be used in the template.
"""
# COMMAND ----------
######################## Load configuration ########################
# Configure hardcoded paths and environment
env = "dev" 
config_path = "../project_config.yml"

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Get model version from upstream task
#model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")
model_version = "latest"  # Placeholder for model version

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=env)

# Extract configuration values
catalog_name = config.catalog_name
schema_name = config.schema_name
feature_table_name = config.feature_table_name
primary_keys = config.primary_keys
model_name = config.model_name
model_path = f"{catalog_name}.{schema_name}"
endpoint_name = config.endpoint_name


logger.info("Configuration loaded:")
logger.info(f"Catalog: {catalog_name}")
logger.info(f"Schema: {schema_name}")
logger.info(f"Model path: {model_path}")
logger.info(f"Endpoint name: {endpoint_name}")
logger.info(f"Model version: {model_version}")

# COMMAND ----------
######################## Initialize Serving Manager ########################
"""
Initialize the feature lookup serving manager with configuration
"""
# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    catalog_name=catalog_name,
    schema_name=schema_name,
    feature_table_name = feature_table_name,
    primary_keys = primary_keys,
    model_name = model_name,
    endpoint_name = endpoint_name
)
logger.info("Feature lookup serving manager initialized.")
# COMMAND ----------
######################## Create Online Table (Optional) ########################
"""
Uncomment this section if you need to create the online feature table
"""
feature_model_server.create_online_table()
# logger.info("Created online table for feature lookup")
# COMMAND ----------
######################## Deploy Model Endpoint ########################
"""
Deploy or update the model serving endpoint with feature lookup. To active inference tables
you need to rerun this cell when endpoint is already deployed.
"""
# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()
logger.info("Started deployment/update of the serving endpoint")
# COMMAND ----------
######################## Verify Endpoint Status (Optional) ########################
"""
Uncomment this section to verify the endpoint status after deployment
"""
endpoint_status = feature_model_server.get_endpoint_status()
logger.info(f"Endpoint status: {endpoint_status}")
# COMMAND ----------
######################## Test Endpoint (Optional) ########################
"""
Uncomment this section to test the endpoint with sample data
"""
sample_data = spark.table(f"{catalog_name}.{schema_name}.test_data").limit(5).drop("update_timestamp_utc")
response = feature_model_server.test_endpoint(sample_data)
logger.info(f"Endpoint test response: {response}")
# COMMAND ----------
