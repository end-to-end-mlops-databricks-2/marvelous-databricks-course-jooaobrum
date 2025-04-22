import argparse
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from components.config import ProjectConfig
from components.serving.fe_online_model_serving import FeatureLookupServing

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", action="store", default=None, type=str, required=True)
parser.add_argument("--env", action="store", default=None, type=str, required=True)
args = parser.parse_args()
root_path = args.root_path

# Load configuration
config_path = f"{root_path}/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Get model version from upstream task
try:
    model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")
except:
    logger.warning("Could not retrieve model version from upstream task. Using 'latest'.")
    model_version = "latest"  # Fallback when not in a workflow

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

# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    catalog_name=catalog_name,
    schema_name=schema_name,
    feature_table_name=feature_table_name,
    primary_keys=primary_keys,
    model_name=model_name,
    endpoint_name=endpoint_name
)
logger.info("Feature lookup serving manager initialized.")

# Create the online table for feature lookup
# feature_model_server.create_online_table()
# logger.info("Created online table for feature lookup")

# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()
logger.info("Started deployment/update of the serving endpoint")

# Verify endpoint status
endpoint_status = feature_model_server.get_endpoint_status()
logger.info(f"Endpoint status: {endpoint_status}")

