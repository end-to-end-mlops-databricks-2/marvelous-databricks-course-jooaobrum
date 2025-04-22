import argparse

import mlflow
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from components.config import ProjectConfig, Tags
from components.models.feature_lookup_model import FeatureLookUpModel

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", action="store", default=None, type=str, required=True)
parser.add_argument("--env", action="store", default=None, type=str, required=True)
parser.add_argument("--git-sha", action="store", default=None, type=str, required=False)
parser.add_argument("--job_run_id", action="store", default=None, type=str, required=True)
parser.add_argument("--branch", action="store", default=None, type=str, required=True)
args = parser.parse_args()
root_path = args.root_path

# Load configuration
config_path = f"{root_path}/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

# Configure MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Create tags
tags_dict = {
    "git_sha": args.git_sha if args.git_sha else "abcd12345",
    "branch": args.branch,
    "job_run_id": args.job_run_id,
}
tags = Tags(**tags_dict)

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))
logger.info(f"Catalog: {config.catalog_name}")
logger.info(f"Schema: {config.schema_name}")
logger.info(f"Model name: {config.model_name}")

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")

# Update feature table
fe_model.update_feature_table()
logger.info("Feature table updated.")

# Load data from training table
fe_model.load_data()
logger.info("Data loaded.")

# Retrieve features from feature store
fe_model.retrieve_features()
logger.info("Features retrieved.")

# Train the model
fe_model.train()
logger.info("Model training completed.")

# Register model if improved
if fe_model.model_improved():
    # Register the model in Unity Catalog
    latest_version = fe_model.register_model()
    logger.info(f"Model registered with version {latest_version}")
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)
else:
    logger.info("Model did not improve. Keeping existing model.")
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
