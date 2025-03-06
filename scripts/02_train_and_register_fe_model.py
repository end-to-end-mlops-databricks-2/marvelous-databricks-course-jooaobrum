import argparse

import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from components.config import ProjectConfig, Tags
from components.models.feature_lookup_model import FeatureLookUpModel

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", action="store", default=None, type=str, required=True)

parser.add_argument("--env", action="store", default=None, type=str, required=True)

parser.add_argument("--git-sha", action="store", default=None, type=str, required=False)

parser.add_argument("--job_run_id", action="store", default=None, type=str, required=True)

parser.add_argument("--branch", action="store", default=None, type=str, required=True)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

spark = SparkSession.builder.getOrCreate()

tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)


# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")

# fe_model.create_feature_table()

fe_model.update_feature_table()
logger.info("Feature table updated.")

fe_model.load_data()
logger.info("Data loaded.")

fe_model.feature_engineering()

fe_model.train()
logger.info("Model training completed.")

# Evaluate model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_data").limit(100)

model_improved = fe_model.model_improved(test_set)
logger.info("Model evaluation completed, model improved: ", model_improved)

if model_improved:
    # Register the model
    latest_version = fe_model.register_model()
    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)

else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
