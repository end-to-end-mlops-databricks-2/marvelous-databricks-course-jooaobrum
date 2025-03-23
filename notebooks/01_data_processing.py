# Databricks notebook source
# COMMAND ----------
#!pip install /Volumes/uc_dev/hotel_reservation/samples/packages/hotel_reservations-latest-py3-none-any.whl


# COMMAND ----------
# Import packages
import argparse

import yaml
from loguru import logger

from components.config import ProjectConfig
from components.data_processor import DataProcessor, generate_synthetic_data
from components.data_reader import DataReader
from components.data_writer import DataWriter

# COMMAND ----------
# Load configuration
config_path = f"../project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Initialize DataReader
data_reader = DataReader(config)

# Read input data
df = data_reader.read_csv(as_pandas=True)

synthetic_df = generate_synthetic_data(df, num_rows=100)
synthetic_df["avg_price_per_room"] = 1.0 * synthetic_df["avg_price_per_room"]
logger.info("Synthetic data generated.")

# COMMAND ----------
# Initialize Data Processor
data_processor = DataProcessor(synthetic_df, config)

# Pre Process the data
data_processor.pre_processing()

# Split the data
train, test = data_processor.split_data()
logger.info(f"Training set shape: {train.shape}")
logger.info(f"Test set shape: {test.shape}")

# COMMAND ----------
data_writer = DataWriter(config)
#data_writer.truncate_table("train_data")
#data_writer.truncate_table("test_data")
#data_writer.save_to_catalog(train, "train_data", "overwrite")
#data_writer.save_to_catalog(test, "test_data", "overwrite")
data_writer.create_or_update(train, "train_data")
data_writer.create_or_update(test, "test_data")

# COMMAND ----------