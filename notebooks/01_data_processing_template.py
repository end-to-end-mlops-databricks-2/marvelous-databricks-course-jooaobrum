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
from components.data_split import DataSplitter


# COMMAND ----------

########################  Check  configuration ########################
"""
For this template, make sure you have the following parameters in the config file:
- input_data
- test_size
- random_state
- num_features
- cat_features
- target
- catalog_name
- schema_name
"""


# COMMAND ----------
########################  Load configuration ########################
config_path = f"../project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

########################  Load Data ########################
"""
Possibilities: 
- Read from CSV
- Read from table
- Read from sql
- Read from sql file

Check data_reader.py for more details.

"""
# Initialize DataReader
data_reader = DataReader(config)

# Read input data
df = data_reader.read_spark_csv(as_pandas=True)

# COMMAND ----------
########################  Data-Processing ########################
"""
Pre-processing steps:
- Validate columns
- Rename columns
- Convert data types
- Check null values


Possibility:
- Apply custom pre-processing pipeline:
1. Create a class with the method apply_transformations

class CustomPreprocessor:
    def apply_transformations(self, df):
        # Apply transformations
        return df
2. Initialize the class and call the method

Check data_processor.py for more details.
"""

class CustomPreprocessor:
    def __init__(self):
        pass
    def apply_transformations(self, df):
        pass
        return df

# Initialize Data Processor
data_processor = DataProcessor(df, config)

# Pre Process the data
df = data_processor.pre_processing(optional_custom_processor=CustomPreprocessor())


# COMMAND ----------
########################  Data Split ########################
"""
Possibilities: 
- Random split
- Stratified random split

Check data_split.py for more details.

"""

# Initialize Data Splitter
data_splitter = DataSplitter(config, df)

# Split the data
train, test = data_splitter.random_split()
logger.info(f"Training set shape: {train.shape}")
logger.info(f"Test set shape: {test.shape}")

# COMMAND ----------
########################  Save train and test ########################
"""
Possibilities:
- Save table to catalog
- Create or update table in catalog
- Truncate table
- Drop table

Check data_writer.py for more details.
"""

data_writer = DataWriter(config)
data_writer.save_to_catalog(train, "train_data", "overwrite")
data_writer.save_to_catalog(test, "test_data", "overwrite")
