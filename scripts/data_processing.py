# Databricks notebook source

# COMMAND ----------
# Import packages
from components.config import ProjectConfig
from components.data_reader import DataReader
from components.data_processor import DataProcessor
from components.data_writer import DataWriter
import yaml

# COMMAND ----------
# Load configuration
config = ProjectConfig.from_yaml("../project_config.yml")
print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Initialize DataReader
data_reader = DataReader(config)

# Read input data
df = data_reader.read_csv()
# COMMAND ----------
# Initialize Data Processor
data_processor = DataProcessor(df, config)

# Pre Process the data
data_processor.pre_processing()

# COMMAND ----------
# Split the data
train, test  = data_processor.split_data()
print("Training set shape:", train.shape)
print("Test set shape:", test.shape)

# COMMAND ----------
data_writer = DataWriter(config)
data_processor.save_train_test(data_writer)

# COMMAND ----------
