# Import packages
import argparse

import yaml
from loguru import logger

from components.config import ProjectConfig
from components.data_processor import DataProcessor, generate_synthetic_data
from components.data_reader import DataReader
from components.data_writer import DataWriter

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path

# Load configuration
config_path = f"{root_path}/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Initialize DataReader
data_reader = DataReader(config)

# Read input data
df = data_reader.read_csv()

synthetic_df = generate_synthetic_data(df, num_rows=100)
synthetic_df["avg_price_per_room"] = 1.0 * synthetic_df["avg_price_per_room"]

logger.info("Synthetic data generated.")

# Initialize Data Processor
data_processor = DataProcessor(synthetic_df, config)

# Pre Process the data
data_processor.pre_processing()

# Split the data
train, test = data_processor.split_data()
logger.info(f"Training set shape: {train.shape}")
logger.info(f"Test set shape: {test.shape}")

data_writer = DataWriter(config)
data_processor.save_train_test(data_writer)
