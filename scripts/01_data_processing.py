import argparse
import yaml
from loguru import logger
from components.config import ProjectConfig
from components.data_processor import DataProcessor, generate_synthetic_data
from components.data_reader import DataReader
from components.data_writer import DataWriter
from components.data_split import DataSplitter

# Parse command-line arguments
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
df = data_reader.read_spark_csv(as_pandas=True)
logger.info("Data loaded.")

# Generate synthetic data if needed
df = generate_synthetic_data(df, num_rows=100)
df["avg_price_per_room"] = 1.0 * df["avg_price_per_room"]
logger.info("Synthetic data generated.")

# Initialize Data Processor
data_processor = DataProcessor(df, config)

# Pre Process the data
df = data_processor.pre_processing()
logger.info("Data pre-processed.")


# Initialize Data Splitter
data_splitter = DataSplitter(config, df)

# Split the data
train, test = data_splitter.random_split()
logger.info(f"Training set shape: {train.shape}")
logger.info(f"Test set shape: {test.shape}")

# Save train and test datasets
data_writer = DataWriter(config)
data_writer.save_to_catalog(train, "train_data", "append")
data_writer.save_to_catalog(test, "test_data", "append")
logger.info("Data saved to catalog.")
