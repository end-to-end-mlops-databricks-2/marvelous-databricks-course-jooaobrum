import argparse

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

from components.config import ProjectConfig
from components.monitoring import create_or_refresh_monitoring

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
config_path = f"{root_path}/files/project_config.yml"
inference_table = "`hotel-reservation-model-serving-fe_inference_payload`"


# Load configuration
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace, table_inference_payload=inference_table)
