import time

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, AutoCaptureConfigInput

from loguru import logger
import pandas as pd


mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


class FeatureLookupServing:
    def __init__(self, model_name: str, endpoint_name: str, catalog_name: str, schema_name: str, feature_table_name: str, primary_keys: list = None, alias: str = 'latest-model'):
        """
        Initializes the Feature Lookup Serving Manager.
        """
        self.workspace = WorkspaceClient()
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.feature_table_name = feature_table_name
        self.online_table_name = f"{self.feature_table_name}_online"
        self.primary_keys = primary_keys
        self.model_name = model_name
        self.full_model_name = f"{self.catalog_name}.{self.schema_name}.{self.model_name}"
        self.alias = alias
        self.endpoint_name = endpoint_name


    def create_online_table(self):
        """
        Creates an online table for house features.
        """


        full_feature_table_name = f"{self.catalog_name}.{self.schema_name}.{self.feature_table_name}"
        logger.info(f"Creating online table: {self.online_table_name} for feature table: {full_feature_table_name}")
        full_online_feature_table_name = f"{self.catalog_name}.{self.schema_name}.{self.online_table_name}"
        logger.info(f"Full online feature table name: {full_online_feature_table_name}")

        spec = OnlineTableSpec(
            primary_key_columns=self.primary_keys,
            source_table_full_name=full_feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
            perform_full_copy=False,
        )
        self.workspace.online_tables.create(name=full_online_feature_table_name, spec=spec)

    def get_latest_model_version(self):
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.full_model_name, alias=self.alias).version
        logger.info(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self, version: str = "latest", workload_size: str = "Small", scale_to_zero: bool = True
    , enable_inference_tables: bool = True):
        """
        Deploys the model serving endpoint in Databricks.
        :param version: str. Version of the model to deploy
        :param workload_seze: str. Workload size (number of concurrent requests). Default is Small = 4 concurrent requests
        :param scale_to_zero: bool. If True, endpoint scales to 0 when unused.
        """

        inference_table_name = f"{self.catalog_name}.{self.schema_name}.{self.endpoint_name}_inference"

        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        if version == "latest":
            entity_version = self.get_latest_model_version()
        else:
            entity_version = version

        if enable_inference_tables:
            auto_capture_config = AutoCaptureConfigInput(
                    catalog_name=inference_table_name.split('.')[0],
                    schema_name=inference_table_name.split('.')[1],
                    table_name_prefix=inference_table_name.split('.')[2],
                    enabled=True
                )

        served_entities = [
            ServedEntityInput(
                entity_name=self.full_model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            logger.info(f"Creating endpoint: {self.endpoint_name}")
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                    
                ),
                
            )

        else:
            logger.info(f"Updating endpoint: {self.endpoint_name}")
            self.workspace.tables.delete(full_name=f"{self.catalog_name}.{self.schema_name}.{self.endpoint_name}_inference_payload")
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, auto_capture_config=auto_capture_config, served_entities=served_entities)

    def update_online_table(self, pipeline_id: str):
        """
        Triggers a Databricks pipeline update and monitors its state.
        """

        update_response = self.workspace.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)

        while True:
            update_info = self.workspace.pipelines.get_update(
                pipeline_id=pipeline_id, update_id=update_response.update_id
            )
            state = update_info.update.state.value

            if state == "COMPLETED":
                logger.info("Pipeline update completed successfully.")
                break
            elif state in ["FAILED", "CANCELED"]:
                logger.error("Pipeline update failed.")
                raise SystemError("Online table failed to update.")
            elif state == "WAITING_FOR_RESOURCES":
                logger.warning("Pipeline is waiting for resources.")
            else:
                logger.info(f"Pipeline is in {state} state.")

            time.sleep(30)


    def get_endpoint_status(self):
        """
        Get the current status of the model serving endpoint.
        
        Returns:
            dict: A dictionary containing status information about the endpoint
        """
        try:
            endpoint = self.workspace.serving_endpoints.get(name=self.endpoint_name)
            
            # Extract relevant status information
            status_info = {
                "name": endpoint.name,
                "state": str(endpoint.state) if hasattr(endpoint, 'state') else None,
                "creator": endpoint.creator,
                "creation_timestamp": endpoint.creation_timestamp,
                "last_updated_timestamp": endpoint.last_updated_timestamp,
                "config": {
                    "served_entities": [
                        {
                            "name": entity.name,
                            "entity_name": entity.entity_name,
                            "entity_version": entity.entity_version,
                            "workload_size": entity.workload_size,
                            "scale_to_zero_enabled": entity.scale_to_zero_enabled
                        }
                        for entity in endpoint.config.served_entities
                    ] if endpoint.config and endpoint.config.served_entities else []
                }
            }
            
            # Get endpoint metrics if available
            try:
                metrics = self.workspace.serving_endpoints.get_metrics(
                    name=self.endpoint_name,
                    start_timestamp_ms=int((time.time() - 3600) * 1000),  # Last hour
                    end_timestamp_ms=int(time.time() * 1000)
                )
                status_info["metrics"] = {
                    "request_count": metrics.metrics.get("requests", {}).get("count", 0),
                    "avg_response_time": metrics.metrics.get("response_time", {}).get("avg", 0),
                    "p99_response_time": metrics.metrics.get("response_time", {}).get("p99", 0)
                }
            except Exception as e:
                logger.warning(f"Could not retrieve endpoint metrics: {str(e)}")
                status_info["metrics"] = "Not available"
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting endpoint status: {str(e)}")
            return {"error": str(e), "status": "Not available"}
    