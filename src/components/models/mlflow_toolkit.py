from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient
from loguru import logger
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient


class MLflowToolkit:
    """
    A toolkit for simplified MLflow workflow management.

    This class provides methods for experiment tracking, model registration,
    and model loading with minimal boilerplate code.
    """

    def __init__(
        self,
        experiment_name: str,
        model_name: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize the MLflow toolkit.

        Parameters
        ----------
        experiment_name : str
            Name of the MLflow experiment
        model_name : str, optional
            Base name for the registered model
        catalog_name : str, optional
            Catalog name for UC model registry (e.g., 'main')
        schema_name : str, optional
            Schema name for UC model registry (e.g., 'ml_models')
        tags : Dict[str, str], optional
            Tags to attach to runs and models
        tracking_uri : str, optional
            MLflow tracking server URI
        """

        self.experiment_name = experiment_name
        self.model_name = model_name
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.tags = tags or {}

        # Add default tags if not provided
        if "created_by" not in self.tags:
            self.tags["created_by"] = "unknown"

        if "creation_timestamp" not in self.tags:
            self.tags["creation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Initialize client
        self.client = MlflowClient()

        # Current run information
        self.current_run_id = None
        self.feature_store_client = FeatureEngineeringClient()

        # Set up experiment
        self._setup_experiment()

    def _setup_experiment(self):
        """Set up or get the experiment."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            logger.info(f"Creating new experiment: {self.experiment_name}")
            mlflow.create_experiment(self.experiment_name)
        else:
            logger.info(f"Using existing experiment: {self.experiment_name} (ID: {experiment.experiment_id})")

        mlflow.set_experiment(self.experiment_name)

    def _get_full_model_name(self) -> str:
        """Get the full model name including catalog and schema if provided."""
        if self.model_name is None:
            raise ValueError("model_name must be set to register or load models")

        if self.catalog_name and self.schema_name:
            return f"{self.catalog_name}.{self.schema_name}.{self.model_name}"
        return self.model_name

    def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> str:
        """
        Start a new MLflow run.

        Parameters
        ----------
        run_name : str, optional
            Name for the run
        nested : bool, default=False
            Whether this is a nested run

        Returns
        -------
        str
            The run ID
        """
        run = mlflow.start_run(run_name=run_name, nested=nested, tags=self.tags)
        self.current_run_id = run.info.run_id

        logger.info(f"Started MLflow run: {self.current_run_id}")
        return self.current_run_id

    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        logger.info(f"Ended MLflow run: {self.current_run_id}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to the current run.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to log
        """
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log metrics to the current run.

        Parameters
        ----------
        metrics : Dict[str, float]
            Metrics to log
        step : int, optional
            Step for the metrics
        """
        mlflow.log_metrics(metrics)

        # Log metrics summary
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        flavor: Any = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        training_set: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Log a model to the current run.

        Parameters
        ----------
        model : Any
            Model to log
        artifact_path : str
            Path within the run to store the model
        flavor : Any, optional
            MLflow flavor module to use
        signature : ModelSignature, optional
            Model signature
        input_example : Any, optional
            Input example for the model
        training_set : Any, optional
            Training set for feature store integration
        feature_names : List[str], optional
            Feature names for the model
        **kwargs :
            Additional arguments to pass to the flavor's log_model

        Returns
        -------
        str
            Model URI
        """
        if self.current_run_id is None:
            raise ValueError("No active run. Call start_run() before logging a model.")

        # Infer signature if not provided
        if signature is None and hasattr(model, "predict"):
            # Try to infer a sample output
            try:
                if isinstance(input_example, pd.DataFrame) and len(input_example) > 0:
                    model_output = model.predict(input_example.iloc[:1])
                    signature = infer_signature(input_example.iloc[:1], model_output)
            except Exception as e:
                logger.warning(f"Could not infer signature: {e}")

        # Check if we should use feature store client
        if self.feature_store_client is not None and training_set is not None:
            print("Using feature store client for logging model")
            return self.feature_store_client.log_model(
                model=model,
                flavor=flavor or mlflow.sklearn,
                artifact_path=artifact_path,
                training_set=training_set,
                signature=signature,
                **kwargs,
            )
        else:
            # Use standard MLflow logging
            print("Using standard MLflow logging")
            logger.info(f"Logging model at {artifact_path}")

            # Use appropriate flavor based on model type if not specified
            if flavor is None:
                logger.warning("Flavor not detected. Defaulting to mlflow.pyfunc.")
                flavor = mlflow.pyfunc

            # Generate signature if not provided but input_example is
            if signature is None and input_example is not None:
                try:
                    if hasattr(model, "predict"):
                        model_output = model.predict(input_example)
                        signature = infer_signature(input_example, model_output)
                except Exception as e:
                    logger.warning(f"Could not infer signature: {e}")

            return flavor.log_model(
                model, artifact_path=artifact_path, signature=signature, input_example=input_example, **kwargs
            )

    def register_model(
        self,
        run_id: Optional[str] = None,
        artifact_path: str = "model",
        model_name: Optional[str] = None,
        alias: str = "latest-model",
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Register a model in the MLflow Model Registry.

        Parameters
        ----------
        run_id : str, optional
            MLflow run ID where the model is logged (defaults to current run)
        artifact_path : str, default="model"
            Path within the run where the model is stored
        model_name : str, optional
            Name to register the model under (overrides the one provided in __init__)
        alias : str, default="latest-model"
            Alias to assign to the new model version
        tags : Dict[str, str], optional
            Tags to attach to the registered model

        Returns
        -------
        Dict[str, Any]
            Information about the registered model
        """
        run_id = run_id or self.current_run_id
        if run_id is None:
            raise ValueError("No run ID provided or no active run")

        # Get model name
        full_model_name = model_name or self._get_full_model_name()

        # Combine with instance tags
        combined_tags = {**self.tags}
        if tags:
            combined_tags.update(tags)

        logger.info(f"Registering model {full_model_name} from run {run_id}")

        model_uri = f"runs:/{run_id}/{artifact_path}"

        # Register the model
        registered_model = mlflow.register_model(model_uri=model_uri, name=full_model_name, tags=combined_tags)

        # Get the version
        latest_version = registered_model.version

        # Set alias if provided
        if alias:
            self.client.set_registered_model_alias(name=full_model_name, alias=alias, version=latest_version)
            logger.info(f"Set alias '{alias}' to version {latest_version}")

        logger.info(f"Model registered successfully as {full_model_name} version {latest_version}")

        # Return information about the registered model
        return {"name": full_model_name, "version": latest_version, "alias": alias, "run_id": run_id}

    def load_model(
        self,
        model_uri: Optional[str] = None,
        version: Optional[str] = None,
        alias: str = "latest-model",
        as_pyfunc: bool = False,
    ) -> Any:
        """
        Load a registered model.

        Parameters
        ----------
        model_uri : str, optional
            Model URI (overrides the model name provided in __init__)
        version : str, optional
            Model version to load
        alias : str, default="latest-model"
            Model alias to load
        as_pyfunc : bool, default=False
            Whether to load as a PyFunc model

        Returns
        -------
        Any
            Loaded model
        """
        # Build model URI if not provided
        if model_uri is None:
            full_model_name = self._get_full_model_name()

            if version is not None:
                model_uri = f"models:/{full_model_name}/{version}"
            else:
                model_uri = f"models:/{full_model_name}@{alias}"

        logger.info(f"Loading model from {model_uri}")

        # Load the model
        if as_pyfunc:
            return mlflow.pyfunc.load_model(model_uri)
        else:
            return mlflow.sklearn.load_model(model_uri)

    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        model_uri: Optional[str] = None,
        version: Optional[str] = None,
        alias: str = "latest-model",
    ) -> Any:
        """
        Load a model and make predictions.

        Parameters
        ----------
        data : DataFrame or array
            Input data for prediction
        model_uri : str, optional
            Model URI (overrides the model name provided in __init__)
        version : str, optional
            Model version to load
        alias : str, default="latest-model"
            Model alias to load

        Returns
        -------
        Any
            Predictions
        """
        # Build model URI if not provided
        if model_uri is None:
            full_model_name = self._get_full_model_name()

            if version is not None:
                model_uri = f"models:/{full_model_name}/{version}"
            else:
                model_uri = f"models:/{full_model_name}@{alias}"

        logger.info(f"Making predictions with model from {model_uri}")

        # Load the model using pyfunc for consistent prediction API
        model = mlflow.pyfunc.load_model(model_uri)

        # Make predictions
        predictions = model.predict(data)

        return predictions

    def set_feature_store_client(self, feature_store_client: Any):
        """
        Set a feature store client for enhanced model logging.

        Parameters
        ----------
        feature_store_client : Any
            Feature store client instance
        """
        self.feature_store_client = feature_store_client
        logger.info(f"Feature store client set: {type(feature_store_client).__name__}")

    def batch_score_with_feature_store(
        self,
        data: pd.DataFrame,
        model_uri: Optional[str] = None,
        version: Optional[str] = None,
        alias: str = "latest-model",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Score a batch using the feature store client.

        Parameters
        ----------
        data : DataFrame
            Input data for prediction
        model_uri : str, optional
            Model URI (overrides the model name provided in __init__)
        version : str, optional
            Model version to load
        alias : str, default="latest-model"
            Model alias to load
        **kwargs :
            Additional arguments to pass to the feature store client

        Returns
        -------
        DataFrame
            Predictions
        """
        if self.feature_store_client is None:
            raise ValueError("Feature store client not set. Call set_feature_store_client() first.")

        # Build model URI if not provided
        if model_uri is None:
            full_model_name = self._get_full_model_name()

            if version is not None:
                model_uri = f"models:/{full_model_name}/{version}"
            else:
                model_uri = f"models:/{full_model_name}@{alias}"

        logger.info(f"Batch scoring with feature store using model: {model_uri}")

        # Use feature store client to score
        predictions = self.feature_store_client.score_batch(model_uri=model_uri, df=data, **kwargs)

        return predictions
