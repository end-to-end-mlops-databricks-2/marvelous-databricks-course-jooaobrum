import mlflow
from typing import List, Union
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from loguru import logger

from components.config import ProjectConfig, Tags

class ModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper class to make the trained model compatible with MLflow.
    """
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: Union[pd.DataFrame, np.ndarray]) -> dict:
        """
        Generate probability predictions for the input data.
        
        Args:
            context: MLflow context (not used in this implementation).
            model_input (pd.DataFrame | np.ndarray): Input data.
        
        Returns:
            dict: Dictionary containing predicted probabilities.
        """
        probas = self.model.predict_proba(model_input)[:, 1]
        return {"probability": probas.tolist()}
    
class CustomModel:
    """
    Custom machine learning model for training, evaluating, and logging a LightGBM regressor using MLflow.
    """
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: List[str]):
        """
        Initialize the model with project configuration.
        
        Args:
            config (ProjectConfig): Project configuration object.
            tags (Tags): Tags for MLflow experiment tracking.
            spark (SparkSession): Spark session instance.
            code_paths (List[str]): List of code paths for dependency tracking.
        """
        self.config = config
        self.spark = spark
        
        # Extract settings from config
        self.num_features = config.num_features
        self.cat_features = config.cat_features
        self.target = config.target
        self.parameters = config.parameters
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name
        self.experiment_name = config.experiment_name
        self.tags = tags.dict()
        self.code_paths = code_paths

    def load_data(self):
        """
        Load training and test datasets from Databricks tables.
        """
        logger.info('Loading data from Databricks table...')
        
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_data")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_data").toPandas()
        self.data_version = "0"

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        
        logger.info('Data loaded successfully.')

    def prepare_features(self):
        """
        Define preprocessing pipeline including one-hot encoding for categorical features.
        """
        logger.info('Defining Preprocessing Pipeline...')
        
        self.preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_features)],
            remainder='passthrough'
        )

        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', LGBMRegressor(**self.parameters))
        ])
        
        logger.info("Preprocessing pipeline defined.")
        
    def train_model(self):
        """
        Train the model using the defined preprocessing pipeline.
        """
        logger.info('Training model...')
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info('Model trained successfully.')

    def log_model_experiment(self):
        """
        Log the trained model and its performance metrics to MLflow.
        """
        mlflow.set_experiment(self.experiment_name)
        additional_pip_deps = ["pyspark==3.5.0"]
        for package in self.code_paths:
            whl_name = package.split('/')[-1]
            additional_pip_deps.append(f"code/{whl_name}")

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            y_prob = self.pipeline.predict(self.X_test)
            y_pred = (y_prob > 0.5).astype(int)

            # Evaluate metrics
            metrics = {
                "recall": recall_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred),
                "f1": f1_score(self.y_test, y_pred),
                "roc_auc": roc_auc_score(self.y_test, y_prob),
                "accuracy": accuracy_score(self.y_test, y_pred),
            }

            logger.info('Logging model...')
            for metric, value in metrics.items():
                logger.info(f'{metric.capitalize()}: {value}')
                mlflow.log_metric(metric, value)

            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)

            # Log model signature
            signature = infer_signature(model_input=self.X_train, model_output={'probability': 0.351388})
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_data",
                version=self.data_version
            )
            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            mlflow.pyfunc.log_model(
                python_model=ModelWrapper(self.pipeline),
                artifact_path="pyfunc-lgbm-hotel-reservations",
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature
            )

        logger.info('Model logged successfully.')
