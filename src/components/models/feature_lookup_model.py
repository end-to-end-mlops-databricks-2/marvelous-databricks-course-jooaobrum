from typing import Union

import mlflow
import numpy as np
import pandas as pd
from databricks import feature_engineering  
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from components.config import ProjectConfig, Tags

from components.models.model_factory import (
    Preprocessor,
    ModelFactory,
    MLPipeline
)

from components.models.mlflow_toolkit import MLflowToolkit
from components.models.model_feature_lookup_toolkit import FeatureStoreWrapper
from components.models.model_metrics import ModelEvaluation


class FeatureLookUpModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        """
        Initialize the model with project configuration.
        """
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.feature_table_name = self.config.feature_table_name

        # Define table names and function name
        self.full_feature_table_name = f"{self.catalog_name}.{self.schema_name}.{self.feature_table_name}"

        # Define primary key
        self.primary_keys = self.config.primary_keys

        # MLflow configuration
        self.experiment_name = self.config.experiment_name
        self.tags = tags.dict()

        self.fs_wrapper = FeatureStoreWrapper(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            feature_table_name=self.feature_table_name,
            spark=self.spark)

    def create_feature_table(self):
        """
        Create or replace the feature table and populate it.
        """

        train_data = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_data")
        test_data = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_data")

        source_df = train_data.unionByName(test_data, allowMissingColumns=True)
        
        # Create Feature table
        self.fs_wrapper.create_table(df = source_df, primary_keys = self.primary_keys)

    def load_data(self):
        """
        Load training and testing data from Delta tables.
        """
        # Get primary key and target column from config
        primary_keys = self.primary_keys
        target_column = self.target
        
        # Load only ID and target columns from training data
        self.train_set = self.spark.table(self.train_table).select(
            primary_keys + [target_column]
        )
        
        # Load full test data
        self.test_set = self.spark.table(self.test_table).toPandas()

        logger.info(f"Data successfully loaded. Train rows: {self.train_set.count()}, Test rows: {len(self.test_set)}")
        
        return self.train_set, self.test_set

    def retrieve_features(self):
        """
        Perform feature engineering by linking data with feature tables.
        """
        logger.info("Starting feature engineering...")
        
        self.training_df = self.fs_wrapper.create_training_set_with_lookups(
                                                                            df=self.train_set,
                                                                            features=self.num_features + self.cat_features,
                                                                            target=self.target)
        

        self.training_df = self.training_df.toPandas()
        self.X_train = self.training_df[self.num_features + self.cat_features]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]

        logger.info("Feature retrieval completed.")

    def train(self):
        logger.info("Starting model training...")
        
        # Get preprocessing strategies from config
        numeric_strategy = getattr(self.config, 'numeric_strategy', 'standard')
        categorical_strategy = getattr(self.config, 'categorical_strategy', 'onehot')
        missing_strategy = getattr(self.config, 'missing_strategy', 'mean')
        
        # Get feature names
        num_feature_names = [f['alias'] for f in self.config.num_features]
        cat_feature_names = [f['alias'] for f in self.config.cat_features]
        
        # 1. Build the preprocessor
        self.preprocessor = build_preprocessor(
            numeric_features=num_feature_names,
            categorical_features=cat_feature_names,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            missing_strategy=missing_strategy
        )
        
        # Get model type from config
        model_type = getattr(self.config, 'model_type', 'lightgbm')
        task = getattr(self.config, 'task', 'classification')
        resampling_strategy = getattr(self.config, 'resampling_strategy', None)
        resampling_params = getattr(self.config, 'resampling_params', None)
        
        # 2. Create the pipeline
        self.pipeline = create_pipeline(
            preprocessor=self.preprocessor,
            model_type=model_type,
            task=task,
            model_params=self.parameters,
            resampling_strategy=resampling_strategy,
            resampling_params=resampling_params
        )
        
        # 3. Train the model with MLflow tracking
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            
            # Train using our toolkit function
            model_pipeline, metrics = train_model(
                pipeline=self.pipeline,
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_test,
                y_val=self.y_test,
                track_with_mlflow=True,  # Already in MLflow run, so it will use this run
                log_params=True
            )
            
            # Store the trained pipeline and metrics
            self.pipeline = model_pipeline
            self.metrics = metrics
            
            # Log model input/output signature
            model_output = {"probability": 0.5} if task == "classification" else {"prediction": 0.0}
            signature = infer_signature(model_input=self.X_train, model_output=model_output)
            
            # Get artifact path from config
            artifact_path = getattr(self.config, 'artifact_path', 'model')
            
            # Log the model with feature store
            self.fe.log_model(
                model=self.pipeline,
                flavor=mlflow.sklearn,
                artifact_path=artifact_path,
                training_set=self.training_set,
                signature=signature,
            )
            
        logger.info(f"Model training completed with metrics: {self.metrics}")
        return self.metrics


    def update_feature_table(self):
        """
        Update the feature table with the latest data.
        """
        logger.info(f"Updating feature table {self.feature_table_name}...")
        
        train_data = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_data")
        test_data = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_data")
        
        # Combine new data
        source_df = train_data.unionByName(test_data, allowMissingColumns=True)
        
        # Write to feature table in merge mode
        self.fe.write_table(
            name=f"{self.catalog_name}.{self.schema_name}.{self.feature_table_name}",
            df=source_df,
            mode="merge"
        )
        
        # Count updated records
        count = source_df.count()
        logger.info(f"Feature table updated with {count} records.")
        
        return count

 