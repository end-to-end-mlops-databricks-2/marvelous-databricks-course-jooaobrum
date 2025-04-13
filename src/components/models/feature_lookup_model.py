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
        self.num_features  = [feature['alias'] for feature in self.config.num_features]
        self.cat_features  = [feature['alias'] for feature in self.config.cat_features]
        self.target = self.config.target['alias']
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
        self.model_name = self.config.model_name
        self.tags = tags.dict()

        self.fs_wrapper = FeatureStoreWrapper(
            catalog=self.catalog_name,
            schema=self.schema_name,
            table_name=self.feature_table_name,
            spark=self.spark)
        
        self.mlflow_client = MLflowToolkit(
            experiment_name=self.experiment_name,
            model_name=self.model_name,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,  
            tags=self.tags
        )

    def create_feature_table(self):
        """
        Create or replace the feature table and populate it.
        """

        train_data = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_data")
        test_data = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_data")

        source_df = train_data.unionByName(test_data, allowMissingColumns=True)
        
        # Create Feature table
        self.fs_wrapper.create_feature_table(df = source_df, primary_keys = self.primary_keys)

    def load_data(self):
        """
        Load training and testing data from Delta tables.
        """
        # Get primary key and target column from config
        primary_keys = self.primary_keys
        target_column = self.target

        # Load only ID and target columns from training data
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_data").select(
            *primary_keys, target_column
        )
        
        # Load full test data
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_data").toPandas()

        logger.info(f"Data successfully loaded. Train rows: {self.train_set.count()}, Test rows: {len(self.test_set)}")
        
        return self.train_set, self.test_set

    def retrieve_features(self):
        """
        Perform feature engineering by linking data with feature tables.
        """
        logger.info("Starting feature engineering...")
        
        self.training_df, self.training_set = self.fs_wrapper.create_training_set_with_lookups(
                                                                            df=self.train_set,
                                                                            features=self.num_features + self.cat_features,
                                                                            lookup_key=self.primary_keys,
                                                                            label=self.target
                                                                            )
        

        

        self.training_df = self.training_df.toPandas()
        self.X_train = self.training_df[self.num_features + self.cat_features]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]

        logger.info("Feature retrieval completed.")

    def train(self):
        logger.info("Starting model training...")
        
        # Get preprocessing strategies from config
        self.numeric_strategy = self.config.numeric_strategy
        self.categorical_strategy = self.config.categorical_strategy
        self.missing_strategy = self.config.missing_strategy
        
        
        # 1. Build the preprocessor
        preprocessor = Preprocessor().create(
            numeric_features = self.num_features,
            categorical_features = self.cat_features,
            numeric_strategy = self.numeric_strategy,
            categorical_strategy = self.categorical_strategy,
            missing_strategy= self.missing_strategy
        )

        # 2. Create model
        model = ModelFactory.create(
            model_name="lightgbm",
            task="classification",
            **self.parameters
        )

        # 3. Create pipeline with preprocessor and model
        self.pipeline = MLPipeline.create(
            preprocessor=preprocessor,
            model=model
        )

        run_id = self.mlflow_client.start_run()
        self.mlflow_client.log_params(self.parameters)

        # 5. Train the model
        self.pipeline .fit(self.X_train, self.y_train)
        logger.info("Model training completed.")

        # 6. Evaluate the model
        y_pred = self.pipeline .predict(self.X_test)
        y_prob = self.pipeline .predict_proba(self.X_test)[:, 1]
        
        # 7. Log metrics
        metrics = ModelEvaluation.classification_metrics(
                                                            y_true = self.y_test,
                                                            y_pred = y_pred,
                                                            y_prob = y_prob
                                                        )

        self.mlflow_client.log_metrics(metrics)

        # 8. Log the model
        self.mlflow_client.log_model(
                                    model=self.pipeline,
                                    artifact_path=self.model_name,
                                    flavor=mlflow.sklearn,
                                    training_set=self.training_set,
                                    feature_names=self.num_features + self.cat_features
                                )
        

        # 9. End run
        self.mlflow_client.end_run()
        logger.info("Model training and logging completed.")
        return self.pipeline       
     

    def update_feature_table(self):
        """
        Update the feature table with the latest data.
        """
        logger.info(f"Updating feature table {self.feature_table_name}...")
        
        train_data = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_data")
        test_data = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_data")
        
        # Combine new data
        source_df = train_data.unionByName(test_data, allowMissingColumns=True)
        
        # Update Feature table
        self.fs_wrapper.update_feature_table(df = source_df)
        
        return None
    
    def model_improved(self) -> bool:
        """
        Evaluate the model's performance on the test set.
        
        Parameters
        ----------
        test_set : pd.DataFrame
            Test set to evaluate the model on
        """     

        # Check if latest model exists
        # If so, load it and evaluate
        # Compare with the current model
        # If the new model is better, return True
        # If the latest model doesnt exist, register the model


        test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_data")
        test_set_df = test_set.toPandas()
        logger.info("Evaluating model...")

        try:
            predictions_latest_model = self.mlflow_client.batch_score_with_feature_store(data=test_set.select(self.primary_keys + self.num_features + self.cat_features))
            predictions_current_model = self.pipeline.predict(test_set_df[self.num_features + self.cat_features])

            # Calculate metrics for both models
            metrics_latest_model = ModelEvaluation.classification_metrics(
                y_true=test_set_df[self.target],
                y_pred=predictions_latest_model.select('prediction').toPandas().values.flatten(),
            )
            metrics_current_model = ModelEvaluation.classification_metrics(
                y_true=test_set_df[self.target],
                y_pred=predictions_current_model,
            )
            logger.info(f"Latest model metrics: {metrics_latest_model}")
            logger.info(f"Current model metrics: {metrics_current_model}")
            # Compare metrics
            model_improved = False

            if metrics_current_model["f1"] > metrics_latest_model["f1"]:
                model_improved = True
                logger.info("Current model is better than the latest model.")
            else:
                logger.info("Current model is not better than the latest model.")
            return model_improved
        
        except Exception as e:
            # Check if error indicates model doesn't exist
            if "does not exist" in str(e):
                logger.info(f"Model does not exist. Condition set to register it for the first time.")
                model_improved = True
                return model_improved


    def register_model(self):
        """
        Register the model in the MLflow Model Registry.
        """
        logger.info("Registering model...")
        
        # Get the latest version of the model
        latest_version = self.mlflow_client.register_model(artifact_path=self.model_name)
        
        logger.info(f"Model registered successfully with version: {latest_version}")
        
        return latest_version


 