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

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.hotel_reservation_features"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name
        self.tags = tags.dict()

    def create_feature_table(self):
        """
        Create or replace the house_features table and populate it.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (
            Booking_ID STRING NOT NULL,
            no_of_adults BIGINT,
            no_of_children BIGINT,
            no_of_weekend_nights BIGINT,
            no_of_week_nights BIGINT,
            type_of_meal_plan STRING,
            required_car_parking_space BIGINT,
            room_type_reserved STRING,
            lead_time BIGINT,
            arrival_year BIGINT,
            arrival_month BIGINT,
            arrival_date BIGINT,
            market_segment_type STRING,
            repeated_guest BIGINT,
            no_of_previous_cancellations BIGINT,
            no_of_previous_bookings_not_canceled BIGINT,
            avg_price_per_room DOUBLE,
            no_of_special_requests BIGINT,
            update_timestamp_utc TIMESTAMP
        )
        """)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT booking_pk PRIMARY KEY(Booking_ID);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"""INSERT INTO {self.feature_table_name}
            SELECT Booking_ID,
            no_of_adults,
            no_of_children,
            no_of_weekend_nights,
            no_of_week_nights,
            type_of_meal_plan,
            required_car_parking_space,
            room_type_reserved,
            lead_time,
            arrival_year,
            arrival_month,
            arrival_date,
            market_segment_type,
            repeated_guest,
            no_of_previous_cancellations,
            no_of_previous_bookings_not_canceled,
            avg_price_per_room,
            no_of_special_requests,
            update_timestamp_utc
            FROM {self.catalog_name}.{self.schema_name}.train_data"""
        )
        self.spark.sql(
            f"""INSERT INTO {self.feature_table_name}
            SELECT Booking_ID,
            no_of_adults,
            no_of_children,
            no_of_weekend_nights,
            no_of_week_nights,
            type_of_meal_plan,
            required_car_parking_space,
            room_type_reserved,
            lead_time,
            arrival_year,
            arrival_month,
            arrival_date,
            market_segment_type,
            repeated_guest,
            no_of_previous_cancellations,
            no_of_previous_bookings_not_canceled,
            avg_price_per_room,
            no_of_special_requests,
            update_timestamp_utc
            FROM {self.catalog_name}.{self.schema_name}.test_data"""
        )
        logger.info("Feature table created and populated.")

    def load_data(self):
        """
        Load training and testing data from Delta tables.
        """
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_data").select(
            "Booking_ID", "booking_status"
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_data").toPandas()

        logger.info("Data successfully loaded.")

    def feature_engineering(self):
        """
        Perform feature engineering by linking data with feature tables.
        """
        logger.info("Starting feature engineering...")
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=self.cat_features + self.num_features,
                    lookup_key="Booking_ID",
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        self.X_train = self.training_df[self.num_features + self.cat_features]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]

        logger.info("Feature engineering completed.")

    def train(self):
        """
        Train the model and log results to MLflow.
        """
        logger.info("Starting training...")

        self.preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("classifier", LGBMClassifier(**self.parameters))]
        )

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            self.pipeline.fit(self.X_train, self.y_train)
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

            logger.info("Logging model...")
            for metric, value in metrics.items():
                logger.info(f"{metric.capitalize()}: {value}")
                mlflow.log_metric(metric, value)

            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)

            # Log model signature
            signature = infer_signature(model_input=self.X_train, model_output={"probability": 0.351388})

            self.fe.log_model(
                model=self.pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )
        logger.info("Model training and experiment completed.")

    def register_model(self):
        """
        Register the model in UC.
        """
        logger.info("Registering model...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservation_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservation_model_fe",
            alias="latest-model",
            version=latest_version,
        )
        logger.info("Model registered successfully.")

    def load_latest_model_and_predict(self, X):
        """
        Load the trained model from MLflow using Feature Engineering Client and make predictions.
        """
        logger.info("Loading model and making predictions...")
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        logger.info("Predictions completed.")
        return predictions
