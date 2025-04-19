# Databricks notebook source
#!pip install /Volumes/uc_dev/hotel_reservation/samples/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, to_utc_timestamp

from components.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_data").toPandas()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_data").toPandas()

# COMMAND ----------
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# Encode categorical and datetime variables
def preprocess_data(df):
    label_encoders = {}
    for col in df.select_dtypes(include=["object", "datetime"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders


train_set, label_encoders = preprocess_data(train_set)

# Define features and target (adjust columns accordingly)
features = train_set.drop(columns=["booking_status"])
target = train_set["booking_status"]

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(features, target)

# Identify the most important features
feature_importances = pd.DataFrame({"Feature": features.columns, "Importance": model.feature_importances_}).sort_values(
    by="Importance", ascending=False
)

print("Top 5 important features:")
print(feature_importances.head(5))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Synthetic Data

# COMMAND ----------
from components.data_processor import generate_synthetic_data

inference_data_skewed = generate_synthetic_data(train_set, drift=True, num_rows=200)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Tables and Update house_features_online

# COMMAND ----------

inference_data_skewed_spark = spark.createDataFrame(inference_data_skewed).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

inference_data_skewed_spark.write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.inference_data_skewed"
)

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

# write into feature table; update online table
spark.sql(f"""
    INSERT INTO {config.catalog_name}.{config.schema_name}.hotel_reservation_fs
    SELECT *
    FROM {config.catalog_name}.{config.schema_name}.inference_data_skewed
""")

update_response = workspace.pipelines.start_update(pipeline_id=config.pipeline_id, full_refresh=False)
while True:
    update_info = workspace.pipelines.get_update(pipeline_id=config.pipeline_id, update_id=update_response.update_id)
    state = update_info.update.state.value
    if state == "COMPLETED":
        break
    elif state in ["FAILED", "CANCELED"]:
        raise SystemError("Online table failed to update.")
    elif state == "WAITING_FOR_RESOURCES":
        print("Pipeline is waiting for resources.")
    else:
        print(f"Pipeline is in {state} state.")
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send Data to the Endpoint

# COMMAND ----------

import datetime
import itertools

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

from components.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

test_set = (
    spark.table(f"{config.catalog_name}.{config.schema_name}.test_data")
    .withColumn("Booking_ID", col("Booking_ID").cast("string"))
    .toPandas()
)


inference_data_skewed = (
    spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed")
    .withColumn("Booking_ID", col("Booking_ID").cast("string"))
    .toPandas()
)


# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------


import time

import requests
from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

# Required columns for inference
required_columns = [
    "Booking_ID",
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "required_car_parking_space",
    "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
    "type_of_meal_plan",
    "room_type_reserved",
    "market_segment_type",
]

# Sample records from inference datasets
sampled_skewed_records = inference_data_skewed[required_columns].to_dict(orient="records")
test_set_records = test_set[required_columns].to_dict(orient="records")

# COMMAND ----------


# Two different way to send request to the endpoint
# 1. Using https endpoint
def send_request_https(dataframe_record):
    model_serving_endpoint = f"https://{host}/serving-endpoints/hotel-reservation-model-serving-fe/invocations"
    response = requests.post(
        model_serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [dataframe_record]},
    )
    return response


# 2. Using workspace client
def send_request_workspace(dataframe_record):
    response = workspace.serving_endpoints.query(
        name="house-prices-model-serving-fe", dataframe_records=[dataframe_record]
    )
    return response


# COMMAND ----------

# Loop over test records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=20)
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)


# COMMAND ----------

# Loop over skewed records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=30)
for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for skewed data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refresh Monitoring

# COMMAND ----------

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from pyspark.sql.functions import col

from components.config import ProjectConfig
from components.monitoring import create_or_refresh_monitoring

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)
