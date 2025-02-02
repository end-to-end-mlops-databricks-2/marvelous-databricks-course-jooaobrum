import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

from components.config import ProjectConfig


class DataWriter:
    """
    A class for writing DataFrames to Databricks tables.

    Attributes:
    -----------
    config : ProjectConfig
        The project configuration containing catalog and schema names.

    Methods:
    --------
    save_to_catalog(df: DataFrame, table_name: str)
        Saves a DataFrame to a Databricks table with a timestamp column.
    """

    def __init__(self, config: ProjectConfig):
        """
        Initializes the DataWriter with the project configuration.

        Parameters:
        -----------
        config : ProjectConfig
            The configuration object containing catalog and schema details.
        """
        self.config = config

    def save_to_catalog(self, df: DataFrame, table_name: str, mode: str):
        """
        Saves a DataFrame to a Databricks table, adding a timestamp column.

        Parameters:
        -----------
        df : DataFrame
            The DataFrame to be saved.
        table_name : str
            The name of the table to save the data into.
        """
        from pyspark.sql import SparkSession  # Import inside to ensure Databricks Connect is active

        spark = SparkSession.getActiveSession()
        if not spark:
            raise RuntimeError("No active Spark session found. Ensure Databricks Connect is properly configured.")

        # Convert pandas DataFrame to PySpark DataFrame if necessary
        if isinstance(df, pd.DataFrame):
            df = spark.createDataFrame(df)

        df_with_timestamp = df.withColumn("update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))

        full_table_name = f"{self.config.catalog_name}.{self.config.schema_name}.{table_name}"
        df_with_timestamp.write.mode(mode).saveAsTable(full_table_name)

        spark.sql(f"ALTER TABLE {full_table_name} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")

        print(f"Successfully saved data to {full_table_name}")
