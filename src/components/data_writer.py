import pandas as pd
from typing import Optional, Dict, Any, Union, List
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.functions import current_timestamp, to_utc_timestamp, lit
from loguru import logger
from components.config import ProjectConfig


class DataWriter:
    """
    A class for writing DataFrames to Databricks tables.
    
    This class handles preparing and writing pandas DataFrames to Databricks
    Delta tables with support for metadata, partitioning, and Delta features.
    
    Attributes:
        config: ProjectConfig containing catalog and schema information
        add_timestamp: Whether to add a timestamp column to tables
        timestamp_column: Name of the timestamp column
        timestamp_timezone: Timezone for the timestamp
        spark: Active SparkSession
    """
    
    def __init__(self, config: ProjectConfig):
        """
        Initialize the DataWriter with project configuration.
        
        Parameters:
            config: ProjectConfig containing catalog and schema details
        """
        self.config = config
        
        # Default settings
        self.add_timestamp = True
        self.timestamp_column = "update_timestamp_utc"
        self.timestamp_timezone = "UTC"
        
        # Initialize Spark session
        logger.info("Initializing DataWriter")
        try:
            self.spark = SparkSession.builder.getOrCreate()
            logger.debug("SparkSession initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SparkSession: {str(e)}")
            raise
    
    def _prepare_dataframe(self, df: pd.DataFrame,
                          add_metadata: bool = True,
                          metadata: Optional[Dict[str, Any]] = None) -> SparkDataFrame:
        """
        Prepare a DataFrame for writing by adding timestamp and metadata.
        
        Parameters:
            df: DataFrame to prepare
            add_metadata: Whether to add metadata columns
            metadata: Dictionary of metadata to add as columns
            
        Returns:
            Prepared Spark DataFrame
        """
        logger.debug(f"Preparing DataFrame with shape {df.shape}")
        
        try:
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(df)
            logger.debug("Converted pandas DataFrame to Spark DataFrame")
            
            # Add timestamp if configured
            if self.add_timestamp:
                logger.debug(f"Adding timestamp column: {self.timestamp_column}")
                spark_df = spark_df.withColumn(
                    self.timestamp_column,
                    to_utc_timestamp(current_timestamp(), self.timestamp_timezone)
                )
            
            # Add metadata columns if requested
            if add_metadata and metadata:
                logger.debug(f"Adding metadata columns: {list(metadata.keys())}")
                for key, value in metadata.items():
                    spark_df = spark_df.withColumn(f"{key}", lit(value))
            
            return spark_df
            
        except Exception as e:
            logger.error(f"Error preparing DataFrame: {str(e)}")
            raise
    
    def save_to_catalog(self, df: pd.DataFrame,
                       table_name: str,
                       mode: str = "overwrite",
                       add_metadata: bool = True,
                       partition_by: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       enable_change_data_feed: bool = True
                       ) -> None:
        """
        Save a DataFrame to a Delta table in the configured catalog and schema.
        
        Parameters:
            df: DataFrame to save
            table_name: Name of the table
            mode: Write mode ('overwrite', 'append', 'ignore', 'error')
            add_metadata: Whether to add metadata columns
            partition_by: List of columns to partition by
            metadata: Dictionary of metadata to add as columns
            enable_change_data_feed: Whether to enable Delta change data feed
        """
        try:
            logger.info(f"Saving DataFrame to table: {table_name}")
            logger.debug(f"Write mode: {mode}, Partitioning: {partition_by or 'None'}")
            
            # Prepare the DataFrame
            prepared_df = self._prepare_dataframe(df, add_metadata, metadata)
            
            # Create full table name
            full_table_name = f"{self.config.catalog_name}.{self.config.schema_name}.{table_name}"
            logger.debug(f"Full table name: {full_table_name}")
            
            # Create a writer object
            writer = prepared_df.write.format("delta").mode(mode)
            
            # Add partitioning if specified
            if partition_by:
                logger.debug(f"Partitioning by: {partition_by}")
                writer = writer.partitionBy(*partition_by)
            
            # Write to table
            logger.info(f"Writing to Delta table: {full_table_name}")
            writer.saveAsTable(full_table_name)
            
            # Set table properties
            properties = {}
            if enable_change_data_feed:
                properties['delta.enableChangeDataFeed'] = 'true'
                logger.debug("Enabling Change Data Feed")
            
            if properties:
                property_str = ", ".join([f"'{k}' = '{v}'" for k, v in properties.items()])
                sql_command = f"ALTER TABLE {full_table_name} SET TBLPROPERTIES ({property_str})"
                logger.debug(f"Setting table properties: {sql_command}")
                self.spark.sql(sql_command)
            
            logger.info(f"Successfully saved data to {full_table_name}")
            
        except Exception as e:
            logger.error(f"Error saving data to {table_name}: {str(e)}")
            raise


    def exists_in_catalog(self, table_name: str) -> bool:
        """
        Check if a table already exists in the configured catalog and schema.
        
        This method queries the Databricks catalog to determine if the specified
        table exists, which is useful for deciding whether to create a new table
        or update an existing one.
        
        Parameters:
            table_name: Name of the table to check
            
        Returns:
            True if the table exists, False otherwise
        """
        try:
            # Construct the fully qualified table name
            full_table_name = f"{self.config.catalog_name}.{self.config.schema_name}.{table_name}"
            logger.debug(f"Checking if table exists: {full_table_name}")
            
            # Query the catalog to see if the table exists
            tables = self.spark.sql(f"SHOW TABLES IN {self.config.catalog_name}.{self.config.schema_name}")
            table_exists = tables.filter(tables.tableName == table_name).count() > 0
            
            if table_exists:
                logger.debug(f"Table {full_table_name} exists")
            else:
                logger.debug(f"Table {full_table_name} does not exist")
                
            return table_exists
            
        except Exception as e:
            logger.error(f"Error checking if table {table_name} exists: {str(e)}")
            return False

    def create_or_update(self, df: pd.DataFrame,
                        table_name: str,
                        add_metadata: bool = True,
                        partition_by: Optional[List[str]] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        enable_change_data_feed: bool = True,
                        update_mode: str = "append"
                        ) -> None:
        """
        Intelligently create a new table or update an existing one based on its presence in the catalog.
        
        This method first checks if the table exists. If it doesn't, it creates a new table.
        If it does exist, it updates the table using the specified update mode.
        
        Parameters:
            df: DataFrame to save
            table_name: Name of the table
            add_metadata: Whether to add metadata columns
            partition_by: List of columns to partition by
            metadata: Dictionary of metadata to add as columns
            enable_change_data_feed: Whether to enable Delta change data feed
            update_mode: Mode to use when updating an existing table ("append" or "overwrite")
        """
        try:
            logger.info(f"Creating or updating table: {table_name}")
            
            # Check if the table exists
            table_exists = self.exists_in_catalog(table_name)
            
            if table_exists:
                # Table exists, update it
                logger.info(f"Table {table_name} exists, updating with mode: {update_mode}")
                self.save_to_catalog(
                    df=df,
                    table_name=table_name,
                    mode=update_mode,
                    add_metadata=add_metadata,
                    partition_by=partition_by,
                    metadata=metadata,
                    enable_change_data_feed=enable_change_data_feed
                )
            else:
                # Table doesn't exist, create it
                logger.info(f"Table {table_name} does not exist, creating new table")
                self.save_to_catalog(
                    df=df,
                    table_name=table_name,
                    mode="overwrite", 
                    add_metadata=add_metadata,
                    partition_by=partition_by,
                    metadata=metadata,
                    enable_change_data_feed=enable_change_data_feed
                )
            
            logger.info(f"Successfully created or updated table: {table_name}")
            
        except Exception as e:
            logger.error(f"Error creating or updating table {table_name}: {str(e)}")
            return False
        
    def drop_table(self, table_name: str, if_exists: bool = True) -> bool:
        """
        Drop a table from the catalog.
        
        This method removes a table and all its data from the catalog.
        It's useful for cleaning up temporary tables or completely 
        replacing a table's structure.
        
        Parameters:
            table_name: Name of the table to drop
            if_exists: Only drop if the table exists (prevents errors)
            
        Returns:
            True if the table was dropped, False if it didn't exist or couldn't be dropped
        """
        try:
            # Construct the fully qualified table name
            full_table_name = f"{self.config.catalog_name}.{self.config.schema_name}.{table_name}"
            
            # Check if the table exists if requested
            if if_exists and not self.exists_in_catalog(table_name):
                logger.info(f"Table {full_table_name} does not exist, nothing to drop")
                return False
                
            # Drop the table
            logger.info(f"Dropping table: {full_table_name}")
            
            # Construct the SQL statement with IF EXISTS clause if requested
            sql = f"DROP TABLE {full_table_name}"
            if if_exists:
                sql = f"DROP TABLE IF EXISTS {full_table_name}"
                
            self.spark.sql(sql)
            logger.info(f"Successfully dropped table: {full_table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error dropping table {table_name}: {str(e)}")
            return False

    def truncate_table(self, table_name: str) -> bool:
        """
        Truncate a table (remove all rows but keep the table structure).
        
        This method removes all data from a table while preserving its schema
        and properties. It's useful when you want to reload all data but keep
        the table definition intact.
        
        Parameters:
            table_name: Name of the table to truncate
            
        Returns:
            True if the table was truncated, False if it doesn't exist or couldn't be truncated
        """
        try:
            # First check if the table exists
            if not self.exists_in_catalog(table_name):
                logger.warning(f"Table {table_name} does not exist, cannot truncate")
                return False
                
            # Construct the fully qualified table name
            full_table_name = f"{self.config.catalog_name}.{self.config.schema_name}.{table_name}"
            logger.info(f"Truncating table: {full_table_name}")
            
            # In Delta Lake, TRUNCATE TABLE is the proper way to remove all rows
            self.spark.sql(f"TRUNCATE TABLE {full_table_name}")
            
            logger.info(f"Successfully truncated table: {full_table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error truncating table {table_name}: {str(e)}")
            return False