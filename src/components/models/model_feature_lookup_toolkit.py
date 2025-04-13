from loguru import logger
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup


class FeatureStoreWrapper:
    """
    Wrapper class for Databricks Feature Store operations.
    
    This class simplifies working with the Databricks Feature Store by providing
    convenient methods for common operations like creating and updating feature tables,
    performing feature lookups, and integrating with MLflow.
    """
    
    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        table_name: Optional[str] = None
    ):
        """
        Initialize the Feature Store wrapper.
        
        Parameters
        ----------
        spark : SparkSession, optional
            Active Spark session
        catalog : str, optional
            Default catalog to use for feature tables
        schema : str, optional
            Default schema to use for feature tables
        """
       
        # Store configuration
        self.catalog = catalog
        self.schema = schema
        self.table_name = table_name

        # Initialize Spark session if not provided
        self.spark = spark or SparkSession.builder.getOrCreate()
        
        # Initialize Feature Store client
        self.fs = FeatureEngineeringClient()
        
        logger.info(f"Feature Store wrapper initialized. Catalog: {catalog}, Schema: {schema}, Table: {table_name}")
    
    
    def create_feature_table(
        self,
        df: Union[pd.DataFrame, SparkDataFrame],
        primary_keys: List[str],
        table_name: Optional[str] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        description: Optional[str] = None,
        partition_columns: Optional[List[str]] = None,
        mode: str = "overwrite",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Create or update a feature table with the provided data.
        
        Parameters
        ----------
        df : DataFrame or SparkDataFrame
            Data to write to the feature table
        name : str
            Name of the feature table (without catalog/schema)
        primary_keys : List[str]
            Primary key column(s) for the feature table
        description : str, optional
            Description of the feature table
        partition_columns : List[str], optional
            Columns to partition the table by
        schema : str, optional
            Schema name (overrides the default)
        catalog : str, optional
            Catalog name (overrides the default)
        mode : str, default="overwrite"
            Write mode: "merge" or "overwrite"
        tags : Dict[str, str], optional
            Tags to apply to the feature table
        """
        # Determine full feature table name
        schema = schema or self.schema
        catalog = catalog or self.catalog
        table_name = table_name or self.table_name
        
        feature_table_name = f"{catalog}.{schema}.{table_name}"
      
        # Convert pandas DataFrame to Spark if needed
        if isinstance(df, pd.DataFrame):
            spark_df = self.spark.createDataFrame(df)
        else:
            spark_df = df
            
        # Create or update feature table
        logger.info(f"Creating/updating feature table: {feature_table_name}")
        
        # Delete if exists and using overwrite mode
        if mode == "overwrite":
            try:
                logger.info(f"Checking if feature table exists: {feature_table_name}")
                self.fs.get_table(feature_table_name)
                logger.info(f"Deleting existing feature table: {feature_table_name}")
                self.fs.drop_table(name=feature_table_name)
            except Exception as e:
                logger.info(f"Feature table doesn't exist or couldn't be deleted: {e}")
        
        # Create the feature table
        self.fs.create_table(
            name=feature_table_name,
            primary_keys=primary_keys,
            df=spark_df,
            description=description,
            partition_columns=partition_columns,
            tags=tags
        )
        
        logger.info(f"Feature table {feature_table_name} created/updated successfully")
    
    def update_feature_table(
        self,
        df: Union[pd.DataFrame, SparkDataFrame],
        schema: Optional[str] = None,
        catalog: Optional[str] = None,
        table_name: Optional[str] = None
    ) -> None:
        """
        Update an existing feature table with new data.
        
        Parameters
        ----------
        df : DataFrame or SparkDataFrame
            Data to write to the feature table
        name : str
            Name of the feature table (without catalog/schema)
        schema : str, optional
            Schema name (overrides the default)
        catalog : str, optional
            Catalog name (overrides the default)
        mode : str, default="merge"
            Write mode: "merge" or "overwrite"
        """
        # Determine full feature table name
        schema = schema or self.schema
        catalog = catalog or self.catalog
        table_name = table_name or self.table_name
       
        feature_table_name = f"{catalog}.{schema}.{table_name}"
        
        # Convert pandas DataFrame to Spark if needed
        if isinstance(df, pd.DataFrame):
            spark_df = self.spark.createDataFrame(df)
        else:
            spark_df = df
            
        # Write to feature table
        logger.info(f"Updating feature table: {feature_table_name}")
        
        self.fs.write_table(
            name=feature_table_name,
            df=spark_df,
            mode="merge"
        )

        # Count updated records
        count = spark_df.count()            
        logger.info(f"Feature table {feature_table_name} updated successfully with {count} records. ")
    
    

    def create_training_set_with_lookups(
        self,
        df: Union[pd.DataFrame, SparkDataFrame],
        features: List[str],
        lookup_key: Optional[Union[str, List[str]]] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        table_name: Optional[str] = None,
        label: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None,
        timestamp_lookup_key: Optional[str] = None
    ) -> SparkDataFrame:
        """
        Create feature lookups and a training set in a single operation.
        
        Parameters
        ----------
        df : DataFrame or SparkDataFrame
            DataFrame with entity data and optional label
        features : List[str]
            List of feature names to look up from the feature table
        lookup_key : str or List[str], optional
            Column(s) to join with (defaults to table's primary key)
        catalog : str, optional
            Catalog name (overrides the default)
        schema : str, optional
            Schema name (overrides the default)
        table_name : str, optional
            Name of the feature table (without catalog/schema, overrides the default)
        label : str, optional
            Name of the label column in the input df
        exclude_columns : List[str], optional
            Columns to exclude from the result
        timestamp_lookup_key : str, optional
            Timestamp column for point-in-time lookups
            s
        Returns
        -------
        Tuple[SparkDataFrame, Any]
            A tuple containing the training DataFrame and the training set object
        """
        # Determine full feature table name
        schema = schema or self.schema
        catalog = catalog or self.catalog
        table_name = table_name or self.table_name
        

        feature_table_name = f"{catalog}.{schema}.{table_name}"
        
        # Create feature lookups
        feature_lookups = []
        for feature in features:
            # Create feature lookup
            feature_lookup = FeatureLookup(
                table_name=feature_table_name,
                feature_names=[feature],
                lookup_key=lookup_key,
                timestamp_lookup_key=timestamp_lookup_key
            )
            feature_lookups.append(feature_lookup)
        
        logger.info(f"Created {len(feature_lookups)} feature lookups for table {feature_table_name}")
        
        # Convert pandas DataFrame to Spark if needed
        if isinstance(df, pd.DataFrame):
            spark_df = self.spark.createDataFrame(df)
        else:
            spark_df = df
        
        # Create training set
        logger.info(f"Creating training set with {len(feature_lookups)} feature lookups")
        
        training_set = self.fs.create_training_set(
            df=spark_df,
            feature_lookups=feature_lookups,
            label=label,
            exclude_columns=exclude_columns
        )
        
        # Load the training set
        training_df = training_set.load_df()
        logger.info(f"Training set created with {training_df.count()} rows and {len(training_df.columns)} columns")
        
        return training_df, training_set