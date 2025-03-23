import pandas as pd
import os
from typing import Optional, Dict, Any, List, Union
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from components.config import ProjectConfig
from loguru import logger


class DataReader:
    """
    A class for reading CSV files using pandas.

    Attributes:
    -----------
    filename : str
        The path to the CSV file.

    Methods:
    --------
    read_csv() -> pd.DataFrame
        Reads the CSV file and returns it as a pandas DataFrame.
    """

    def __init__(self, config: ProjectConfig):
        """
        Initialize the DataReader with project configuration.
        
        Parameters:
            config: ProjectConfig object containing necessary configuration
        """
        self.config = config
        self.input_data = config.input_data
        logger.info(f"Initializing DataReader with input_data: {self.input_data}")
        

    def read_spark_csv(self, filepath: Optional[str] = None, **options) -> SparkDataFrame:
        """
        Read a CSV file into a Spark DataFrame.
        
        This method uses Spark's native CSV reader with configurable options for
        handling various CSV formats and structures.
        
        Parameters:
            filepath: Path to the CSV file (if None, uses config.input_data)
            **options: Additional options to pass to Spark CSV reader
                    (e.g., header=True, inferSchema=True)
        
        Returns:
            Spark DataFrame containing the data
        """
        file_path = filepath or self.input_data
        logger.info(f"Reading CSV file using Spark from: {file_path}")
        
        # Set default options if not provided
        default_options = {
            "header": "true",
            "inferSchema": "true"
        }
        
        # Merge default options with provided options
        all_options = {**default_options, **options}
        logger.debug(f"Spark CSV read options: {all_options}")
        
        try:
            # Use proper session initialization based on environment
            if not hasattr(self, 'spark') or self.spark is None:
                try:
                    # Try to import and use DatabricksSession first (Databricks Connect)
                    from databricks.connect import DatabricksSession
                    self.spark = DatabricksSession.builder.getOrCreate()
                except (ImportError, RuntimeError):
                    # Fall back to regular SparkSession for local development
                    from pyspark.sql import SparkSession
                    self.spark = SparkSession.builder.getOrCreate()
            
            # Read with Spark
            df_spark = self.spark.read.format("csv").options(**all_options).load(file_path)
            
            # Log some basic information about the DataFrame
            num_rows = df_spark.count()
            num_cols = len(df_spark.columns)
            logger.info(f"Successfully read CSV with Spark: {num_rows} rows, {num_cols} columns")
            
            return df_spark
            
        except Exception as e:
            logger.error(f"Error reading CSV file with Spark {file_path}: {str(e)}")
            raise

    def read_pandas_csv(self, filepath: Optional[str] = None, **options) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame.
        
        This method uses pandas' read_csv function with configurable options,
        which may be different from Spark's CSV reader options.
        
        Parameters:
            filepath: Path to the CSV file (if None, uses config.input_data)
            **options: Additional options to pass to pandas.read_csv()
                    (e.g., index_col=0, parse_dates=['date_column'])
        
        Returns:
            pandas DataFrame containing the data
        """
        file_path = filepath or self.input_data
        logger.info(f"Reading CSV file using pandas from: {file_path}")
        
        # Set default options for pandas (different from Spark defaults)
        default_options = {
            "header": 0,  # pandas uses 0-based index for header row
            "dtype": None,  # Let pandas infer types
            "na_values": ["NA", "N/A", "", "null"]  # Common NA values
        }
        
        # Merge default options with provided options
        all_options = {**default_options, **options}
        logger.debug(f"Pandas CSV read options: {all_options}")
        
        try:
            # Read directly with pandas
            df_pandas = pd.read_csv(file_path, **all_options)
            
            # Log basic information
            logger.info(f"Successfully read CSV with pandas: {df_pandas.shape[0]} rows, {df_pandas.shape[1]} columns")
            
            return df_pandas
            
        except Exception as e:
            logger.error(f"Error reading CSV file with pandas {file_path}: {str(e)}")
            raise
        
    
    def read_table(self, table_name: str, as_pandas: bool = False) -> Union[SparkDataFrame, pd.DataFrame]:
        """
        Read a table from the current catalog/schema.
        
        Parameters:
            table_name: Name of the table to read
            as_pandas: If True, returns a pandas DataFrame, otherwise a Spark DataFrame
        
        Returns:
            Spark DataFrame or pandas DataFrame containing the table data
        """
        logger.info(f"Reading table: {table_name}")
        
        self.spark = SparkSession.builder.getOrCreate()

        try:
            df_spark = self.spark.table(table_name)
            logger.info(f"Successfully read table")
            
            if as_pandas:
                logger.debug("Converting Spark DataFrame to pandas DataFrame")
                return df_spark.toPandas()
            return df_spark
            
        except Exception as e:
            logger.error(f"Error reading table {table_name}: {str(e)}")
            raise
    
    def read_sql(self, query: str, as_pandas: bool = False) -> Union[SparkDataFrame, pd.DataFrame]:
        """
        Execute a SQL query and return the results.
        
        Parameters:
            query: SQL query to execute
            as_pandas: If True, returns a pandas DataFrame, otherwise a Spark DataFrame
        
        Returns:
            Spark DataFrame or pandas DataFrame containing the query results
        """
        # Truncate query for logging if it's very long
        log_query = query if len(query) < 500 else query[:500] + "..."
        logger.info(f"Executing SQL query: {log_query}")
        
        self.spark = SparkSession.builder.getOrCreate()

        try:
            df_spark = self.spark.sql(query)
            logger.info(f"Query executed successfully")
            
            if as_pandas:
                logger.debug("Converting Spark DataFrame to pandas DataFrame")
                return df_spark.toPandas()
            return df_spark
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            raise
    
    def read_sql_file(self, sql_file: str, params: Optional[Dict[str, Any]] = None, 
                       as_pandas: bool = False) -> Union[SparkDataFrame, pd.DataFrame]:
        """
        Execute SQL from a file and return results.
        
        Parameters:
            sql_file: Path to the SQL file
            params: Dictionary of parameters to substitute in the SQL
            as_pandas: If True, returns a pandas DataFrame, otherwise a Spark DataFrame
        
        Returns:
            Spark DataFrame or pandas DataFrame containing the query results
        """
        logger.info(f"Reading SQL from file: {sql_file}")
        
        self.spark = SparkSession.builder.getOrCreate()

        try:
            # Read the SQL file
            with open(sql_file, 'r') as f:
                query = f.read()
            
            # Substitute parameters if provided
            if params:
                logger.debug(f"Substituting parameters: {params}")
                for key, value in params.items():
                    # Support both :param and ${param} syntax
                    query = query.replace(f":{key}", str(value))
                    query = query.replace(f"${{{key}}}", str(value))
            
            # Execute the query
            return self.read_sql(query, as_pandas)
            
        except FileNotFoundError:
            logger.error(f"SQL file not found: {sql_file}")
            raise
        except Exception as e:
            logger.error(f"Error processing SQL file {sql_file}: {str(e)}")
            raise
     