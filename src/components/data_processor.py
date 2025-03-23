import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Callable, Tuple
from sklearn.model_selection import train_test_split
from loguru import logger
from components.config import ProjectConfig
from components.data_writer import DataWriter


class DataProcessor:
    """
    Data processor for ML pipeline development.
    
    Attributes
    ----------
    data : pd.DataFrame
        The input dataset
    config : ProjectConfig
        Configuration object with processing parameters
    train : pd.DataFrame, optional
        Training dataset after splitting
    test : pd.DataFrame, optional
        Test dataset after splitting
    """

    def __init__(self, data: pd.DataFrame, config: ProjectConfig) -> None:
        """
        Initialize with input data and configuration.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset to process
        config : ProjectConfig
            Configuration with processing parameters
        """
        self.data = data.copy()
        self.config = config
        self.train = None
        self.test = None
        
        logger.info(f"DataProcessor initialized with {len(data)} rows and {len(data.columns)} columns")
        
    def validate_columns(self) -> "DataProcessor":
        """
        Check if required columns exist in the dataframe.

        Works with both string features and object features with attributes.
        """
        # Get all required column names
        required_columns = []

        # Extract names from numeric features
        for feature in self.config.num_features:
            if isinstance(feature, dict) or hasattr(feature, 'name'):
                # For dictionary-like objects
                name = feature.get('name') if isinstance(feature, dict) else feature.name
                required_columns.append(name)
            else:
                # For string features
                required_columns.append(feature)

        # Extract names from categorical features
        for feature in self.config.cat_features:
            if isinstance(feature, dict) or hasattr(feature, 'name'):
                # For dictionary-like objects
                name = feature.get('name') if isinstance(feature, dict) else feature.name
                required_columns.append(name)
            else:
                # For string features
                required_columns.append(feature)

        # Add target column
        target = self.config.target
        if isinstance(target, dict) or hasattr(target, 'name'):
            # For dictionary-like target
            name = target.get('name') if isinstance(target, dict) else target.name
            required_columns.append(name)
        else:
            # For string target
            required_columns.append(target)

        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        logger.info("All required columns present")
        return self
        
            
        return self
    
    def rename_columns(self) -> "DataProcessor":
        """
        Rename columns in the DataFrame based on aliases in the configuration.
        
        This method handles both object-style features (with attributes) and 
        dictionary-style features (with keys).
        
        Returns
        -------
        DataProcessor
            Self for method chaining
        """
        # Create a mapping dictionary of original names to aliases
        rename_mapping = {}
        
        # Add numeric features to mapping
        for feature in self.config.num_features:
            if isinstance(feature, dict):
                # Handle dictionary-style features
                if 'name' in feature and 'alias' in feature:
                    rename_mapping[feature['name']] = feature['alias']
            elif hasattr(feature, 'name') and hasattr(feature, 'alias'):
                # Handle object-style features
                rename_mapping[feature.name] = feature.alias
        
        # Add categorical features to mapping
        for feature in self.config.cat_features:
            if isinstance(feature, dict):
                # Handle dictionary-style features
                if 'name' in feature and 'alias' in feature:
                    rename_mapping[feature['name']] = feature['alias']
            elif hasattr(feature, 'name') and hasattr(feature, 'alias'):
                # Handle object-style features
                rename_mapping[feature.name] = feature.alias
        
        # Add target to mapping
        target = self.config.target
        if isinstance(target, dict):
            # Handle dictionary-style target
            if 'name' in target and 'alias' in target:
                rename_mapping[target['name']] = target['alias']
        elif hasattr(target, 'name') and hasattr(target, 'alias'):
            # Handle object-style target
            rename_mapping[target.name] = target.alias
        
    def rename_columns(self) -> "DataProcessor":
        """
        Rename columns in the DataFrame based on aliases in the configuration.
        
        This method handles both object-style features (with attributes) and 
        dictionary-style features (with keys).
        
        Returns
        -------
        DataProcessor
            Self for method chaining
        """
        # Create a mapping dictionary of original names to aliases
        rename_mapping = {}
        
        # Add numeric features to mapping
        for feature in self.config.num_features:
            if isinstance(feature, dict):
                # Handle dictionary-style features
                if 'name' in feature and 'alias' in feature:
                    rename_mapping[feature['name']] = feature['alias']
            elif hasattr(feature, 'name') and hasattr(feature, 'alias'):
                # Handle object-style features
                rename_mapping[feature.name] = feature.alias
        
        # Add categorical features to mapping
        for feature in self.config.cat_features:
            if isinstance(feature, dict):
                # Handle dictionary-style features
                if 'name' in feature and 'alias' in feature:
                    rename_mapping[feature['name']] = feature['alias']
            elif hasattr(feature, 'name') and hasattr(feature, 'alias'):
                # Handle object-style features
                rename_mapping[feature.name] = feature.alias
        
        # Add target to mapping
        target = self.config.target
        if isinstance(target, dict):
            # Handle dictionary-style target
            if 'name' in target and 'alias' in target:
                rename_mapping[target['name']] = target['alias']
        elif hasattr(target, 'name') and hasattr(target, 'alias'):
            # Handle object-style target
            rename_mapping[target.name] = target.alias
        
        # Apply renaming if there are mappings
        if rename_mapping:
            self.data.rename(columns=rename_mapping, inplace=True)
            logger.info(f"Renamed columns: {rename_mapping}")
        else:
            logger.info("No columns to rename")
        
        return self
    
    def convert_datatypes(self) -> "DataProcessor":
        """
        Convert column data types based on configuration.
        """
        # Get all features (numeric + categorical + target)
        features = []
        features.extend(self.config.num_features)
        features.extend(self.config.cat_features)
        features.append(self.config.target)
        
        # Convert each feature
        for feature in features:

            # Access dictionary keys, not attributes
            name = feature['name']
            alias = feature.get('alias', name)  # Use get() to handle missing keys
            dtype = feature['dtype']

            if name == self.config.target['name']:
                logger.info(f"Applying mapping to target column '{alias}'")
                mapping = feature['mapping']
                self.data[alias] = self.data[alias].map(mapping)
                logger.info(f"Target mapped to values: {self.data[alias].value_counts().to_dict()}")
            
            else:
                # Convert if needed
                self.data[alias] = self.data[alias].astype(dtype)
                logger.debug(f"Converted {alias} to {dtype}")
        

        
        logger.info("Data type conversion completed")
        return self
        
    def check_null_values(self, raise_error: bool = True) -> "DataProcessor":
        """
        Check for null values in the DataFrame.
        """
        
        # Get null counts by column
        null_counts = self.data.isnull().sum()
        
        # Filter to only columns with nulls
        columns_with_nulls = null_counts[null_counts > 0.25]
        
        if len(columns_with_nulls) > 0:
            # Calculate percentage of nulls
            null_percentages = 100 * columns_with_nulls / len(self.data)
            
            # Create a report
            report = "Null values report:\n"
            report += "-" * 40 + "\n"
            report += "Column                 Nulls    Percentage\n"
            report += "-" * 40 + "\n"
            
            for col, count in columns_with_nulls.items():
                percentage = null_percentages[col]
                report += f"{col[:20]:<20} {count:>8} {percentage:>12.2f}%\n"
                
            report += "-" * 40 + "\n"
            report += f"Total null values: {columns_with_nulls.sum()}\n"
            
            # Log the report
            logger.warning(report)
            
            # Raise error if requested
            if raise_error:
                raise ValueError("Unexpected null values found. See log for details.")
        else:
            logger.info("No null values found in the dataset.")
            
        return self

    def pre_processing(self) -> pd.DataFrame:
        """
        Perform all preprocessing steps on the data.
        
        Steps include:
        1. Validate required columns
        2. Rename columns based on configuration
        3. Convert data types based on configuration
        4. Apply custom numeric feature transformations
        5. Check for null values

        Observations: You can add any custom transformation steps here.
        
        Parameters
        ----------
        custom_processor : CustomProcessing, optional
            Instance of CustomProcessing class with custom transformation methods
            
        Returns
        -------
        pd.DataFrame
            The processed DataFrame
        """
        logger.info("Starting preprocessing pipeline")
        
        # Validate columns exist
        self.validate_columns()

        # Rename columns based on configuration
        self.rename_columns()
        
        # Convert data types based on configuration
        self.convert_datatypes()

        # Check for null values
        self.check_null_values()
        
        return self.data
        

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the processed data into training and test sets.

        Uses parameters from config:
        - test_size: Proportion of dataset to include in the test split
        - random_state: Random seed for reproducibility

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing (training_data, test_data)
        """
        self.train, self.test = train_test_split(
            self.data, test_size=self.config.test_size, random_state=self.config.random_state
        )
        return self.train, self.test


def generate_synthetic_data(df, drift=False, num_rows=10):
    """
    Generates synthetic data based on the input DataFrame
    """
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if column == "Booking_ID":
            randint = np.random.randint(40000, 99999, num_rows)
            id_col = [f"INS{i}" for i in randint]
            synthetic_data[column] = id_col

        elif pd.api.types.is_string_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )
        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    if drift:
        skew_features = ["no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"]
        for feature in skew_features:
            synthetic_data[feature] = synthetic_data[feature] * 2

        synthetic_data["arrival_year"] = np.random.randint(
            df["arrival_year"].max() + 1, df["arrival_year"].max() + 3, num_rows
        )

    return synthetic_data


