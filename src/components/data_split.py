from typing import Optional, Tuple

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    A class for splitting data into training and testing sets with various strategies.

    This class provides multiple ways to split data, including:
    - Random splitting
    - Stratified splitting (for classification tasks)
    - Group-based splitting (for hierarchical data)
    - Custom splitting using user-defined functions

    Attributes:
        config: Configuration object containing splitting parameters
        data: The data to be split (can be set later)
    """

    def __init__(self, config=None, data: Optional[pd.DataFrame] = None):
        """
        Initialize the DataSplitter with configuration and optional data.

        Parameters:
            config: Configuration object with splitting parameters
            data: Optional DataFrame to split
        """
        self.config = config
        self.data = data
        self.train = None
        self.test = None

        # Set default parameters if not in config
        self.test_size = getattr(config, "test_size", 0.2)
        self.val_size = getattr(config, "val_size", 0.0)  # 0 means no validation set
        self.random_state = getattr(config, "random_state", 42)
        self.target_column = getattr(config, "target", None)

        logger.info(f"DataSplitter initialized with test_size={self.test_size}, val_size={self.val_size}")

    def random_split(self, data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform a basic random split of the data.

        Parameters:
            data: Optional DataFrame to split (uses self.data if not provided)

        Returns:
            Tuple of (training_data, test_data)
        """
        data = data if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for splitting")

        logger.info(f"Performing random split with test_size={self.test_size}")

        self.train, self.test = train_test_split(data, test_size=self.test_size, random_state=self.random_state)

        logger.info(f"Split completed: train={self.train.shape[0]} rows, test={self.test.shape[0]} rows")
        return self.train, self.test

    def stratified_split(
        self, target_column: Optional[str] = None, data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform a stratified split based on the target column.

        Parameters:
            target_column: Name of the column to stratify by
            data: Optional DataFrame to split (uses self.data if not provided)

        Returns:
            Tuple of (training_data, test_data)
        """
        data = data if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for splitting")

        target_column = target_column if target_column is not None else self.target_column
        if target_column is None:
            raise ValueError("No target provided for stratification")

        logger.info(f"Performing stratified split on column '{target_column}' with test_size={self.test_size}")

        # Split with stratification
        self.train, self.test = train_test_split(
            data, test_size=self.test_size, stratify=data[target_column], random_state=self.random_state
        )

        # Verify stratification effectiveness
        train_dist = self.train[target_column].value_counts(normalize=True)
        test_dist = self.test[target_column].value_counts(normalize=True)
        logger.debug(f"Train distribution:\n{train_dist}")
        logger.debug(f"Test distribution:\n{test_dist}")

        logger.info(f"Stratified split completed: train={self.train.shape[0]} rows, test={self.test.shape[0]} rows")
        return self.train, self.test
