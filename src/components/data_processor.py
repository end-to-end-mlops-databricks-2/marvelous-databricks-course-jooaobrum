import pandas as pd
from sklearn.model_selection import train_test_split
from components.config import ProjectConfig
from components.data_writer import DataWriter



class DataProcessor:
    """
    A class for processing and preparing data for machine learning tasks.

    This class handles various data preprocessing steps including:
    - Converting feature types (numeric and categorical)
    - Target variable transformation
    - Train-test splitting

    Attributes
    ----------
    data : pd.DataFrame
        The input dataset to be processed
    cfg : ProjectConfig
        Configuration object containing processing parameters
    train : pd.DataFrame, optional
        Training dataset after splitting
    test : pd.DataFrame, optional
        Test dataset after splitting

    """

    def __init__(self, data: pd.DataFrame, cfg: ProjectConfig) -> None:
        """
        Initialize the DataProcessor with input data and configuration.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset to be processed
        cfg : ProjectConfig
            Configuration object containing processing parameters
        """
        self.data = data
        self.cfg = cfg
        self.train = None
        self.test = None

    def _treat_target(self, row: pd.Series) -> int:
        """
        Transform target variable into binary format.

        Parameters
        ----------
        row : pd.Series
            A single row from the DataFrame

        Returns
        -------
        int
            0 for 'Not_Canceled', 1 for 'Canceled'
        """
        return 0 if row[self.cfg.target] == 'Not_Canceled' else 1

    def _treat_num_features(self) -> None:
        """
        Convert numeric features to proper numeric type.
        
        Modifies the DataFrame in place by converting all features specified
        in cfg.num_features to numeric type.
        """
        for feature in self.cfg.num_features:
            self.data[feature] = pd.to_numeric(self.data[feature])

    def _treat_cat_features(self) -> None:
        """
        Convert categorical features to string type.
        
        Modifies the DataFrame in place by converting all features specified
        in cfg.cat_features to string type.
        """
        for feature in self.cfg.cat_features:
            self.data[feature] = self.data[feature].astype(str)

    def pre_processing(self) -> pd.DataFrame:
        """
        Perform all preprocessing steps on the data.

        Steps include:
        1. Transform target variable
        2. Convert numeric features
        3. Convert categorical features

        Returns
        -------
        pd.DataFrame
            The processed DataFrame
        """
        self.data[self.cfg.target] = self.data.apply(self._treat_target, axis=1)
        self._treat_num_features()
        self._treat_cat_features()
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
            self.data, 
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state
        )
        return self.train, self.test
    
    def save_train_test(self, writer: DataWriter) -> None:
        """
        Save training and test datasets to Databricks tables.

        Parameters
        ----------
        writer : DataWriter
            DataWriter object for saving DataFrames
        """
        try:
            writer.save_to_catalog(self.train, "train_data", "overwrite")
            writer.save_to_catalog(self.test, "test_data", "overwrite")
        except Exception as e:
            print(f"Error saving data: {e}")
