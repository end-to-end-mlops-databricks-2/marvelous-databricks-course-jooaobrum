import pandas as pd

from components.config import ProjectConfig


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
        Initializes the DataReader with the specified CSV filename.

        Parameters:
        -----------
        filename : str
            The path to the CSV file.
        """
        self.input_data = config.input_data

    def read_csv(self) -> pd.DataFrame:
        """
        Reads the CSV file and returns its contents as a pandas DataFrame.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the data from the CSV file.
        """
        return pd.read_csv(self.input_data)
