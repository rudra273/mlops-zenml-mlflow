import logging
import pandas as pd
from zenml import step


class IngestData:

    def __init__(self, data_path: str):
        """Initialize the IngestData class.

        Args:
            data_path (str): The path to the data file.
        """
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:

        """Reads data from a given path and returns it as a pandas DataFrame.

        Args:
            data_path (str): The path to the data file.

        Returns:
            pandas.DataFrame: The ingested data.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)  

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """Ingests data from a given path and returns it as a pandas DataFrame.

    Args:
        data_path: The path to the data file.

    Returns:
        A pandas DataFrame containing the ingested data.

    Raises:
        Exception: If there is an error ingesting the data.
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    
    except Exception as e:
        logging.error(f"Error ingesting data: {e}") 
        raise e 
