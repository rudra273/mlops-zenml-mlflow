import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy

from typing_extensions import Annotated
from typing import Tuple 


@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"], 
]:
    
    """
    A step that cleans a given DataFrame and divides it into training and test data.

    This step takes a Pandas DataFrame as input and applies the following steps:
    1. Preprocesses the data using the DataPreProcessStrategy, and
    2. Divides the data into training and test data using the DataDivideStrategy.

    The output of this step is a tuple of four elements:
    - X_train: a Pandas DataFrame containing the training data
    - X_test: a Pandas DataFrame containing the test data
    - y_train: a Pandas Series containing the training labels
    - y_test: a Pandas Series containing the test labels
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handel_data()

        devide_strategy = DataDivideStrategy() 
        data_cleaning = DataCleaning(processed_data, devide_strategy) 

        X_train, X_test, y_train, y_test = data_cleaning.handel_data()

        logging.info("cleaned data")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise e

