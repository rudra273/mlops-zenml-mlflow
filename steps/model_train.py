import logging
import pandas as pd
from zenml import step
from zenml.client import Client
from sklearn.base import RegressorMixin
import mlflow

from src.model_dev import LinearRegressionModel
from .config import ModelNameConfig


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name) 
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig
) -> RegressorMixin:
    """Trains a machine learning model on the given data.

    Args:
        X_train: The training features.
        X_test: The test features.
        y_train: The training labels.
        y_test: The test labels.

    Returns:
        The trained model.
    """
    try:
        model = None
        if config.model_name == "linear_regression":
            mlflow.sklearn.autolog() 
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)

            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} is not supported.") 


    except Exception as e:
        logging.error(f"Error training model: {e}") 
        raise e


