import logging
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
from sklearn.base import RegressorMixin

from zenml import step
from zenml.client import Client

from src.evaluation import MSE, RMSE, R2

import mlflow


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[
    Annotated[float, "rmse"],
    Annotated[float, "r2"] 
]:
    """Evaluates a machine learning model on a given dataset.

    Args:
        model: The trained machine learning model.
        X_test: The test features.
        y_test: The test labels.

    Returns:
        
    """

    try:
        prediction = model.predict(X_test)

        mse = MSE().calulate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)

        rmse = RMSE().calulate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        r2 = R2().calulate_scores(y_test, prediction) 
        mlflow.log_metric("r2_score", r2) 

        return rmse, r2 

    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise e
