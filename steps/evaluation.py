import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import MSE, RMSE, R2
from typing import Tuple
from typing_extensions import Annotated

@step
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
        rmse = RMSE().calulate_scores(y_test, prediction)
        r2 = R2().calulate_scores(y_test, prediction) 

        return rmse, r2 

    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise e

