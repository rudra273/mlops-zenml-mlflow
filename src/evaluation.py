import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score 

class Evaluation(ABC):

    @abstractmethod
    def calulate_scores(self, y_true: np.array, y_pred: np.array) -> None:
        pass    


class MSE(Evaluation):
    """Evaluates Strategy thats use Mean Squared Error. """

    def calulate_scores(self, y_true: np.array, y_pred: np.array):

        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse

        except Exception as e:
            logging.error(f"Error calculating MSE: {e}")
            raise e 
            

class R2(Evaluation):
    """Evaluates Strategy thats use R2. """

    def calulate_scores(self, y_true: np.array, y_pred: np.array):

        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2

        except Exception as e:
            logging.error(f"Error calculating R2: {e}")
            raise e 

class RMSE(Evaluation):
    """Evaluates Strategy thats use Root Mean Squared Error. """

    def calulate_scores(self, y_true: np.array, y_pred: np.array):

        try:
            logging.info("Calculating RMSE")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"RMSE: {rmse}")
            return rmse

        except Exception as e:
            logging.error(f"Error calculating RMSE: {e}")
            raise e


