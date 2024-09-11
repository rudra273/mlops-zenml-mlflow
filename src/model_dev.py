import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """ Abstract class for machine learning models. """

    @abstractmethod
    def train(self, X_train, y_train):
        pass


class LinearRegressionModel(Model):
    """ Linear Regression model. """

    def train(self, X_train, y_train, **kwargs):
        
        try:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info("model trained") 
            return reg

        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise e


