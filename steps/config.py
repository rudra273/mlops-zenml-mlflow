from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    model_name: str = "linear_regression"
    