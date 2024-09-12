from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":

    print(Client().active_stack.experiment_tracker.get_tracking_uri()) 
    train_pipeline("/home/sigmoid/Desktop/mlops/mlops-zenml-mlflow/data/olist_customers_dataset.csv") 
    print(f"track experiment - \n mlflow ui --backend-store-uri {Client().active_stack.experiment_tracker.get_tracking_uri()}")



# mlflow ui --backend-store-uri "file:/home/sigmoid/.config/zenml/local_stores/c9ff10f5-4a58-48e8-a271-9fc03c6c15b1/mlruns"

