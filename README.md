### **MLOps Project using ZenML and MLflow**

---

## **1\. Introduction**

This project implements a machine learning pipeline using ZenML for orchestration and MLflow for experiment tracking and model management. The pipeline involves data ingestion, cleaning, model training, and evaluation, all tracked through MLflow for reproducibility and experiment comparison.

The dataset used in this project is the olist\_customers\_dataset.csv, which provides information about customer orders and reviews. The pipeline follows the steps of data cleaning, model development, and evaluation while leveraging the ZenML and MLflow integration for managing the entire workflow.

---

## **2\. ZenML Overview**

ZenML is an open-source MLOps framework designed to simplify the development, deployment, and management of machine learning workflows. It abstracts the complexities of setting up ML pipelines, experiment tracking, model deployment, and allows seamless integration with existing tools and infrastructure such as Kubernetes, Docker, MLflow, and cloud platforms.

### **Key Features:**

* **Pipeline Orchestration**: Allows easy creation and management of ML pipelines.  
* **Reproducibility**: Ensures consistent results with tracked pipeline configurations.  
* **Tool Integration**: Works with popular tools like TensorFlow, scikit-learn, and PyTorch.  
* **Experiment Tracking**: Supports tracking experiments using MLflow.  
* **Extensibility**: Flexible enough to work with local or cloud environments.

---

## **3\. MLflow Overview**

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It covers four main functionalities:

* **Tracking**: Logs and queries experiments for code, data, config, and metrics.  
* **Projects**: Packages code into reusable components.  
* **Models**: Manages models, enabling easy deployment across platforms.  
* **Model Registry**: Facilitates model versioning and lifecycle management.

In this project, MLflow is integrated with ZenML to handle experiment tracking and model logging.

---

## **4\. Project Folder Structure**

The following is the folder structure of the project:

├── data  
│   └── olist\_customers\_dataset.csv  
├── \_\_init\_\_.py  
├── pipelines  
│   ├── training\_pipeline.py  
│   └── utils.py  
├── req.txt  
├── run\_pipeline.py  
├── src  
│   ├── data\_cleaning.py  
│   ├── evaluation.py  
│   └── model\_dev.py  
├── steps  
│   ├── clean\_data.py  
│   ├── config.py  
│   ├── evaluation.py  
│   ├── ingest\_data.py  
│   └── model\_train.py

### **Folder and File Descriptions:**

* **data**: Contains the dataset olist\_customers\_dataset.csv used for training.  
* **pipelines**: Holds the core pipeline scripts.  
  * training\_pipeline.py: The main training pipeline orchestrated by ZenML.  
  * utils.py: Utility functions for testing and handling data within the pipeline.  
* **req.txt**: Contains the required dependencies for the project.  
* **run\_pipeline.py**: Script to run the pipeline and track the experiment via MLflow.  
* **src**: Contains source code for various utilities.  
  * data\_cleaning.py: Handles data preprocessing and splitting.  
  * evaluation.py: Contains different evaluation metrics like RMSE, R2, and MSE.  
  * model\_dev.py: Contains the model development logic for Linear Regression.  
* **steps**: Holds individual steps of the pipeline.  
  * clean\_data.py: Step to clean and prepare the data.  
  * config.py: Step configuration to specify model parameters.  
  * evaluation.py: Step to evaluate the trained model.  
  * ingest\_data.py: Step to ingest the dataset.  
  * model\_train.py: Step to train the machine learning model.

---

## **5\. Pipeline Overview**

### **Pipeline Steps:**

1. **Data Ingestion**:  
   * Reads the data from the specified CSV file.  
   * File: steps/ingest\_data.py  
   * Function: ingest\_df(data\_path)  
2. **Data Cleaning**:  
   * Preprocesses the data by dropping irrelevant columns and handling missing values.  
   * Divides the dataset into training and test sets.  
   * File: steps/clean\_data.py  
   * Function: clean\_df(df)  
3. **Model Training**:  
   * Trains a Linear Regression model using the cleaned data.  
   * The training process is logged with MLflow.  
   * File: steps/model\_train.py  
   * Function: train\_model(X\_train, X\_test, y\_train, y\_test)  
4. **Model Evaluation**:  
   * Evaluates the model using metrics such as RMSE, MSE, and R2.  
   * The evaluation results are logged with MLflow.  
   * File: steps/evaluation.py  
   * Function: evaluate\_model(model, X\_test, y\_test)

---

## **6\. Running the Pipeline**

### **Command to run the pipeline:**

python run\_pipeline.py

This script runs the entire machine learning pipeline, from ingestion to evaluation, while tracking the experiment in MLflow. Once the script finishes, it will print the URI for tracking the experiment.

### **MLflow UI:**

You can view the experiment tracking results in the MLflow UI by running the following command:

mlflow ui \--backend-store-uri "file:/home/sigmoid/.config/zenml/local\_stores/c9ff10f5-4a58-48e8-a271-9fc03c6c15b1/mlruns"

---

## **7\. ZenML and MLflow Integration**

This project integrates MLflow with ZenML for seamless experiment tracking and model logging. Below are the commands used to integrate MLflow:

zenml integration install mlflow \-y  
zenml experiment-tracker register mlflow\_tracker \--flavor=mlflow  
zenml model-deployer register mlflow \--flavor=mlflow  
zenml stack register mlflow\_stack \-a default \-o default \-d mlflow \-e mlflow\_tracker \--set

The integration ensures that each step of the pipeline is tracked and managed using MLflow.

---

## **8\. Setup Guide**

To run the project locally, follow the steps below:

### **Step 1: Clone the repository**

git clone \<repository-url\>  
cd \<repository-folder\>

### **Step 2: Install dependencies**

Install the required dependencies using the req.txt file.

pip install \-r req.txt

### **Step 3: Initialize ZenML**

zenml init

### **Step 4: Install ZenML and MLflow integration**

zenml integration install mlflow \-y  
zenml experiment-tracker register mlflow\_tracker \--flavor=mlflow  
zenml model-deployer register mlflow \--flavor=mlflow  
zenml stack register mlflow\_stack \-a default \-o default \-d mlflow \-e mlflow\_tracker \--set

### **Step 5: Run the pipeline**

python run\_pipeline.py

---

## **9\. Dependencies (requirements.txt)**

zenml==0.41.0  
mlflow==2.3.2  
scikit-learn==1.2.2  
pandas==2.0.3  
numpy==1.24.3  
typing\_extensions==4.5.0

### **Workflow:**

### **1\. Folder Structure Explanation**

The folder structure reflects an organized, modular approach to implementing an end-to-end machine learning pipeline. Each component is decoupled into its own module, allowing for better scalability, maintainability, and reusability. Let's break it down:

#### **Root Level**

* **data/**: This folder contains the raw dataset (`olist_customers_dataset.csv`). It's used as input to the pipeline. By separating data, it becomes easier to track, update, and manage different datasets.  
* **`__init__.py`**: An initialization file, commonly used in Python packages to signal that a directory is a package. This is useful when you need to import modules from this project.  
* **`run_pipeline.py`**: This is the main script that triggers the ZenML pipeline. It imports the pipeline from `pipelines/training_pipeline.py` and executes it with the dataset path as input. It also uses ZenML’s client to track the experiment in MLflow.  
* **`req.txt`**: A file specifying the required Python dependencies for the project. It ensures that the project’s environment can be replicated with the necessary libraries.

#### **`pipelines/`**

* **`training_pipeline.py`**: This is where the pipeline is defined using ZenML's `@pipeline` decorator. It orchestrates the entire machine learning workflow by defining the sequence of steps (data ingestion, cleaning, training, evaluation). Each step is a separate function imported from the `steps/` folder.  
* **`utils.py`**: Contains utility functions (like `get_data_for_test`) used in other parts of the pipeline. This function is useful for data sampling, preprocessing, and preparing data for model evaluation or testing.

#### **`src/`**

This folder holds the source code for different steps in the pipeline.

* **`data_cleaning.py`**: Contains the classes and logic to clean and preprocess the data. It applies strategies for handling missing values, dropping irrelevant columns, and splitting the data into training and testing sets.  
* **`evaluation.py`**: Contains the implementation of different evaluation metrics (MSE, RMSE, R2). These are used after the model is trained to evaluate its performance on the test set.  
* **`model_dev.py`**: Contains the model development code. In this case, it includes a class for `LinearRegressionModel`, which encapsulates the logic for training the machine learning model (Linear Regression).

#### **`steps/`**

This folder contains the ZenML steps. Each step represents a phase of the ML workflow (data ingestion, data cleaning, model training, and evaluation). Each step is annotated with ZenML’s `@step` decorator, allowing it to be orchestrated within the ZenML pipeline.

* **`ingest_data.py`**: Defines a step that ingests data from the specified file path and loads it into a Pandas DataFrame.  
* **`clean_data.py`**: Defines a step that cleans the dataset using the `DataCleaning` class from `src/data_cleaning.py`. It handles missing values, drops unnecessary columns, and splits the data into training and test sets.  
* **`model_train.py`**: Contains the logic for training the model. It uses the `LinearRegressionModel` class from `src/model_dev.py` and automatically tracks the experiment using MLflow through ZenML’s experiment tracker.  
* **`evaluation.py`**: Defines a step that evaluates the trained model on the test set using various metrics (RMSE, MSE, R2).  
* **`config.py`**: Defines the configuration parameters (like `model_name`) for the pipeline. ZenML uses this to configure the pipeline dynamically.

---

### **2\. How ZenML Works in This Workflow**

ZenML is an MLOps framework that simplifies building, orchestrating, and scaling machine learning pipelines. It decouples each step of the machine learning process and allows you to organize them in a pipeline. ZenML helps you focus on the logic of each step without worrying about the orchestration or experiment tracking.

Here’s how it works in this setup:

#### **a. Pipeline Definition**

The machine learning pipeline is defined in `training_pipeline.py` using ZenML’s `@pipeline` decorator. This creates a blueprint for how the pipeline should be structured. It defines four sequential steps:

1. **Data Ingestion**: Loads the data.  
2. **Data Cleaning**: Cleans and preprocesses the data.  
3. **Model Training**: Trains a machine learning model.  
4. **Evaluation**: Evaluates the model's performance.

#### **b. Steps in ZenML**

Each logical block of work (ingestion, cleaning, training, evaluation) is defined as a step. In ZenML, steps are the atomic units of a pipeline. You can cache, modify, or extend them easily.

* Each step is decorated with `@step` in the `steps/` folder.  
* These steps are reusable and can be used in multiple pipelines or different projects with minimal changes.

#### **c. Experiment Tracking with MLflow**

In your code we have integrated MLflow as an experiment tracker with ZenML. This means:

* **MLflow automatically tracks** parameters, metrics, and artifacts (like the trained model).  
* **ZenML orchestrates** the tracking, so every time the pipeline runs, it will log the relevant details of that run to MLflow.  
* **ZenML Stacks** are used to manage the components of your ML workflow, including the experiment tracker. A ZenML stack is a combination of an orchestrator, artifact store, experiment tracker, and model deployer, ensuring all parts of your ML pipeline work together seamlessly. For example:  
  * **Orchestrator**: Manages the execution of the pipeline steps. In your case, this could be Apache Airflow or another tool.  
  * **Artifact Store**: Stores artifacts such as models and datasets. This could be local storage or cloud storage.  
  * **Experiment Tracker**: Logs experiments, metrics, and model versions. You’ve configured MLflow as your experiment tracker using ZenML.  
  * **Model Deployer**: Handles deploying models into production, which might be configured separately or integrated into your stack.

MLflow is configured in the ZenML stack via the command:  
bash  
Copy code  
`zenml experiment-tracker register mlflow_tracker --flavor=mlflow`  
`zenml stack register mlflow_stack -e mlflow_tracker --set`

* 

#### **d. Pipeline Execution**

The pipeline is executed by running the `run_pipeline.py` file. This script:

1. Retrieves the active ZenML stack (which includes the MLflow experiment tracker).  
2. Calls the `train_pipeline` function to trigger the pipeline execution.  
3. Prints the tracking URI where you can view the logged MLflow experiment.

#### **e. Caching and Reproducibility**

ZenML allows caching of steps, so if there is no change in the inputs of a step, it will skip execution for that step in future pipeline runs. This helps in efficient re-runs of pipelines.

---

### **3\. ZenML Workflow Example**

When you run `run_pipeline.py`, here’s the sequence of events that happen:

1. **Ingestion Step**: `ingest_df()` reads the CSV file and loads it as a Pandas DataFrame.  
2. **Cleaning Step**: `clean_df()` preprocesses the data, handling missing values, dropping unnecessary columns, and splitting it into training and testing sets.  
3. **Training Step**: `train_model()` trains a `LinearRegressionModel` on the cleaned data and logs the model parameters and artifacts (like the trained model) to MLflow.  
4. **Evaluation Step**: `evaluate_model()` uses the test data to evaluate the model using metrics like RMSE, R2, and logs the evaluation metrics to MLflow.

This modular approach simplifies the process of building, maintaining, and scaling machine learning projects.

---

### **Summary of Key Benefits**

* **Modularity**: By separating concerns (data ingestion, cleaning, training, etc.) into steps, the codebase becomes modular and easier to maintain.  
* **Reusability**: Each step can be reused in other pipelines or projects, allowing you to build new pipelines quickly.  
* **Trackability**: MLflow integrated with ZenML provides a seamless way to track every run, model, and metric, ensuring reproducibility.  
* **Scalability**: ZenML abstracts the complexity of scaling pipelines across different environments, whether running locally, on a cloud platform, or with container orchestration tools like Kubernetes.

