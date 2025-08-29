import os
import pandas as pd
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from src.datascience.entity.config_entity import (ModelEvaluationConfig)
from src.datascience.utils.common import save_json
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        return accuracy
    
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)

            accuracy = self.eval_metrics(test_y, predicted_qualities)

            score = {"accuracy score": accuracy}
            save_json(path=Path(self.config.metric_file_name), data=score)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metrics(score)

            if tracking_url_type_store!="file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="SVCModel")

            else:
                mlflow.sklearn.log_model(model, "model")