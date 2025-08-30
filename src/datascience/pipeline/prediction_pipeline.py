import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv
import os

class PredictionPipeline:

    def __init__(self, model_name: str):
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")

    def predict(self, data):
        load_dotenv()
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        prediction = self.model.predict(data)
        return prediction