import pandas as pd
import numpy as np
import mlflow

class PredictionPipeline:

    def __init__(self, model_name: str):
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")

    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction