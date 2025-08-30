import pandas as pd
import numpy as np
import mlflow

class PredictionPipeline:

    def __init__(self):
        self.model = mlflow.pyfunc.load_model('runs:/a5e2b27a912e4224932a73c788342c34/model')

    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction