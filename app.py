from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homepage():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        try:
            #  reading the inputs given by the user
            fixed_acidity =float(request.form['fixed_acidity'])
            volatile_acidity =float(request.form['volatile_acidity'])
            citric_acid =float(request.form['citric_acid'])
            residual_sugar =float(request.form['residual_sugar'])
            chlorides =float(request.form['chlorides'])
            free_sulfur_dioxide =float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide =float(request.form['total_sulfur_dioxide'])
            density =float(request.form['density'])
            pH =float(request.form['pH'])
            sulphates =float(request.form['sulphates'])
            alcohol =float(request.form['alcohol'])
       
         
            data = [fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]
            data = np.array([data])
            
            obj = PredictionPipeline("SVCModel")
            predict = "GOOD" if obj.predict(data)[0]==1 else "BAD"

            col_names = ["Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar",
                         "Chlorides","Free Sulfur Dioxide","Total Sulfur Dioxide","Density",
                         "pH","Sulphates","Alcohol"]
            
            data_dict = dict(zip(col_names, data[0]))

            return render_template('results.html', data_dict=data_dict, prediction=predict)
        
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	
	app.run(host="0.0.0.0", port = 8080)