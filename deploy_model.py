# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 07:22:50 2021

@author: jayamani
"""
#required packages

import joblib
from flask import Flask,request
import numpy as np
import json

app = Flask(__name__)

model=joblib.load(r"D:\jupyter_notebooks\student_data_model")

@app.route("/predict",methods=["POST"])
def predict():
    event=json.loads(request.data)
    values=event["Hours"]
    pre=np.array(values)
    pre=pre.reshape(1,-1)
    res=model.predict(pre)   
    return str(res[0])

if __name__ =="__main__":
    app.run(debug=True)