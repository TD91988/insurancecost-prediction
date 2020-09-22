# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:14:35 2020

@author: Tushar
"""
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__, template_folder="templates")

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Insurance cost should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(port= 5000, debug=True)