from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("bank_marketing_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    job = int(request.form['job'])
    marital = int(request.form['marital'])
    education = int(request.form['education'])
    default = int(request.form['default'])
    balance = float(request.form['balance'])
    housing = int(request.form['housing'])
    loan = int(request.form['loan'])
    contact = int(request.form['contact'])
    day = int(request.form['day'])
    month = int(request.form['month'])
    duration = float(request.form['duration'])
    campaign = int(request.form['campaign'])
    pdays = int(request.form['pdays'])
    previous = int(request.form['previous'])
    poutcome = int(request.form['poutcome'])

    features = np.array([[age, job, marital, education, default, balance, housing,
                          loan, contact, day, month, duration, campaign, pdays,
                          previous, poutcome]])

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    if prediction == 1:
        result = "Customer Will Subscribe"
    else:
        result = "Customer Will Not Subscribe"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
