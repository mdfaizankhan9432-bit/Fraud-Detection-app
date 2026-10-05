# Fraud-Detection-app
Fraud detection ML model deployed with Streamlit

This is a Machine Learning project that I built to detect whether a financial transaction is **Fraud** or **Legitimate**.

I built this project to understand how Machine Learning can be used in a real-world problem like fraud detection.

## What this project does

The app takes some basic transaction details such as:

* Transaction type
* Transaction amount
* Sender balance
* Receiver balance

and then predicts whether the transaction looks **Fraud** or **Not Fraud**.

The app also shows the **fraud probability** of the transaction.

## Machine Learning Model

I trained and tested different Machine Learning models and selected the model that gave the best overall results for this problem.

The final trained model is saved in the repository as:

`best_fraud_model.joblib`

The model information and required features are stored in:

`model_metadata.json`

## Streamlit App

I used **Streamlit** to turn the Machine Learning model into a simple web application.

The main application file is:

`app.py`

## Files in this repository

* `app.py` → Streamlit application
* `best_fraud_model.joblib` → Trained Machine Learning model
* `model_metadata.json` → Model and feature information
* `requirements.txt` → Required Python libraries

## Why I built this project

I wanted to work on a project where Machine Learning could solve a practical business problem instead of only working with simple datasets.

This project helped me learn about:

* Data cleaning and analysis
* Feature engineering
* Machine Learning model training
* Model evaluation
* Fraud detection
* Saving and loading a trained model
* Building a simple ML web app with Streamlit

## Fraud Detection App link -

https://fraud-detection-app-durv9bpbq7ufxlmf6utwa2.streamlit.app/

## Note

This is a learning and portfolio project built to understand the complete Machine Learning workflow from data to deployment.
