# Fraud-Detection-app

A simple app that checks whether a money transaction is **Fraud** or **Not Fraud**.

**Try the app here:** https://fraud-detection-app-durv9bpbq7ufxlmf6utwa2.streamlit.app/

## About this project

I made this project to learn how a computer can be taught to spot fraud in money transactions. I took the project all the way from raw data to a working web app that anyone can use.

## What the app does

You enter the details of one transaction:

* Transaction type (for example PAYMENT, TRANSFER, CASH_OUT)
* Time of the transaction (in hours since the start of the data)
* Amount
* Sender's balance before and after
* Receiver's balance before and after

The app then tells you if the transaction looks **Fraud** or **Not Fraud**, and also shows how likely it is to be fraud (in percent).

Other things you can do in the app:

* Use the sidebar slider to decide how strict the fraud alert should be
* Upload a CSV file to check many transactions at once
* Click the example buttons to try a normal and a fraud-like transaction

## The data

* About 11,142 mobile money transactions
* Around 1 in 10 of them is fraud

## The model

I tried 6 different models and compared them. The best one was **LightGBM**, so that is the model used in the app.

I tested it on 2,229 transactions that it had never seen before. Out of 228 fraud transactions:

* It caught **225**
* It missed only **3**
* It wrongly flagged **0** genuine transactions

## Files in this repository

* `app.py` → the Streamlit app
* `best_fraud_model.joblib` → the trained model
* `model_metadata.json` → information about the model and data
* `sample_transactions.csv` → sample transactions to try the CSV upload tab
* `requirements.txt` → Python libraries needed to run the app

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What I learned

* Cleaning and understanding data
* Creating useful new columns from existing data
* Training and comparing models
* Checking how good a model really is
* Saving a trained model and using it again
* Building and deploying a simple web app with Streamlit

## Note

This is a learning and portfolio project built to understand the complete Machine Learning workflow from data to deployment.
