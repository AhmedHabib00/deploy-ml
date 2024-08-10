# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
import pandas as pd
from ml.model import train_model, compute_model_metrics, inference
import joblib
import os
# Add the necessary imports for the starter code.
print(os.system("pwd"))
data = pd.read_csv("./data/census.csv")
for col in data.columns:
    data.rename(columns={col: col.lstrip()}, inplace=True)
# Add code to load in the data.

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

# Proces the test data with the process_data function.
model = train_model(X_train, y_train)
predictions = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
# Train and save a model.
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-beta: {fbeta}")
print(f"Best Parameters: {model.get_params()}")
joblib.dump((model, encoder, lb), "model/model.pkl")

with open("metrics.txt", "w") as outfile:
    outfile.write(f"Precision: {precision}\n")
    outfile.write(f"Recall: {recall}\n")
    outfile.write(f"F-beta: {fbeta}\n")
    outfile.write(f"Model Parameters: {model.get_params()}")
