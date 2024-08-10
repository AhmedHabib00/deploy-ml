from sklearn.model_selection import train_test_split
from starter.ml.model import train_model, compute_model_metrics, inference
from starter.ml.data import process_data
import pandas as pd
import joblib

categorical_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

data = pd.read_csv("data/census.csv")
train, test = train_test_split(data, test_size=0.20)
model, encoder, lb = joblib.load("model/model.pkl")
X_test, y_test, _, _ = process_data(test, categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb)
predictions = inference(model, X_test)
results = []

for feature in categorical_features:
    for v in test[feature].unique():
        mask = test[feature] == v
        percision, recall, fbeta = compute_model_metrics(y_test[mask], predictions[mask])
        results.append({"feature": feature, "value": v, "percision": percision, "recall": recall, "fbeta": fbeta})

slices = pd.DataFrame(results)
slices.to_csv("slices.txt", index=False)
print(slices.head())