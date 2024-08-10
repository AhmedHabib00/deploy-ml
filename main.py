# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.data import process_data
from starter.ml.model import inference
import pandas as pd
import joblib

class Features(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    salary: int

app = FastAPI()

model, encoder, lb = joblib.load("model/model.pkl")
categorical_features = [x for (x, t) in Features.__annotations__.items() if t == "categorical"]


@app.get("/")
async def read_root():
    return {"message": "Welcome to the API"}

@app.post("/predict")
async def predict(payload: Features):
    data = pd.DataFrame([payload.__dict__,[0]])
    X, _, _, _ = process_data(data, categorical_features=encoder, label="salary", training=False, encoder=encoder, lb=lb)
    prediction = inference(model, X)
    return lb.inverse_transform(prediction)[0]