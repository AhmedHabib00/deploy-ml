import json
from fastapi.testclient import TestClient
from main import app, Features

client = TestClient(app)
examples = json.loads(open('examples.json').read())


def test_conn():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the API"}

def test_predict():
    for example in examples:
        response = client.post("/predict", json=example)
        assert response.status_code == 200
        assert response.json() == " <=50K" or response.json() == " >50K"

