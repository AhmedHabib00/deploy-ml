from starter.ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pytest

@pytest.mark.parametrize('train_model', [train_model])
def test_training(train_model):
    try:
        pytest.model = train_model(pytest.X_train, pytest.y_train)
    except Exception as e:
        raise Exception("Training failed", e)
    assert pytest.model is not None, "Model not trained"
    assert isinstance(pytest.model, RandomForestClassifier), "Model is not a RandomForestClassifier"

    try:
        pytest.model.best_estimator_
    except AttributeError:
        raise AttributeError("Model does not have best_estimator_ attribute")
    

@pytest.mark.parametrize('compute_model_metrics', [compute_model_metrics])
def test_metrics(compute_model_metrics):
    try:
        pytest.precision, pytest.recall, pytest.fbeta = compute_model_metrics(pytest.y_test, pytest.predictions)
    except Exception as e:
        raise Exception("Metrics computation failed", e)
    assert pytest.precision is not None, "Precision not computed"
    assert pytest.recall is not None, "Recall not computed"
    assert pytest.fbeta is not None, "Fbeta not computed"
    assert isinstance(pytest.precision, float), "Precision not a float"
    assert isinstance(pytest.recall, float), "Recall not a float"
    assert isinstance(pytest.fbeta, float), "Fbeta not a float"
    assert pytest.precision >= 0, "Precision is negative"
    assert pytest.recall >= 0, "Recall is negative"
    assert pytest.fbeta >= 0, "Fbeta is negative"
    assert pytest.precision <= 1, "Precision is greater than 1"
    assert pytest.recall <= 1, "Recall is greater than 1"
    assert pytest.fbeta <= 1, "Fbeta is greater than 1"

@pytest.mark.parametrize('inference', [inference])
def test_predictions(inference):
    try:
        pytest.predictions = inference(pytest.model, pytest.X_test)
    except Exception as e:
        raise Exception("Inference failed", e)
    assert pytest.predictions is not None, "Predictions not made"
    assert isinstance(pytest.predictions, np.ndarray), "Predictions not a numpy array"
    assert len(pytest.predictions) == len(pytest.y_test), "Predictions not same length as y_test"

