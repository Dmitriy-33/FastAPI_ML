import base64

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_without_auth(client):
    response = client.post(
        "/predict",
        json={"Pregnancies": 2, "Glucose": 140, "BMI": 35.5, "Age": 32},
    )
    assert response.status_code == 401


def test_predict_with_auth(client):
    credentials = base64.b64encode(b"admin:secret123").decode()
    response = client.post(
        "/predict",
        json={"Pregnancies": 2, "Glucose": 140, "BMI": 35.5, "Age": 32},
        headers={"Authorization": f"Basic {credentials}"},
    )
    assert response.status_code == 200
    assert response.json()["prediction"] in (0, 1)


def test_predict_validation_error(client):
    credentials = base64.b64encode(b"admin:secret123").decode()
    response = client.post(
        "/predict",
        json={"Pregnancies": -5, "Glucose": 140, "BMI": 35.5, "Age": 32},
        headers={"Authorization": f"Basic {credentials}"},
    )
    assert response.status_code == 422
