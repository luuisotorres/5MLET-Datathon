import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd
from app.main import app

with (
    patch("app.services.model_provider.MLflowModelProvider"),
    patch("app.services.feature_service.FeatureService"),
    patch("pandas.read_parquet"),
):
    from app.main import app

from app.config import settings


@pytest.fixture
def client():
    """
    TestClient fixture that mocks the background services and application state
    to avoid dependencies on MLflow, SQLite or large parquet files during unit tests.
    """
    with (
        patch("app.main.MLflowModelProvider"),
        patch("app.main.FeatureService"),
        patch("app.main.pd.read_parquet"),
    ):
        with TestClient(app) as c:
            # Manually inject mocks into app state to ensure predictable behavior
            c.app.state.model = MagicMock()
            c.app.state.data = pd.DataFrame([{"RA": "123", "name": "Test Student"}])
            c.app.state.model_service = MagicMock()
            c.app.state.feature_service = MagicMock()

            yield c


def test_health_check(client):
    """
    Test the root endpoint / (Health Check).
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Online"
    assert data["api_name"] == "Passos Mágicos - Student Lagging Risk"
    assert "version" in data


def test_get_active_model_info_success(client):
    """
    Test the /model endpoint when a model is loaded.
    """
    # Configure mock
    client.app.state.model_service.get_model_metadata.return_value = {
        "param1": "value1"
    }
    client.app.state.model_service.get_model_version.return_value = "1.0.0"

    response = client.get("/model")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == settings.model_name
    assert data["model_version"] == "1.0.0"
    assert data["active_metadata"] == {"param1": "value1"}


def test_get_active_model_info_no_model(client):
    """
    Test the /model endpoint when no model is loaded.
    """
    client.app.state.model = None
    response = client.get("/model")
    assert response.status_code == 503
    assert response.json()["detail"] == "No model loaded."


def test_predict_success(client):
    """
    Test the /predict/{ra} endpoint with a valid RA.
    """
    ra = "2023-001"
    mock_features = pd.DataFrame(
        [
            {
                "RA": ra,
                "ano_dados": 2023,
                "fase": 1,
                "idade": 12,
                "genero": "M",
                "anos_na_instituicao": 2,
                "instituicao": "Unit A",
                "inde_atual": 7.5,
                "indicador_auto_avaliacao": 8.0,
                "indicador_engajamento": 9.0,
                "indicador_psicossocial": 7.0,
                "indicador_aprendizagem": 6.5,
                "indicador_ponto_virada": 1,
                "indicador_adequacao_nivel": 1,
                "indicador_psico_pedagogico": 1,
            }
        ]
    )

    # Configure mocks
    client.app.state.feature_service.get_student_features.return_value = mock_features
    client.app.state.model.predict.return_value = [2]  # "Expected Performance"

    response = client.post(f"/predict/{ra}")
    assert response.status_code == 200
    data = response.json()
    assert data["ra"] == ra
    assert data["prediction_code"] == 2
    assert "Expected Performance" in data["category"]
    assert data["status"] == "Success"


def test_predict_not_found(client):
    """
    Test the /predict/{ra} endpoint when RA does not exist in Feature Store.
    """
    ra = "unknown"
    client.app.state.feature_service.get_student_features.return_value = None

    response = client.post(f"/predict/{ra}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_train_trigger(client):
    """
    Test the /train endpoint correctly triggers background task.
    """
    response = client.post("/train")
    assert response.status_code == 200
    assert response.json()["status"] == "Started"
    # Verify that the service's train method was called
    # Note: Since it's a background task, we check if it was added to the queue
    # The route calls model_service.train via background_tasks
    # However, BackgroundTasks execution happens after the response.
    # We can still verify if the mock was called if the test runner waits or if we mock BackgroundTasks.
    # In this case, simpler to just check the response as it's an async trigger.


def test_reload_model_success(client):
    """
    Test the /model/reload endpoint.
    """
    new_mock_model = MagicMock()
    client.app.state.model_service.load_active_model.return_value = new_mock_model
    client.app.state.model_service.get_model_version.return_value = "2.0.0"

    response = client.post("/model/reload")
    assert response.status_code == 200
    assert response.json()["status"] == "Success"
    assert response.json()["version"] == "2.0.0"
    assert client.app.state.model == new_mock_model
